"""Command-line entry point for CodePilot's GitHub issue coding loop."""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import re
import shlex
import signal
import subprocess
import tempfile
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from pathlib import Path
from threading import Condition, Event, Lock, Thread, get_ident
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

from codepilot.agents.classifier import classify_issue
from codepilot.agents.coder_models import CoderStatus
from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.config.bootstrap import create_orchestrator
from codepilot.config.settings import ConfigurationError, Settings
from codepilot.github.client import GitHubClient
from codepilot.github.models import GitHubIssue
from codepilot.github.polling import GitHubPollingService
from codepilot.github.pr_workflow import (
    HumanApprovalRejectedError,
    MergeConflictError,
    PullRequestWorkflow,
    build_structured_commit_message,
)
from codepilot.graph.lifecycle import TaskStatus, transition_task
from codepilot.guardrails import CodePilotGuardrails, get_guardrails
from codepilot.memory.session_summary import (
    SessionTracker,
    load_recent_summaries,
    recently_failed_issue_numbers,
)
from codepilot.sandbox.execution_models import CommandStatus
from codepilot.sandbox.executor import SandboxExecutor
from codepilot.sandbox.models import SecurityDecision
from codepilot.sandbox.policy import SandboxPolicy
from codepilot.skills.skill_registry import get_skill

logger = logging.getLogger("codepilot.runner")


class _TUILogForwardingHandler(logging.Handler):
    """Send application/library log records to the Textual log panel, not stderr."""

    _FORWARDED_PREFIXES = (
        "codepilot",
        "deepagents",
        "langchain",
        "langgraph",
        "openai",
        "github",
        "httpx",
        "httpcore",
    )

    def __init__(self, runtime: Any) -> None:
        super().__init__(level=logging.INFO)
        self._runtime = runtime
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith(self._FORWARDED_PREFIXES):
            return
        try:
            self._runtime._post_ui(
                "append_log",
                self.format(record),
                source=record.name.rsplit(".", 1)[-1],
            )
        except Exception:
            # A logger must never write a diagnostic traceback to stderr while
            # Textual owns the terminal. Failed UI forwarding is best-effort.
            return


def _normalize_tool_names(tools: list[Any]) -> list[Any]:
    """Return tools with unique names accepted by OpenAI function calling."""
    normalized: list[Any] = []
    used_names: set[str] = set()
    for tool in tools:
        original_name = getattr(tool, "name", None)
        if not isinstance(original_name, str) or not original_name.strip():
            normalized.append(tool)
            continue
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", original_name.strip())
        base_name = name or "tool"
        name = base_name
        suffix = 2
        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(name)
        if name == original_name:
            normalized.append(tool)
        elif callable(getattr(tool, "model_copy", None)):
            normalized.append(tool.model_copy(update={"name": name}))
        elif callable(getattr(tool, "copy", None)):
            normalized.append(tool.copy(update={"name": name}))
        else:
            cloned = copy(tool)
            cloned.name = name
            normalized.append(cloned)
    return normalized


def _request_approval(operation: str, details: Mapping[str, Any]) -> bool:
    """Ask for explicit interactive approval before a GitHub write operation."""
    print("\nCodePilot requires human approval.")
    print(f"Operation: {operation}")
    for key, value in details.items():
        print(f"  {key}: {value}")
    try:
        answer = input("Approve this operation? Type 'yes' to continue: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("Approval not granted.")
        return False
    return answer == "yes"


def _category_name_for(issue: Any) -> str:
    """Return the task classification used for session memory."""
    try:
        return classify_issue(issue).value
    except Exception:
        return "unknown"


def _skill_name_for(issue: Any) -> str:
    """Return the skill the Orchestrator selects for this issue (display only)."""
    try:
        return get_skill(classify_issue(issue)).name
    except Exception:
        return "—"


class GuardrailBlockedError(RuntimeError):
    """Raised when the NeMo input rail blocks a task before it reaches the LLM."""


def _check_guardrail_command(
    command_text: str,
    *,
    approval_callback: Callable[[str, Mapping[str, Any]], bool] | None = None,
    guardrails: CodePilotGuardrails | None = None,
) -> str:
    """Run the sandbox policy check and append the NeMo input-rail verdict."""
    result = _check_sandbox_policy_command(command_text, approval_callback=approval_callback)
    if guardrails is None or not command_text.strip():
        return result
    return f"{result}\n{guardrails.check_input(command_text).describe()}"


def _check_sandbox_policy_command(
    command_text: str,
    *,
    approval_callback: Callable[[str, Mapping[str, Any]], bool] | None = None,
) -> str:
    """Evaluate one command against the sandbox policy without running commands.

    This is a deliberate check-only pathway for safely demonstrating guardrails.
    Prohibited commands are confirmed through SandboxExecutor's rejection path.
    Commands requiring approval can surface the existing HITL prompt, but approval
    in this preview never launches the command.
    """
    if not command_text.strip():
        return "BLOCKED: enter a command to check. No subprocess was launched."

    try:
        arguments = shlex.split(command_text)
    except ValueError as exc:
        return f"GUARDRAIL BLOCKED: invalid command syntax: {exc}. No subprocess was launched."

    if not arguments:
        return "BLOCKED: enter a command to check. No subprocess was launched."

    with tempfile.TemporaryDirectory(prefix="codepilot-guardrail-preview-") as sandbox_root:
        executor = SandboxExecutor(SandboxPolicy(sandbox_root))
        policy_result = executor.policy.check_command(
            arguments,
            working_directory=sandbox_root,
        )
        rendered_command = shlex.join(arguments)
        details = {
            "command": list(arguments),
            "working_directory": sandbox_root,
            "reason": policy_result.reason,
            "mode": "check-only; the command will not be executed",
        }

        if policy_result.decision == SecurityDecision.BLOCK:
            result = executor.execute(arguments, working_directory=sandbox_root)
            if result.status != CommandStatus.BLOCKED:
                return (
                    "GUARDRAIL CHECK ERROR: policy and executor decisions disagreed. "
                    "No further action was taken."
                )
            return (
                "GUARDRAIL BLOCKED\n"
                f"Command: {rendered_command}\n"
                f"Reason: {result.reason}\n"
                "No subprocess was launched."
            )

        if policy_result.decision == SecurityDecision.REQUIRES_APPROVAL:
            result = executor.execute(arguments, working_directory=sandbox_root)
            if result.status != CommandStatus.REQUIRES_APPROVAL:
                return (
                    "GUARDRAIL CHECK ERROR: expected an approval-required result. "
                    "No further action was taken."
                )
            if approval_callback is None:
                return (
                    "APPROVAL REQUIRED\n"
                    f"Command: {rendered_command}\n"
                    f"Reason: {policy_result.reason}\n"
                    "No command was executed."
                )
            try:
                approved = approval_callback("execute_command_requires_approval", details) is True
            except Exception as exc:
                return (
                    "APPROVAL UNAVAILABLE\n"
                    f"Command: {rendered_command}\n"
                    f"Reason: {type(exc).__name__}: {exc}\n"
                    "No command was executed."
                )
            if approved:
                return (
                    "APPROVAL GRANTED FOR INSPECTION ONLY\n"
                    f"Command: {rendered_command}\n"
                    "This guardrail check never executes commands; the command was not executed."
                )
            return (
                "APPROVAL REJECTED OR UNAVAILABLE\n"
                f"Command: {rendered_command}\n"
                "No command was executed."
            )

        return (
            "POLICY ALLOWS COMMAND\n"
            f"Command: {rendered_command}\n"
            "This is a check-only interaction; the command was not executed."
        )


def _clone_repository(settings: Settings, destination: Path) -> None:
    """Clone the configured repository without putting the token in command args."""
    if not settings.github_token:
        raise ConfigurationError("GITHUB_TOKEN is required to clone the configured repository.")
    owner = settings.github_repository_owner
    name = settings.github_repository_name
    if not owner or not name:
        raise ConfigurationError("GitHub repository owner and name must be configured.")

    # Git reads this header from its process environment; the credential is not
    # included in the command line and the clone URL remains free of secrets.
    credential = base64.b64encode(f"x-access-token:{settings.github_token}".encode()).decode()
    clone_env = os.environ.copy()
    clone_env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {credential}",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    url = f"https://github.com/{owner}/{name}.git"
    command = [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        settings.github_default_branch,
        url,
        str(destination),
    ]
    logger.info("Cloning %s/%s for issue processing.", owner, name)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=clone_env,
        timeout=180,
    )
    if completed.returncode != 0:
        # Do not log command environment or any credential-bearing values.
        detail = completed.stderr.strip()[-2000:] or "git clone failed without diagnostic output"
        raise RuntimeError(f"Could not clone {owner}/{name}: {detail}")


def _print_coder_result(issue_number: int, result: Any) -> None:
    """Report the actual coding and test outcome without overstating success."""
    print(f"\n--- CodePilot result for issue #{issue_number} ---")
    print(f"Status: {result.status.value}")
    print(f"Attempts: {result.attempts}")
    if result.plan:
        print("\nImplementation plan:\n")
        print(result.plan)
    if result.test_result is not None:
        print(
            f"\nTests: {'PASSED' if result.test_result.succeeded else 'FAILED'} "
            f"(exit code {result.test_result.return_code})"
        )
    if result.diff_path is not None:
        print(f"Proposed diff: {result.diff_path}")
    if result.error:
        print(f"Error: {result.error}")


class CodePilotRuntime:
    """Coordinate polling, coding workflows, approval gates, and optional TUI updates."""

    # Direct tasks need a positive workspace identity because the existing
    # CoderWorkspace is issue-number based. These IDs are internal only: they
    # are never used to create GitHub branches or claim that a GitHub issue exists.
    DIRECT_TASK_ID_START = 2_000_000_000

    def __init__(
        self,
        settings: Settings,
        *,
        stop_event: Event | None = None,
        interactive: bool = False,
    ) -> None:
        self.settings = settings
        self.stop_event = stop_event or Event()
        self.interactive = interactive
        self._app: Any | None = None
        self._ui_thread_id: int | None = None
        self._task_lock = Lock()
        self._state_lock = Lock()
        self._queue_condition = Condition()
        self._pending_issues: deque[GitHubIssue] = deque()
        self._pending_issue_numbers: set[int] = set()
        self._attempted_issues: set[int] = set()
        self._skipped_issue_numbers: set[int] = set()
        self._issue_statuses: dict[int, str] = {}
        self._latest_open_issues: tuple[GitHubIssue, ...] = ()
        self._latest_claimed_issue_numbers: frozenset[int] = frozenset()
        self._direct_task_number = self.DIRECT_TASK_ID_START
        # Skill selected for the task currently shown in the Active Task panel.
        self._current_skill = "—"
        self._started = False

        self.client = GitHubClient(settings)
        # NeMo Guardrails input rail: screens issue/direct-task text before the LLM.
        self.guardrails = get_guardrails()
        self._approval_callback = self._request_approval
        self.pr_workflow = PullRequestWorkflow(
            github_client=self.client,
            default_branch=settings.github_default_branch,
            approval_callback=self._approval_callback,
        )

        # Keep working copies outside the CodePilot checkout to avoid recursively
        # copying .codepilot workspaces into themselves.
        self.workspace_root = (
            Path.cwd().parent / f".{Path.cwd().name}-codepilot-workspaces"
        ).resolve()
        self.orchestrator: CodePilotOrchestrator = create_orchestrator(
            # LangChain GitHubToolkit, scoped to read-only tools for the root LLM.
            tools=self.client.orchestrator_tools,
            workspace_root=self.workspace_root,
            pr_workflow=self.pr_workflow,
            repository=(f"{settings.github_repository_owner}/{settings.github_repository_name}"),
            approval_callback=self._approval_callback,
            repo_map_token_budget=settings.repo_map_token_budget,
            repo_map_top_k=settings.repo_map_top_k,
        )

        # Episodic memory across sessions (Component 5): read the last three
        # session summaries now; write this session's summary on stop().
        repository_name = f"{settings.github_repository_owner}/{settings.github_repository_name}"
        memory_directory = Path(
            os.getenv("CODEPILOT_MEMORY_DIR") or (Path.cwd() / ".codepilot" / "memory")
        )
        self.session_tracker = SessionTracker(
            repository=repository_name, memory_directory=memory_directory
        )
        self.recent_session_summaries = load_recent_summaries(
            memory_directory, repository=repository_name
        )
        self.recently_failed_issues = recently_failed_issue_numbers(
            self.recent_session_summaries
        )

        polling_options: dict[str, Any] = {
            "merge_monitor": self.orchestrator.merge_monitor,
        }
        if interactive:
            polling_options["on_issues"] = self._on_issues_snapshot
        self.polling = GitHubPollingService(self.client, settings, **polling_options)

        self._polling_thread: Thread | None = None
        self._processor_thread: Thread | None = None

    def attach_app(self, app: Any) -> None:
        """Attach the Textual app after construction and record its event-loop thread."""
        self._app = app
        self._ui_thread_id = get_ident()

    def _post_ui(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """Dispatch a presentation update safely to the Textual application thread."""
        app = self._app
        if app is None:
            return
        callback = getattr(app, method_name)
        try:
            if get_ident() == self._ui_thread_id:
                callback(*args, **kwargs)
            else:
                app.call_from_thread(callback, *args, **kwargs)
        except RuntimeError:
            # The app may already be shutting down. Runtime work must not fail
            # merely because a log or status update can no longer be displayed.
            logger.debug("Skipping TUI update during shutdown: %s", method_name)

    def _log(self, message: str, *, source: str = "Runtime") -> None:
        if self.interactive:
            self._post_ui("append_log", message, source=source)
        else:
            logger.info("[%s] %s", source, message)

    def _update_task(
        self,
        title: str,
        status: str,
        *,
        agent: str = "Orchestrator",
        skill: str | None = None,
    ) -> None:
        resolved_skill = skill if skill is not None else self._current_skill
        self._post_ui(
            "update_active_task",
            {"title": title, "status": status, "agent": agent, "skill": resolved_skill},
        )

    def _request_approval(self, operation: str, details: Mapping[str, Any]) -> bool:
        """Use the TUI approval panel in interactive mode, otherwise use the console."""
        if not self.interactive:
            return _request_approval(operation, details)
        if self._app is None:
            logger.error("Approval requested before the TUI was attached; denying it.")
            return False
        return self._app.request_approval_from_thread(operation, details)

    def _on_issues_snapshot(
        self,
        issues: Sequence[GitHubIssue],
        in_progress_issue_numbers: frozenset[int],
    ) -> None:
        with self._state_lock:
            self._latest_open_issues = tuple(issues)
            self._latest_claimed_issue_numbers = in_progress_issue_numbers
        self._render_issues_snapshot()

    def _render_issues_snapshot(self) -> None:
        with self._state_lock:
            issues = self._latest_open_issues
            claimed = set(self._latest_claimed_issue_numbers)
            skipped = set(self._skipped_issue_numbers)
            outcomes = dict(self._issue_statuses)
        with self._queue_condition:
            queued = set(self._pending_issue_numbers)
        rows = []
        for issue in issues:
            if issue.number in outcomes:
                status = outcomes[issue.number]
            elif issue.number in skipped:
                status = "SKIPPED"
            elif issue.number in queued:
                status = "QUEUED"
            elif issue.number in claimed:
                status = "IN PROGRESS"
            else:
                status = "OPEN"
            rows.append({"number": issue.number, "title": issue.title, "status": status})
        self._post_ui("update_issues", rows)

    def _format_coder_result(self, issue_number: int, result: Any) -> list[str]:
        """Format observed result fields for the log panel without claiming unverified success."""
        lines = [
            f"Issue #{issue_number} result: {result.status.value}",
            f"Attempts: {result.attempts}",
        ]
        if result.plan:
            lines.append(f"Implementation plan:\n{result.plan}")
        if result.test_result is not None:
            verdict = "PASSED" if result.test_result.succeeded else "FAILED"
            lines.append(f"Tests: {verdict} (exit code {result.test_result.return_code})")
        if result.diff_path is not None:
            lines.append(f"Proposed diff: {result.diff_path}")
        if result.error:
            lines.append(f"Error: {result.error}")
        return lines

    def _report_coder_result(self, issue_number: int, result: Any) -> None:
        if not self.interactive:
            _print_coder_result(issue_number, result)
            return
        for line in self._format_coder_result(issue_number, result):
            self._log(line, source="Coder")

    def process_issue(self, issue: GitHubIssue) -> None:
        """Run the existing issue-to-tested-diff-to-approved-PR workflow."""
        if self.stop_event.is_set():
            self.polling.release_issue(issue.number)
            return
        with self._state_lock:
            already_attempted = issue.number in self._attempted_issues
            if not already_attempted:
                self._attempted_issues.add(issue.number)
        if already_attempted:
            self._log(f"Skipping issue #{issue.number}: already attempted this session.")
            self.polling.release_issue(issue.number)
            return
        if issue.number in self.recently_failed_issues:
            self._log(
                f"Skipping issue #{issue.number}: it FAILED in one of the last "
                f"{len(self.recent_session_summaries)} session(s) (episodic memory). "
                "Review it before retrying; delete .codepilot/memory/sessions.jsonl to reset.",
                source="Memory",
            )
            self.polling.release_issue(issue.number)
            return

        task_started = time.monotonic()
        files_modified: list[str] = []
        pull_request_url: str | None = None

        with self._task_lock:
            title = f"Issue #{issue.number}: {issue.title}"
            self._current_skill = _skill_name_for(issue)
            self._update_task(title, "TRIAGED", agent="Orchestrator")
            self._log(f"Starting coding workflow for issue #{issue.number}: {issue.title}")

            lifecycle_status = TaskStatus.TRIAGED

            def report_lifecycle(target: TaskStatus) -> None:
                nonlocal lifecycle_status

                if target == lifecycle_status:
                    return

                lifecycle_status = transition_task(lifecycle_status, target)

                agent = {
                    TaskStatus.EXPLORING: "Repo Explorer",
                    TaskStatus.IMPLEMENTING: "Coder",
                    TaskStatus.TESTING: "Test Agent",
                    TaskStatus.PR_OPENED: "PR Agent",
                    TaskStatus.DONE: "PR Agent",
                    TaskStatus.FAILED: "Runtime",
                }.get(target, "Runtime")

                self._update_task(title, target.value, agent=agent)

            try:
                verdict = self.guardrails.check_input(
                    f"{issue.title}\n\n{getattr(issue, 'body', '') or ''}"
                )
                if verdict.blocked:
                    with self._state_lock:
                        self._issue_statuses[issue.number] = "BLOCKED"
                    report_lifecycle(TaskStatus.FAILED)
                    self._log(
                        f"Issue #{issue.number} not sent to the LLM. {verdict.describe()}",
                        source="Guardrails",
                    )
                    logger.warning(
                        "Guardrails blocked issue #%s (rule=%s, engine=%s).",
                        issue.number,
                        verdict.rule,
                        verdict.engine,
                    )
                    return

                with tempfile.TemporaryDirectory(
                    prefix=f"codepilot-source-{issue.number}-"
                ) as temp_dir:
                    checkout = Path(temp_dir) / "repository"
                    lifecycle_status = transition_task(
                        lifecycle_status,
                        TaskStatus.EXPLORING,
                    )
                    self._update_task(title, "EXPLORING", agent="Repo Explorer")
                    self._log(f"Cloning repository for issue #{issue.number}.")
                    _clone_repository(self.settings, checkout)

                    result = self.orchestrator.run_coding_task(
                        issue,
                        repository_path=str(checkout),
                        lifecycle_callback=report_lifecycle,
                    )
                    files_modified = self._changed_files_for(result)

                    self._report_coder_result(issue.number, result)
                    if result.status != CoderStatus.SUCCEEDED:
                        with self._state_lock:
                            self._issue_statuses[issue.number] = "FAILED"
                        self._update_task(title, "FAILED", agent="Coder")
                        self._log(
                            f"Issue #{issue.number} was not published because coding/tests failed.",
                            source="Runtime",
                        )
                        return

                    self._update_task(title, "PR_OPENING", agent="PR Agent")

                    summary = (
                        f"Automated implementation for issue #{issue.number}.\n\n"
                        f"Implementation plan:\n{result.plan.strip()}"
                    )

                    # Build the structured commit message required by the assignment.
                    commit_summary = " ".join(issue.title.split()).rstrip(".")
                    commit_summary = commit_summary[:72].rstrip()

                    issue_body = " ".join(getattr(issue, "body", "").split())
                    why_summary = (
                        issue_body[:180].rstrip() if issue_body else "Address the reported issue."
                    )

                    commit_message = build_structured_commit_message(
                        issue_number=issue.number,
                        summary=commit_summary,
                        what_changed=commit_summary,
                        why=why_summary,
                    )

                    pull_request = self.orchestrator.publish_coder_result(
                        issue,
                        result,
                        commit_message=commit_message,
                        summary=summary,
                        reviewer=getattr(issue, "reporter", None),
                        base_branch=self.settings.github_default_branch,
                    )

                pull_request_url = getattr(pull_request, "pull_url", None)
                with self._state_lock:
                    self._issue_statuses[issue.number] = "PR OPENED"

                report_lifecycle(TaskStatus.PR_OPENED)
                report_lifecycle(TaskStatus.DONE)

                self._log(
                    f"Pull request created: {pull_request.pull_url}",
                    source="PR Agent",
                )
                for warning in pull_request.metadata_warnings:
                    message = (
                        f"Pull request #{pull_request.pull_number} exists, "
                        f"but metadata setup was incomplete: {warning}"
                    )
                    self._log(message, source="PR Agent")
                    logger.warning(message)
                logger.info("Created PR #%s for issue #%s.", pull_request.pull_number, issue.number)
            except KeyboardInterrupt:
                with self._state_lock:
                    self._issue_statuses[issue.number] = "INTERRUPTED"
                self._update_task(title, "FAILED", agent="Runtime")
                logger.warning("Interrupted while processing issue #%s.", issue.number)
                raise
            except MergeConflictError as exc:
                with self._state_lock:
                    self._issue_statuses[issue.number] = "FAILED"
                self._update_task(title, "FAILED", agent="PR Agent")
                self._log(
                    f"MERGE CONFLICT for issue #{issue.number}: {exc} "
                    "Human action needed: rebase or re-run after reviewing the base branch.",
                    source="PR Agent",
                )
                logger.warning("Merge conflict for issue #%s: %s", issue.number, exc)
            except HumanApprovalRejectedError as exc:
                with self._state_lock:
                    self._issue_statuses[issue.number] = "REJECTED"
                self._update_task(title, "REJECTED", agent="PR Agent")
                self._log(
                    f"Publication rejected for issue #{issue.number}: {exc}",
                    source="Human Approval",
                )
                logger.info("Publication rejected for issue #%s.", issue.number)
            except Exception as exc:
                with self._state_lock:
                    self._issue_statuses[issue.number] = "FAILED"
                self._update_task(title, "FAILED", agent="Runtime")
                self._log(f"Coding workflow failed for issue #{issue.number}: {exc}")
                logger.exception("Coding workflow failed for issue #%s", issue.number)
            finally:
                self.polling.release_issue(issue.number)
                self._record_session_task(
                    issue,
                    started=task_started,
                    files_modified=files_modified,
                    pull_request_url=pull_request_url,
                )

    def _changed_files_for(self, result: Any) -> list[str]:
        try:
            return list(self.orchestrator.changed_source_files(result))
        except Exception:
            logger.debug("Could not determine changed files for session memory.", exc_info=True)
            return []

    def _record_session_task(
        self,
        issue: Any,
        *,
        started: float,
        files_modified: Sequence[str],
        pull_request_url: str | None = None,
        source: str = "github_issue",
        outcome: str | None = None,
    ) -> None:
        """Record a finished task in the session log and the episodic store."""
        with self._state_lock:
            status = outcome or self._issue_statuses.get(issue.number)
        if not status:
            return
        outcome_value = status.replace(" ", "_")
        try:
            self.session_tracker.record(
                issue_number=issue.number,
                title=getattr(issue, "title", ""),
                task_type=_category_name_for(issue),
                outcome=outcome_value,
                files_modified=files_modified,
                duration_seconds=time.monotonic() - started,
                source=source,
                pull_request_url=pull_request_url,
            )
            record_episode = getattr(self.orchestrator, "record_episode", None)
            if callable(record_episode) and isinstance(issue, GitHubIssue):
                record_episode(
                    issue,
                    succeeded=outcome_value in {"PR_OPENED", "COMPLETED"},
                    summary=(
                        f"{source} outcome {outcome_value}; files: "
                        f"{', '.join(files_modified) or 'none'}"
                    ),
                )
        except Exception:
            logger.warning("Could not record task #%s in session memory.", issue.number)

    def _announce_recent_sessions(self) -> None:
        count = len(self.recent_session_summaries)
        if not count:
            self._log("Episodic memory: no previous session summaries.", source="Memory")
            return
        failed = sorted(self.recently_failed_issues)
        self._log(
            f"Episodic memory: loaded {count} previous session summary(ies); "
            f"recently failed issues that will be skipped: {failed or 'none'}.",
            source="Memory",
        )

    def run_headless(self) -> None:
        """Run the original synchronous issue-processing loop without a TUI."""
        print(
            f"CodePilot coding loop started for "
            f"{self.settings.github_repository_owner}/"
            f"{self.settings.github_repository_name}. Press Ctrl+C to stop."
        )
        self._announce_recent_sessions()
        self.polling.run_forever(self.process_issue, self.stop_event)

    def start(self) -> None:
        """Start polling and queued issue processing in background threads for the TUI."""
        if not self.interactive:
            raise RuntimeError("The runtime must be created with interactive=True for TUI startup.")
        if self._started:
            return
        self._started = True
        self._announce_recent_sessions()
        self._processor_thread = Thread(
            target=self._consume_pending_issues,
            name="codepilot-issue-processor",
            daemon=True,
        )
        self._polling_thread = Thread(
            target=self.polling.run_forever,
            args=(self._enqueue_issue, self.stop_event),
            name="codepilot-github-polling",
            daemon=True,
        )
        self._processor_thread.start()
        self._polling_thread.start()
        self._log("Runtime started; GitHub issue polling is active.")

    def stop(self) -> None:
        """Request a prompt shutdown of polling and queued work."""
        self.stop_event.set()
        try:
            self.session_tracker.write_summary()
        except Exception:
            logger.warning("Could not write the episodic session summary.", exc_info=True)
        with self._queue_condition:
            self._queue_condition.notify_all()

    def _enqueue_issue(self, issue: GitHubIssue) -> None:
        should_release_claim = False
        with self._queue_condition:
            with self._state_lock:
                already_seen = (
                    issue.number in self._attempted_issues
                    or issue.number in self._skipped_issue_numbers
                )
            already_queued = issue.number in self._pending_issue_numbers
            if self.stop_event.is_set() or already_seen:
                should_release_claim = True
            elif not already_queued:
                self._pending_issues.append(issue)
                self._pending_issue_numbers.add(issue.number)
                self._queue_condition.notify()
        # Releasing a claim notifies the TUI. Do so outside the queue lock because
        # Textual's thread bridge waits for the event-loop callback to complete.
        if should_release_claim:
            self.polling.release_issue(issue.number)
            return
        if already_queued:
            return
        self._render_issues_snapshot()
        self._log(f"Queued issue #{issue.number}: {issue.title}", source="Orchestrator")

    def _consume_pending_issues(self) -> None:
        while True:
            pending_to_release: list[GitHubIssue] | None = None
            with self._queue_condition:
                while not self._pending_issues and not self.stop_event.is_set():
                    self._queue_condition.wait(timeout=0.5)
                if self.stop_event.is_set():
                    pending_to_release = list(self._pending_issues)
                    self._pending_issues.clear()
                    self._pending_issue_numbers.clear()
                else:
                    issue = self._pending_issues.popleft()
                    self._pending_issue_numbers.discard(issue.number)
            if pending_to_release is not None:
                for pending_issue in pending_to_release:
                    self.polling.release_issue(pending_issue.number)
                self._render_issues_snapshot()
                return
            self._render_issues_snapshot()
            self.process_issue(issue)

    def skip_issue(self) -> None:
        """Skip the oldest queued issue; an already-running task is not force-cancelled."""
        if not self.interactive:
            raise RuntimeError("Issue skipping is available only in interactive mode.")
        with self._queue_condition:
            issue = self._pending_issues.popleft() if self._pending_issues else None
            if issue is not None:
                self._pending_issue_numbers.discard(issue.number)
        if issue is None:
            self._log("No queued issue is available to skip; the active task is not cancelled.")
            return
        with self._state_lock:
            self._skipped_issue_numbers.add(issue.number)
            self._attempted_issues.add(issue.number)
            self._issue_statuses[issue.number] = "SKIPPED"
        self.polling.release_issue(issue.number)
        self._render_issues_snapshot()
        self._log(f"Skipped issue #{issue.number} for this session.", source="TUI")

    async def skip_issue_async(self) -> None:
        """Run the thread-based queue operation without blocking the TUI event loop."""
        await asyncio.to_thread(self.skip_issue)

    def _allocate_direct_task_number(self) -> int:
        with self._state_lock:
            while (self.workspace_root / f"issue-{self._direct_task_number}").exists():
                self._direct_task_number += 1
            number = self._direct_task_number
            self._direct_task_number += 1
            return number

    def process_direct_task(self, instruction: str) -> str:
        """Code and test a user-submitted task without inventing or closing a GitHub issue."""
        task = instruction.strip()
        if not task:
            raise ValueError("A direct coding task must not be empty.")

        task_started = time.monotonic()
        verdict = self.guardrails.check_input(task)
        if verdict.blocked:
            self._update_task(f"Direct task: {task}", "FAILED", agent="Guardrails")
            self._log(f"Direct task not sent to the LLM. {verdict.describe()}", source="Guardrails")
            self._record_session_task(
                SimpleNamespace(number=0, title=task),
                started=task_started,
                files_modified=(),
                source="direct_task",
                outcome="BLOCKED",
            )
            raise GuardrailBlockedError(verdict.describe())

        with self._task_lock:
            internal_number = self._allocate_direct_task_number()
            issue = GitHubIssue(
                number=internal_number,
                title=task[:200],
                body=task,
            )
            title = f"Direct task: {task}"
            self._current_skill = _skill_name_for(issue)
            self._update_task(title, "EXPLORING", agent="Repo Explorer")
            self._log("Starting a direct task against the configured repository.", source="TUI")
            try:
                with tempfile.TemporaryDirectory(prefix="codepilot-direct-source-") as temp_dir:
                    checkout = Path(temp_dir) / "repository"
                    _clone_repository(self.settings, checkout)
                    self._update_task(title, "IMPLEMENTING", agent="Coder")
                    result = self.orchestrator.run_coding_task(
                        issue,
                        repository_path=str(checkout),
                    )
                    files_modified = self._changed_files_for(result)
            except Exception as exc:
                self._update_task(title, "FAILED", agent="Runtime")
                self._log(f"Direct task failed: {exc}", source="TUI")
                self._record_session_task(
                    issue,
                    started=task_started,
                    files_modified=(),
                    source="direct_task",
                    outcome="FAILED",
                )
                raise

            self._report_coder_result(internal_number, result)
            succeeded = result.status == CoderStatus.SUCCEEDED
            self._record_session_task(
                issue,
                started=task_started,
                files_modified=files_modified,
                source="direct_task",
                outcome="COMPLETED" if succeeded else "FAILED",
            )
            if not succeeded:
                self._update_task(title, "FAILED", agent="Coder")
                raise RuntimeError(
                    f"Direct task did not complete successfully: "
                    f"{result.error or result.status.value}. "
                    f"Proposed diff: {result.diff_path or 'not available'}"
                )

            self._update_task(title, "COMPLETED", agent="Coder")
            diff_path = str(result.diff_path) if result.diff_path is not None else "not reported"
            self._log(
                "Direct task passed its configured test workflow. No PR was opened because "
                "the task is not linked to a GitHub issue. Review the proposed diff at: "
                f"{diff_path}",
                source="TUI",
            )
            return (
                "Direct task completed and its configured test workflow passed.\n"
                f"Proposed diff: {diff_path}\n"
                "No pull request was opened for this direct task."
            )

    async def submit_direct_task(self, instruction: str) -> str:
        """Execute a direct task off the Textual event-loop thread."""
        return await asyncio.to_thread(self.process_direct_task, instruction)

    async def submit_guardrail_check(self, command: str) -> str:
        """Check a command in a worker thread without executing it."""
        return await asyncio.to_thread(
            _check_guardrail_command,
            command,
            approval_callback=self._request_approval,
            guardrails=self.guardrails,
        )


def _resolve_settings(settings: Settings | None) -> Settings:
    if settings is None:
        load_dotenv(Path.cwd() / ".env")
        resolved_settings = Settings.from_env()
    else:
        resolved_settings = settings
    resolved_settings.validate_github_configuration()
    resolved_settings.validate_llm_configuration()
    return resolved_settings


def run(*, settings: Settings | None = None, stop_event: Event | None = None) -> int:
    """Run the polling and coding workflow in console mode (no Textual interface)."""
    resolved_settings = _resolve_settings(settings)
    stop = stop_event or Event()
    runtime = CodePilotRuntime(resolved_settings, stop_event=stop)

    def request_stop(_signum: int, _frame: Any) -> None:
        runtime.stop()
        raise KeyboardInterrupt

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = None
    if hasattr(signal, "SIGTERM"):
        previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        runtime.run_headless()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
    return 0


def run_tui(*, settings: Settings | None = None) -> int:
    """Launch CodePilot with the Textual interface connected to the runtime."""
    resolved_settings = _resolve_settings(settings)
    # Keep Textual imports lazy so console-mode use and runner unit tests do not
    # require the UI module to be imported before it is needed.
    from codepilot.tui.app import CodePilotApp

    runtime = CodePilotRuntime(resolved_settings, interactive=True)
    app = CodePilotApp(
        task_handler=runtime.submit_direct_task,
        guardrail_handler=runtime.submit_guardrail_check,
        skip_handler=runtime.skip_issue_async,
        startup_handler=runtime.start,
        shutdown_handler=runtime.stop,
    )
    runtime.attach_app(app)

    # Textual owns the terminal while the TUI is active. Temporarily replace
    # root console handlers with a bridge that forwards application log records
    # into Agent Logs instead of printing over the dashboard.
    root_logger = logging.getLogger()
    previous_handlers = root_logger.handlers[:]
    previous_level = root_logger.level
    tui_handler = _TUILogForwardingHandler(runtime)
    for handler in previous_handlers:
        root_logger.removeHandler(handler)
    root_logger.addHandler(tui_handler)
    root_logger.setLevel(logging.INFO)
    try:
        app.run()
    finally:
        runtime.stop()
        root_logger.removeHandler(tui_handler)
        for handler in previous_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(previous_level)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; the TUI is default and ``--headless`` preserves console mode."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="CodePilot multi-agent coding platform")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run the console polling loop instead of the Textual interface",
    )
    arguments = parser.parse_args(argv)
    try:
        return run() if arguments.headless else run_tui()
    except ConfigurationError as exc:
        logger.error("Invalid configuration: %s", exc)
        return 2
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
