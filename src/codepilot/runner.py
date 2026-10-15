"""Command-line entry point for CodePilot's GitHub issue coding loop."""
from __future__ import annotations

import base64
import logging
import re
from copy import copy
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from threading import Event
from typing import Any
from collections.abc import Mapping

from codepilot.agents.coder_models import CoderStatus
from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.config.bootstrap import create_orchestrator
from codepilot.config.settings import ConfigurationError, Settings
from codepilot.github.client import GitHubClient
from codepilot.github.polling import GitHubPollingService
from codepilot.github.pr_workflow import PullRequestWorkflow
from dotenv import load_dotenv

logger = logging.getLogger("codepilot.runner")


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
            setattr(cloned, "name", name)
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
    command = ["git", "clone", "--depth", "1", "--branch", settings.github_default_branch, url, str(destination)]
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


def run(*, settings: Settings | None = None, stop_event: Event | None = None) -> int:
    """Poll GitHub issues, implement each selected issue, test, and request PR approval."""
    if settings is None:
        load_dotenv(Path.cwd() / ".env")
        resolved_settings = Settings.from_env()
    else:
        resolved_settings = settings
    resolved_settings.validate_github_configuration()
    resolved_settings.validate_llm_configuration()

    client = GitHubClient(resolved_settings)
    pr_workflow = PullRequestWorkflow(
        github_client=client,
        default_branch=resolved_settings.github_default_branch,
        approval_callback=_request_approval,
    )
    # Keep working copies outside the CodePilot checkout to avoid recursively
    # copying .codepilot workspaces into themselves.
    workspace_root = Path.cwd().parent / f".{Path.cwd().name}-codepilot-workspaces"
    orchestrator: CodePilotOrchestrator = create_orchestrator(
        tools=[],
        workspace_root=workspace_root,
        pr_workflow=pr_workflow,
    )
    polling = GitHubPollingService(client, resolved_settings)
    stop = stop_event or Event()
    attempted_issues: set[int] = set()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()
        raise KeyboardInterrupt

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = None
    if hasattr(signal, "SIGTERM"):
        previous_sigterm = signal.signal(signal.SIGTERM, request_stop)

    def process_issue(issue: Any) -> None:
        if issue.number in attempted_issues:
            logger.info("Skipping issue #%s: already attempted this session.", issue.number)
            polling.release_issue(issue.number)
            return
        attempted_issues.add(issue.number)
        logger.info("Starting coding workflow for issue #%s: %s", issue.number, issue.title)
        try:
            with tempfile.TemporaryDirectory(prefix=f"codepilot-source-{issue.number}-") as temp_dir:
                checkout = Path(temp_dir) / "repository"
                _clone_repository(resolved_settings, checkout)
                result = orchestrator.run_coding_task(issue, repository_path=str(checkout))

            _print_coder_result(issue.number, result)
            if result.status != CoderStatus.SUCCEEDED:
                logger.warning("Issue #%s was not published because coding/tests failed.", issue.number)
                return

            summary = (
                f"Automated implementation for issue #{issue.number}.\n\n"
                f"Implementation plan:\n{result.plan.strip()}"
            )
            pull_request = orchestrator.publish_coder_result(
                issue,
                result,
                commit_message=f"fix: address issue #{issue.number} - {issue.title}"[:200],
                summary=summary,
                reviewer=getattr(issue, "reporter", None),
                base_branch=resolved_settings.github_default_branch,
            )
            print(f"\nPull request created: {pull_request.pull_url}")
            logger.info("Created PR #%s for issue #%s.", pull_request.pull_number, issue.number)
        except KeyboardInterrupt:
            logger.warning("Interrupted while processing issue #%s.", issue.number)
            raise
        except Exception:
            logger.exception("Coding workflow failed for issue #%s.", issue.number)
        finally:
            polling.release_issue(issue.number)

    try:
        print(
            f"CodePilot coding loop started for "
            f"{resolved_settings.github_repository_owner}/"
            f"{resolved_settings.github_repository_name}. Press Ctrl+C to stop."
        )
        polling.run_forever(process_issue, stop)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
    return 0


def main() -> int:
    """CLI entry point with a concise configuration error message."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return run()
    except ConfigurationError as exc:
        logger.error("Invalid configuration: %s", exc)
        return 2
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
