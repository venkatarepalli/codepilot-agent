"""Coder Agent: plan, implement, test, retry, and prepare a proposed diff."""

import difflib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codepilot.agents.coder_models import CoderResult, CoderStatus
from codepilot.agents.test_agent import TestAgent
from codepilot.agents.test_models import TestResult, TestStatus
from codepilot.github.models import GitHubIssue
from codepilot.graph.lifecycle import TaskStatus
from codepilot.sandbox.coder_workspace import CoderWorkspace
from codepilot.sandbox.execution_models import CommandExecutionResult
from codepilot.sandbox.workspace import WorkspaceInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImplementationFeedback:
    """Feedback about an implementation attempt that is not a test result."""

    summary: str

    def __str__(self) -> str:
        """Expose the feedback text directly to the model prompt."""
        return self.summary


Planner = Callable[[GitHubIssue, WorkspaceInfo], str]
ApprovalCallback = Callable[[str, Mapping[str, Any]], bool]


Implementer = Callable[
    [
        GitHubIssue,
        WorkspaceInfo,
        str,
        int,
        TestResult | ImplementationFeedback | None,
    ],
    None,
]


class CoderAgent:
    """Coordinate implementation attempts inside an issue workspace."""

    MAX_ATTEMPTS = 3
    # HITL gate (Component 6): retrying after this many FAILED test runs
    # requires explicit human approval.
    RETRY_APPROVAL_AFTER_FAILED_TEST_RUNS = 2
    ARTIFACT_DIRECTORY = "working"
    PLAN_FILENAME = "implementation_plan.md"
    DIFF_FILENAME = "proposed_diff.txt"

    def __init__(
        self,
        workspace: CoderWorkspace,
        *,
        planner: Planner,
        implementer: Implementer,
        test_command: Sequence[str],
        test_agent: TestAgent | None = None,
        approval_callback: ApprovalCallback | None = None,
        lifecycle_callback: Callable[[TaskStatus], None] | None = None,
    ) -> None:
        """Initialize the Coder Agent with explicit or default Test Agent."""
        if not test_command:
            raise ValueError("test_command cannot be empty.")
        if not all(isinstance(argument, str) and argument.strip() for argument in test_command):
            raise ValueError("Every test command argument must be a non-empty string.")

        self.workspace = workspace
        self.planner = planner
        self.implementer = implementer
        self.lifecycle_callback = lifecycle_callback
        self.test_command = tuple(test_command)
        self.test_agent = test_agent if test_agent is not None else TestAgent(workspace)
        self.approval_callback = approval_callback

    def _report_lifecycle(self, status: TaskStatus) -> None:
        """Report a lifecycle event when a callback is configured."""
        if self.lifecycle_callback is not None:
            self.lifecycle_callback(status)

    def run(
        self,
        issue: GitHubIssue,
        repository_path: str | Path,
    ) -> CoderResult:
        """Execute the coding workflow for a GitHub issue."""
        task_workspace = self.workspace.create(repository_path, issue.number)
        plan = ""

        try:
            plan = self.planner(issue, task_workspace)
            if not isinstance(plan, str) or not plan.strip():
                raise ValueError("The planner must return a non-empty implementation plan.")
            self.workspace.write_file(
                task_workspace,
                f"{self.ARTIFACT_DIRECTORY}/{self.PLAN_FILENAME}",
                plan.rstrip() + "\n",
            )
        except Exception as exc:
            return self._failure(
                issue,
                task_workspace,
                error=f"Planning failed: {type(exc).__name__}: {exc}",
                plan=plan if isinstance(plan, str) else "",
            )

        previous_test_assessment: TestResult | None = None
        implementation_feedback: ImplementationFeedback | None = None
        failed_test_runs = 0

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                self._report_lifecycle(TaskStatus.IMPLEMENTING)
                self.implementer(
                    issue,
                    task_workspace,
                    plan,
                    attempt,
                    (
                        implementation_feedback
                        if implementation_feedback is not None
                        else previous_test_assessment
                    ),
                )
                # A normal return means this implementation call accepted
                # its input; do not carry stale empty-response feedback onward.
                implementation_feedback = None
            except ValueError as exc:
                if (
                    "DeepAgents returned an empty change set" in str(exc)
                    and attempt < self.MAX_ATTEMPTS
                ):
                    implementation_feedback = ImplementationFeedback(
                        summary=(
                            "The previous implementation attempt returned an "
                            "empty change set. No file changes were proposed. "
                            "Return a non-empty JSON change set using the "
                            "required schema. Include the complete contents "
                            "of each changed file. Do not claim success without "
                            "proposing actual file changes."
                        )
                    )
                    logger.info(
                        "Attempt %s produced no file changes; retrying implementation "
                        "with explicit feedback.",
                        attempt,
                    )
                    continue

                return self._failure(
                    issue,
                    task_workspace,
                    error=(
                        f"Implementation failed on attempt {attempt}: {type(exc).__name__}: {exc}"
                    ),
                    plan=plan,
                    attempts=attempt,
                    test_assessment=previous_test_assessment,
                )
            except Exception as exc:
                return self._failure(
                    issue,
                    task_workspace,
                    error=(
                        f"Implementation failed on attempt {attempt}: {type(exc).__name__}: {exc}"
                    ),
                    plan=plan,
                    attempts=attempt,
                    test_assessment=previous_test_assessment,
                )

            try:
                self._report_lifecycle(TaskStatus.TESTING)
                assessment = self.test_agent.run(
                    issue_number=issue.number,
                    attempt=attempt,
                    workspace_info=task_workspace,
                    command=self.test_command,
                )
            except Exception as exc:
                return self._failure(
                    issue,
                    task_workspace,
                    error=(f"Test Agent failed on attempt {attempt}: {type(exc).__name__}: {exc}"),
                    plan=plan,
                    attempts=attempt,
                    test_assessment=previous_test_assessment,
                )

            previous_test_assessment = assessment
            command_result = assessment.command_result

            if assessment.status == TestStatus.PASSED:
                try:
                    if command_result is None:
                        raise ValueError("Passed test assessment has no command result.")
                    diff_path = self._write_proposed_diff(task_workspace)
                except Exception as exc:
                    return self._failure(
                        issue,
                        task_workspace,
                        error=(f"Proposed diff generation failed: {type(exc).__name__}: {exc}"),
                        plan=plan,
                        attempts=attempt,
                        test_result=command_result,
                        test_assessment=assessment,
                    )

                return CoderResult(
                    status=CoderStatus.SUCCEEDED,
                    issue_number=issue.number,
                    workspace=task_workspace,
                    plan=plan,
                    attempts=attempt,
                    test_result=command_result,
                    test_assessment=assessment,
                    diff_path=diff_path,
                )

            if assessment.status == TestStatus.FAILED:
                failed_test_runs += 1

            # Count real failed test runs, not loop iterations: an empty
            # change-set attempt never ran tests and must not trip the gate.
            if (
                assessment.status == TestStatus.FAILED
                and failed_test_runs == self.RETRY_APPROVAL_AFTER_FAILED_TEST_RUNS
                and attempt < self.MAX_ATTEMPTS
            ):
                logger.info(
                    "Issue #%s: %s failed test runs; requesting human approval to retry.",
                    issue.number,
                    failed_test_runs,
                )
                approval_details = {
                    "issue_number": issue.number,
                    "failed_test_runs": failed_test_runs,
                    "next_attempt": attempt + 1,
                    "test_summary": assessment.summary or assessment.error or "Tests failed.",
                    "workspace": str(task_workspace.path),
                }
                approved = False
                if self.approval_callback is not None:
                    try:
                        approved = (
                            self.approval_callback(
                                "retry_after_two_failed_test_runs", approval_details
                            )
                            is True
                        )
                    except Exception as exc:
                        return self._failure(
                            issue,
                            task_workspace,
                            error=(
                                "Human approval callback failed; retry was stopped: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                            plan=plan,
                            attempts=attempt,
                            test_result=command_result,
                            test_assessment=assessment,
                        )
                if not approved:
                    return self._failure(
                        issue,
                        task_workspace,
                        error=(
                            "Human approval was not granted for retrying after two "
                            "failed test runs. The workspace has been retained for inspection."
                        ),
                        plan=plan,
                        attempts=attempt,
                        test_result=command_result,
                        test_assessment=assessment,
                    )

            if assessment.status != TestStatus.FAILED:
                details = assessment.error or assessment.summary or assessment.status.value

                if assessment.status == TestStatus.ERROR:
                    error_message = f"Test execution failed: {details}"
                else:
                    error_message = (
                        f"Test workflow stopped with status '{assessment.status.value}': {details}"
                    )

                return self._failure(
                    issue,
                    task_workspace,
                    error=error_message,
                    plan=plan,
                    attempts=attempt,
                    test_result=command_result,
                    test_assessment=assessment,
                )

        last_command_result = (
            previous_test_assessment.command_result
            if previous_test_assessment is not None
            else None
        )
        return self._failure(
            issue,
            task_workspace,
            error=(
                f"Tests failed after {self.MAX_ATTEMPTS} implementation "
                "attempts. The workspace has been retained for inspection."
            ),
            plan=plan,
            attempts=self.MAX_ATTEMPTS,
            test_result=last_command_result,
            test_assessment=previous_test_assessment,
        )

    def _write_proposed_diff(self, workspace: WorkspaceInfo) -> Path:
        """Generate and save a unified diff for source-file changes."""
        changes = self.workspace.get_changes(workspace)
        sections: list[str] = []

        for category in ("modified", "added", "deleted"):
            for relative_path in changes[category]:
                if self._is_working_artifact(relative_path):
                    continue

                original_path = workspace.source_repository / relative_path
                workspace_path = workspace.path / relative_path
                original_lines = self._read_text_lines(original_path)
                updated_lines = self._read_text_lines(workspace_path)

                if original_lines is None or updated_lines is None:
                    raise ValueError(
                        "Cannot generate a text diff for "
                        f"binary or unreadable file: {relative_path}"
                    )

                sections.extend(
                    difflib.unified_diff(
                        original_lines,
                        updated_lines,
                        fromfile=f"a/{relative_path}",
                        tofile=f"b/{relative_path}",
                    )
                )

        if not sections:
            diff_content = (
                "No source-file changes were detected.\n"
                "Review the implementation plan and test results.\n"
            )
        else:
            diff_content = "".join(sections)

        relative_diff_path = f"{self.ARTIFACT_DIRECTORY}/{self.DIFF_FILENAME}"
        self.workspace.write_file(workspace, relative_diff_path, diff_content)
        return self.workspace.resolve_path(workspace, relative_diff_path)

    @staticmethod
    def _is_working_artifact(relative_path: str) -> bool:
        """Exclude CodePilot's own task artifacts from the source diff."""
        return relative_path == "working" or relative_path.startswith("working/")

    @staticmethod
    def _read_text_lines(path: Path) -> list[str] | None:
        """Read text lines, returning an empty list for a missing file."""
        if not path.exists():
            return []
        try:
            return path.read_text(encoding="utf-8").splitlines(keepends=True)
        except (UnicodeDecodeError, OSError):
            return None

    @staticmethod
    def _failure(
        issue: GitHubIssue,
        workspace: WorkspaceInfo,
        *,
        error: str,
        plan: str = "",
        attempts: int = 0,
        test_result: CommandExecutionResult | None = None,
        test_assessment: TestResult | None = None,
    ) -> CoderResult:
        """Build a consistent failure result."""
        return CoderResult(
            status=CoderStatus.FAILED,
            issue_number=issue.number,
            workspace=workspace,
            plan=plan,
            attempts=attempts,
            test_result=test_result,
            test_assessment=test_assessment,
            error=error,
        )
