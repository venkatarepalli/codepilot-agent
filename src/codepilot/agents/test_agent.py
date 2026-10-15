"""Test Agent for executing and reporting test outcomes."""

from pathlib import Path

from codepilot.agents.test_models import TestResult, TestStatus
from codepilot.sandbox.coder_workspace import CoderWorkspace
from codepilot.sandbox.execution_models import CommandExecutionResult, CommandStatus
from codepilot.sandbox.workspace import WorkspaceInfo


class TestAgent:
    """Execute a test command once and report a structured outcome.

    Retry decisions belong to the Coder Agent.
    """

    __test__ = False

    def __init__(self, workspace: CoderWorkspace) -> None:
        """Initialize the Test Agent with the workspace executor."""
        self.workspace = workspace

    def run(
        self,
        *,
        issue_number: int,
        attempt: int,
        command: tuple[str, ...] | None = None,
        workspace_info: WorkspaceInfo | None = None,
        test_command: tuple[str, ...] | None = None,
        workspace_path: Path | None = None,
    ) -> TestResult:
        """Execute tests and return a structured result.

        ``workspace_info`` and ``command`` are the preferred arguments.
        ``workspace_path`` and ``test_command`` remain supported for callers
        using the earlier Test Agent interface.
        """
        if issue_number < 1:
            raise ValueError("issue number must be positive.")
        if not 1 <= attempt <= 3:
            raise ValueError("attempt must be between 1 and 3.")

        if workspace_info is not None and workspace_path is not None:
            raise ValueError("Supply workspace_info or workspace_path, not both.")
        if workspace_info is None and workspace_path is None:
            raise ValueError("workspace_info or workspace_path is required.")

        if command is not None and test_command is not None:
            raise ValueError("Supply command or test_command, not both.")
        resolved_command = command if command is not None else test_command
        if not resolved_command or any(
            not isinstance(arg, str) or not arg.strip() for arg in resolved_command
        ):
            raise ValueError("command must contain non-empty string arguments.")

        target_workspace = workspace_info if workspace_info is not None else workspace_path
        try:
            command_result = self.workspace.execute(target_workspace, resolved_command)
        except Exception as exc:
            return TestResult(
                issue_number=issue_number,
                attempt=attempt,
                status=TestStatus.ERROR,
                command_result=None,
                summary="Test execution raised an exception.",
                error=f"{type(exc).__name__}: {exc}",
            )

        status_mapping = {
            CommandStatus.SUCCEEDED: TestStatus.PASSED,
            CommandStatus.FAILED: TestStatus.FAILED,
            CommandStatus.BLOCKED: TestStatus.BLOCKED,
            CommandStatus.REQUIRES_APPROVAL: TestStatus.REQUIRES_APPROVAL,
            CommandStatus.TIMED_OUT: TestStatus.TIMED_OUT,
        }
        status = status_mapping.get(command_result.status)
        if status is None:
            return TestResult(
                issue_number=issue_number,
                attempt=attempt,
                status=TestStatus.ERROR,
                command_result=None,
                summary="Test execution returned an unrecognized status.",
                error=f"Unrecognized command status: {command_result.status!r}",
            )

        return TestResult(
            issue_number=issue_number,
            attempt=attempt,
            status=status,
            command_result=command_result,
            summary=self._build_summary(command_result),
        )

    @staticmethod
    def _build_summary(result: CommandExecutionResult) -> str:
        """Create a concise summary of the command outcome."""
        if result.status == CommandStatus.SUCCEEDED:
            return result.stdout.strip() or "Test command completed successfully."
        if result.status == CommandStatus.FAILED:
            return result.stderr.strip() or result.stdout.strip() or "Test command failed."
        if result.status == CommandStatus.BLOCKED:
            return f"Test command was blocked: {result.reason}"
        if result.status == CommandStatus.REQUIRES_APPROVAL:
            return f"Test command requires human approval: {result.reason}"
        if result.status == CommandStatus.TIMED_OUT:
            return f"Test command timed out: {result.reason}"
        return "Test command returned an unrecognized status."
