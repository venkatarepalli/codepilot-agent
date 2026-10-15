"""Structured result models for the Test Agent."""

from dataclasses import dataclass
from enum import StrEnum

from codepilot.sandbox.execution_models import (
    CommandExecutionResult,
    CommandStatus,
)


class TestStatus(StrEnum):
    """Possible outcomes of a Test Agent execution."""

    __test__ = False

    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    REQUIRES_APPROVAL = "requires_approval"
    TIMED_OUT = "timed_out"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TestResult:
    """Immutable, structured result returned by the Test Agent."""

    __test__ = False

    issue_number: int
    attempt: int
    status: TestStatus
    command_result: CommandExecutionResult | None
    summary: str = ""
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate the result and keep its status consistent."""
        if self.issue_number < 1:
            raise ValueError("issue number must be positive.")

        if not 1 <= self.attempt <= 3:
            raise ValueError("attempt must be between 1 and 3.")

        if not isinstance(self.status, TestStatus):
            raise ValueError("Status must be a valid TestStatus.")

        if self.status == TestStatus.ERROR:
            if not self.error or not self.error.strip():
                raise ValueError("Error details are required for an error result.")

            if self.command_result is not None:
                raise ValueError("An error result must not contain a command result.")

            return

        if self.error is not None:
            raise ValueError("Error details are only allowed for an error result.")

        if self.command_result is None:
            raise ValueError("A command result is required for this test status.")

        expected_command_status = {
            TestStatus.PASSED: CommandStatus.SUCCEEDED,
            TestStatus.FAILED: CommandStatus.FAILED,
            TestStatus.BLOCKED: CommandStatus.BLOCKED,
            TestStatus.REQUIRES_APPROVAL: CommandStatus.REQUIRES_APPROVAL,
            TestStatus.TIMED_OUT: CommandStatus.TIMED_OUT,
        }[self.status]

        if self.command_result.status != expected_command_status:
            raise ValueError(
                f"Test status '{self.status.value}' must match command "
                f"status '{expected_command_status.value}'."
            )
