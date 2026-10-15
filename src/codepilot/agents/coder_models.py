"""Structured models for the Coder Agent workflow."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from codepilot.agents.test_models import TestResult
from codepilot.sandbox.execution_models import CommandExecutionResult
from codepilot.sandbox.workspace import WorkspaceInfo


class CoderStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CoderResult:
    status: CoderStatus
    issue_number: int
    workspace: WorkspaceInfo
    plan: str = ""
    attempts: int = 0
    test_result: CommandExecutionResult | None = None
    diff_path: Path | None = None
    error: str | None = None
    test_assessment: TestResult | None = None

    def __post_init__(self) -> None:
        """Validate success and failure result consistency."""
        if self.issue_number <= 0:
            raise ValueError("issue_number must be positive.")
        if not 0 <= self.attempts <= 3:
            raise ValueError("attempts must be between 0 and 3.")
        if self.status == CoderStatus.SUCCEEDED:
            if self.test_result is None or not self.test_result.succeeded:
                raise ValueError("A successful task must have a successful test result.")
            if self.diff_path is None:
                raise ValueError("A successful task must include a proposed diff path.")
            if self.error is not None:
                raise ValueError("A successful task cannot contain an error.")
            if (
                self.test_assessment is not None
                and self.test_assessment.command_result != self.test_result
            ):
                raise ValueError("Test assessment and command result must match.")
        if self.status == CoderStatus.FAILED and not self.error:
            raise ValueError("A failed task must include an error.")
