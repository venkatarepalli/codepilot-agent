"""Structured models for task execution results."""

from dataclasses import dataclass
from enum import StrEnum

from codepilot.graph.lifecycle import TaskStatus


class ExecutionStatus(StrEnum):
    """Outcome of an individual execution step or complete task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result of executing a single task step."""

    name: str
    status: ExecutionStatus
    output: str = ""
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate the consistency of the step result."""
        if not self.name.strip():
            raise ValueError("Step name must not be empty.")

        if self.status == ExecutionStatus.FAILED and not self.error:
            raise ValueError("A failed step must include an error.")

        if self.status != ExecutionStatus.FAILED and self.error is not None:
            raise ValueError("Only a failed step may include an error.")


@dataclass(frozen=True, slots=True)
class TaskExecutionResult:
    """Structured outcome of executing a complete task."""

    issue_number: int | None
    status: ExecutionStatus
    steps: tuple[StepResult, ...] = ()
    attempts: int = 0
    error: str | None = None
    lifecycle_status: TaskStatus | None = None

    def __post_init__(self) -> None:
        """Validate the consistency of the task result."""
        if self.issue_number is not None and self.issue_number < 1:
            raise ValueError("Issue number must be positive.")

        if self.attempts < 0:
            raise ValueError("Attempts must not be negative.")

        if not isinstance(self.steps, tuple):
            raise TypeError("Steps must be provided as a tuple.")

        if self.status == ExecutionStatus.FAILED and not self.error:
            raise ValueError("A failed execution must include an error.")

        if self.status != ExecutionStatus.FAILED and self.error is not None:
            raise ValueError("Only a failed execution may include an error.")
