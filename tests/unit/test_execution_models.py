"""Tests for CodePilot task execution models."""

from dataclasses import FrozenInstanceError

import pytest

from codepilot.agents.execution_models import (
    ExecutionStatus,
    StepResult,
    TaskExecutionResult,
)


def test_execution_status_values() -> None:
    """Execution statuses expose stable string values."""
    assert ExecutionStatus.PENDING == "pending"
    assert ExecutionStatus.RUNNING == "running"
    assert ExecutionStatus.SUCCEEDED == "succeeded"
    assert ExecutionStatus.FAILED == "failed"


def test_successful_step_result() -> None:
    """A successful step can contain output."""
    result = StepResult(
        name="implementation",
        status=ExecutionStatus.SUCCEEDED,
        output="Changes implemented successfully.",
    )

    assert result.name == "implementation"
    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output == "Changes implemented successfully."
    assert result.error is None


def test_failed_step_requires_error() -> None:
    """A failed step must explain the failure."""
    with pytest.raises(ValueError, match="must include an error"):
        StepResult(
            name="testing",
            status=ExecutionStatus.FAILED,
        )


def test_non_failed_step_rejects_error() -> None:
    """Non-failed steps must not carry failure errors."""
    with pytest.raises(ValueError, match="Only a failed step"):
        StepResult(
            name="testing",
            status=ExecutionStatus.SUCCEEDED,
            error="Unexpected error",
        )


def test_step_name_must_not_be_empty() -> None:
    """Step names must contain non-whitespace characters."""
    with pytest.raises(ValueError, match="must not be empty"):
        StepResult(
            name="   ",
            status=ExecutionStatus.PENDING,
        )


def test_step_result_is_immutable() -> None:
    """A step result cannot be modified after creation."""
    result = StepResult(
        name="testing",
        status=ExecutionStatus.SUCCEEDED,
    )

    with pytest.raises(FrozenInstanceError):
        result.output = "Modified output"  # type: ignore[misc]


def test_successful_task_execution_result() -> None:
    """A task result aggregates successful execution steps."""
    steps = (
        StepResult(
            name="exploration",
            status=ExecutionStatus.SUCCEEDED,
            output="Relevant files identified.",
        ),
        StepResult(
            name="testing",
            status=ExecutionStatus.SUCCEEDED,
            output="All tests passed.",
        ),
    )

    result = TaskExecutionResult(
        issue_number=42,
        status=ExecutionStatus.SUCCEEDED,
        steps=steps,
        attempts=1,
    )

    assert result.issue_number == 42
    assert result.status == ExecutionStatus.SUCCEEDED
    assert len(result.steps) == 2
    assert result.attempts == 1
    assert result.error is None


def test_failed_task_requires_error() -> None:
    """A failed task execution must include an explanation."""
    with pytest.raises(ValueError, match="must include an error"):
        TaskExecutionResult(
            issue_number=42,
            status=ExecutionStatus.FAILED,
        )


def test_non_failed_task_rejects_error() -> None:
    """A non-failed task cannot carry a failure error."""
    with pytest.raises(ValueError, match="Only a failed execution"):
        TaskExecutionResult(
            issue_number=42,
            status=ExecutionStatus.SUCCEEDED,
            error="Unexpected error",
        )


@pytest.mark.parametrize("issue_number", [0, -1])
def test_issue_number_must_be_positive(
    issue_number: int,
) -> None:
    """Provided issue numbers must be positive."""
    with pytest.raises(ValueError, match="must be positive"):
        TaskExecutionResult(
            issue_number=issue_number,
            status=ExecutionStatus.PENDING,
        )


def test_issue_number_can_be_absent_for_direct_tasks() -> None:
    """Direct TUI tasks need not have a GitHub issue number."""
    result = TaskExecutionResult(
        issue_number=None,
        status=ExecutionStatus.PENDING,
    )

    assert result.issue_number is None


def test_attempt_count_cannot_be_negative() -> None:
    """Execution attempts cannot be negative."""
    with pytest.raises(ValueError, match="must not be negative"):
        TaskExecutionResult(
            issue_number=42,
            status=ExecutionStatus.PENDING,
            attempts=-1,
        )


def test_steps_must_be_a_tuple() -> None:
    """Aggregate execution steps must be immutable tuples."""
    with pytest.raises(TypeError, match="must be provided as a tuple"):
        TaskExecutionResult(
            issue_number=42,
            status=ExecutionStatus.PENDING,
            steps=[],  # type: ignore[arg-type]
        )


def test_task_execution_result_is_immutable() -> None:
    """An aggregate result cannot be modified after creation."""
    result = TaskExecutionResult(
        issue_number=42,
        status=ExecutionStatus.PENDING,
    )

    with pytest.raises(FrozenInstanceError):
        result.attempts = 2  # type: ignore[misc]
