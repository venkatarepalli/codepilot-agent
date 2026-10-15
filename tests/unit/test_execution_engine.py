"""Tests for the task execution engine."""

from unittest.mock import Mock

import pytest

from codepilot.agents.execution_engine import (
    ExecutionStage,
    TaskExecutionEngine,
)
from codepilot.agents.execution_models import ExecutionStatus
from codepilot.graph.lifecycle import TaskStatus


def make_stage(
    name: str,
    target_status: TaskStatus,
    output: str = "completed",
) -> ExecutionStage:
    """Create a stage with a predictable result."""
    return ExecutionStage(
        name=name,
        target_status=target_status,
        run=lambda: output,
    )


def test_executes_stages_in_order_and_records_lifecycle() -> None:
    """Stages execute in order and the final lifecycle state is recorded."""
    execution_order: list[str] = []

    stages = [
        ExecutionStage(
            name="repository_analysis",
            target_status=TaskStatus.EXPLORING,
            run=lambda: execution_order.append("exploration") or "repository mapped",
        ),
        ExecutionStage(
            name="code_changes",
            target_status=TaskStatus.IMPLEMENTING,
            run=lambda: execution_order.append("implementation") or "code updated",
        ),
        ExecutionStage(
            name="automated_tests",
            target_status=TaskStatus.TESTING,
            run=lambda: execution_order.append("testing") or "tests passed",
        ),
    ]

    result = TaskExecutionEngine(stages).execute(issue_number=42)

    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.lifecycle_status == TaskStatus.TESTING
    assert result.issue_number == 42
    assert result.attempts == 1
    assert execution_order == [
        "exploration",
        "implementation",
        "testing",
    ]
    assert [step.name for step in result.steps] == [
        "repository_analysis",
        "code_changes",
        "automated_tests",
    ]
    assert all(step.status == ExecutionStatus.SUCCEEDED for step in result.steps)


def test_captures_stage_output() -> None:
    """Stage output is preserved in the structured result."""
    stage = make_stage(
        "repository_analysis",
        TaskStatus.EXPLORING,
        "Found 12 Python files.",
    )

    result = TaskExecutionEngine([stage]).execute(issue_number=1)

    assert result.steps[0].output == "Found 12 Python files."


def test_converts_non_string_output_to_string() -> None:
    """Non-string stage output is converted to a readable string."""
    stage = make_stage(
        "repository_analysis",
        TaskStatus.EXPLORING,
        output="",
    )
    stage = ExecutionStage(
        name=stage.name,
        target_status=stage.target_status,
        run=lambda: {"files": 12},
    )

    result = TaskExecutionEngine([stage]).execute(issue_number=1)

    assert result.steps[0].output == "{'files': 12}"


def test_none_output_becomes_empty_string() -> None:
    """A stage returning None produces an empty output string."""
    stage = ExecutionStage(
        name="repository_analysis",
        target_status=TaskStatus.EXPLORING,
        run=lambda: None,
    )

    result = TaskExecutionEngine([stage]).execute(issue_number=1)

    assert result.steps[0].output == ""


def test_stops_after_stage_failure_and_marks_task_failed() -> None:
    """A failed stage stops execution and records the lifecycle failure."""
    later_stage = Mock(return_value="should not run")

    def fail() -> None:
        raise RuntimeError("implementation failed")

    stages = [
        make_stage(
            "repository_analysis",
            TaskStatus.EXPLORING,
        ),
        ExecutionStage(
            name="code_changes",
            target_status=TaskStatus.IMPLEMENTING,
            run=fail,
        ),
        ExecutionStage(
            name="automated_tests",
            target_status=TaskStatus.TESTING,
            run=later_stage,
        ),
    ]

    result = TaskExecutionEngine(stages).execute(issue_number=42)

    assert result.status == ExecutionStatus.FAILED
    assert result.lifecycle_status == TaskStatus.FAILED
    assert result.error == "RuntimeError: implementation failed"
    assert len(result.steps) == 2
    assert result.steps[-1].status == ExecutionStatus.FAILED
    assert result.steps[-1].error == "RuntimeError: implementation failed"
    later_stage.assert_not_called()


def test_direct_task_can_execute_without_github_issue_number() -> None:
    """Direct TUI tasks do not require a GitHub issue number."""
    stage = make_stage(
        "repository_analysis",
        TaskStatus.EXPLORING,
    )

    result = TaskExecutionEngine([stage]).execute(issue_number=None)

    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.issue_number is None
    assert result.lifecycle_status == TaskStatus.EXPLORING


def test_requires_at_least_one_stage() -> None:
    """An engine cannot be created without stages."""
    with pytest.raises(ValueError, match="At least one execution stage"):
        TaskExecutionEngine([])


def test_requires_unique_stage_names() -> None:
    """Stage names must be unique."""
    stages = [
        make_stage("analysis", TaskStatus.EXPLORING),
        make_stage("analysis", TaskStatus.EXPLORING),
    ]

    with pytest.raises(ValueError, match="must be unique"):
        TaskExecutionEngine(stages)


def test_requires_nonempty_stage_name() -> None:
    """A stage cannot have a blank name."""
    with pytest.raises(ValueError, match="must not be empty"):
        ExecutionStage(
            name="   ",
            target_status=TaskStatus.EXPLORING,
            run=lambda: "completed",
        )


def test_supports_valid_retry_transition_from_testing_to_implementing() -> None:
    """The lifecycle can return to implementation for a test-fix retry."""
    stages = [
        make_stage(
            "repository_analysis",
            TaskStatus.EXPLORING,
        ),
        make_stage(
            "code_changes",
            TaskStatus.IMPLEMENTING,
        ),
        make_stage(
            "automated_tests",
            TaskStatus.TESTING,
        ),
        make_stage(
            "test_fix",
            TaskStatus.IMPLEMENTING,
        ),
        make_stage(
            "rerun_tests",
            TaskStatus.TESTING,
        ),
    ]

    result = TaskExecutionEngine(stages).execute(issue_number=42)

    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.lifecycle_status == TaskStatus.TESTING
    assert len(result.steps) == 5


def test_invalid_transition_fails_without_running_stage() -> None:
    """An invalid lifecycle transition fails before the stage runs."""
    stage_function = Mock(return_value="should not run")
    stage = ExecutionStage(
        name="open_pull_request",
        target_status=TaskStatus.PR_OPENED,
        run=stage_function,
    )

    result = TaskExecutionEngine([stage]).execute(
        issue_number=42,
        initial_status=TaskStatus.TRIAGED,
    )

    assert result.status == ExecutionStatus.FAILED
    assert result.lifecycle_status == TaskStatus.FAILED
    assert "Cannot transition task from TRIAGED to PR_OPENED" in result.error
    assert result.steps[0].status == ExecutionStatus.FAILED
    stage_function.assert_not_called()


def test_stage_can_run_when_already_in_its_target_state() -> None:
    """A stage can execute without a redundant lifecycle transition."""
    stage_function = Mock(return_value="tests passed")
    stage = ExecutionStage(
        name="automated_tests",
        target_status=TaskStatus.TESTING,
        run=stage_function,
    )

    result = TaskExecutionEngine([stage]).execute(
        issue_number=42,
        initial_status=TaskStatus.TESTING,
    )

    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.lifecycle_status == TaskStatus.TESTING
    stage_function.assert_called_once_with()
