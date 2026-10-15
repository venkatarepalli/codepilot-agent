"""Tests for Orchestrator integration with the task execution engine."""

from unittest.mock import MagicMock

import pytest

from codepilot.agents.execution_engine import ExecutionStage
from codepilot.agents.execution_models import ExecutionStatus
from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.github.models import GitHubIssue
from codepilot.graph.lifecycle import TaskStatus


@pytest.fixture
def issue() -> GitHubIssue:
    """Return a sample GitHub issue."""
    return GitHubIssue(
        number=42,
        title="Fix login crash",
        body="The application crashes during login.",
        url="https://github.com/example/repo/issues/42",
        labels=("bug",),
        reporter="reporter",
    )


def make_orchestrator(**kwargs) -> CodePilotOrchestrator:
    """Create an orchestrator with a mocked DeepAgents root agent."""
    return CodePilotOrchestrator(
        model="test-model",
        agent_factory=MagicMock(return_value=MagicMock()),
        **kwargs,
    )


def test_execute_requires_execution_stages_factory(
    issue: GitHubIssue,
) -> None:
    """Execution must fail clearly when no stage factory is configured."""
    orchestrator = make_orchestrator()

    with pytest.raises(
        RuntimeError,
        match="execution stages factory must be configured",
    ):
        orchestrator.execute(issue)


def test_execute_runs_stages_and_returns_structured_result(
    issue: GitHubIssue,
) -> None:
    """Configured stages run in order and produce a structured result."""
    execution_order: list[str] = []
    received_tasks: list[dict] = []

    def build_stages(task: dict) -> list[ExecutionStage]:
        received_tasks.append(task)

        return [
            ExecutionStage(
                name="repository_exploration",
                target_status=TaskStatus.EXPLORING,
                run=lambda: execution_order.append("exploration") or "Repository explored",
            ),
            ExecutionStage(
                name="implementation",
                target_status=TaskStatus.IMPLEMENTING,
                run=lambda: execution_order.append("implementation") or "Implementation completed",
            ),
            ExecutionStage(
                name="testing",
                target_status=TaskStatus.TESTING,
                run=lambda: execution_order.append("testing") or "Tests passed",
            ),
        ]

    orchestrator = make_orchestrator(
        execution_stages_factory=build_stages,
    )

    result = orchestrator.execute(issue)

    assert len(received_tasks) == 1

    prepared_task = received_tasks[0]
    assert prepared_task["issue"] == issue
    assert prepared_task["status"] == TaskStatus.TRIAGED
    assert "GitHub issue #42" in prepared_task["prompt"]

    assert execution_order == [
        "exploration",
        "implementation",
        "testing",
    ]

    assert result.issue_number == 42
    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.lifecycle_status == TaskStatus.TESTING
    assert result.attempts == 1
    assert result.error is None

    assert [step.name for step in result.steps] == [
        "repository_exploration",
        "implementation",
        "testing",
    ]

    assert [step.output for step in result.steps] == [
        "Repository explored",
        "Implementation completed",
        "Tests passed",
    ]

    assert all(step.status == ExecutionStatus.SUCCEEDED for step in result.steps)


def test_execute_stops_and_returns_failed_result_when_stage_fails(
    issue: GitHubIssue,
) -> None:
    """A failed stage stops subsequent execution and records the error."""
    later_stage = MagicMock(return_value="This stage must not run")

    def fail_implementation() -> None:
        raise RuntimeError("Simulated implementation failure")

    def build_stages(task: dict) -> list[ExecutionStage]:
        assert task["issue"] == issue

        return [
            ExecutionStage(
                name="repository_exploration",
                target_status=TaskStatus.EXPLORING,
                run=lambda: "Repository explored",
            ),
            ExecutionStage(
                name="implementation",
                target_status=TaskStatus.IMPLEMENTING,
                run=fail_implementation,
            ),
            ExecutionStage(
                name="testing",
                target_status=TaskStatus.TESTING,
                run=later_stage,
            ),
        ]

    orchestrator = make_orchestrator(
        execution_stages_factory=build_stages,
    )

    result = orchestrator.execute(issue)

    assert result.issue_number == 42
    assert result.status == ExecutionStatus.FAILED
    assert result.lifecycle_status == TaskStatus.FAILED
    assert result.attempts == 1
    assert "Simulated implementation failure" in result.error

    assert [step.name for step in result.steps] == [
        "repository_exploration",
        "implementation",
    ]

    assert result.steps[0].status == ExecutionStatus.SUCCEEDED
    assert result.steps[1].status == ExecutionStatus.FAILED
    assert "Simulated implementation failure" in result.steps[1].error

    later_stage.assert_not_called()
