"""Coordinate task execution stages and lifecycle transitions."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from codepilot.agents.execution_models import (
    ExecutionStatus,
    StepResult,
    TaskExecutionResult,
)
from codepilot.graph.lifecycle import (
    InvalidTransitionError,
    TaskStatus,
    transition_task,
)

StageFunction = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class ExecutionStage:
    """A named execution stage with an explicit lifecycle target."""

    name: str
    target_status: TaskStatus
    run: StageFunction

    def __post_init__(self) -> None:
        """Validate the stage definition."""
        if not self.name.strip():
            raise ValueError("Execution stage name must not be empty.")


class TaskExecutionEngine:
    """Execute ordered stages while tracking the task lifecycle."""

    def __init__(self, stages: Sequence[ExecutionStage]) -> None:
        """Initialize the engine with an explicit ordered stage sequence."""
        if not stages:
            raise ValueError("At least one execution stage is required.")

        names = [stage.name.strip() for stage in stages]

        if len(names) != len(set(names)):
            raise ValueError("Execution stage names must be unique.")

        self._stages = tuple(stages)

    def execute(
        self,
        *,
        issue_number: int | None,
        initial_status: TaskStatus = TaskStatus.TRIAGED,
    ) -> TaskExecutionResult:
        """Execute configured stages and return a structured result."""
        current_status = initial_status
        step_results: list[StepResult] = []

        for stage in self._stages:
            try:
                current_status = self._transition_if_needed(
                    current_status,
                    stage.target_status,
                )

                output = stage.run()

                step_results.append(
                    StepResult(
                        name=stage.name,
                        status=ExecutionStatus.SUCCEEDED,
                        output=self._format_output(output),
                    )
                )

            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"

                step_results.append(
                    StepResult(
                        name=stage.name,
                        status=ExecutionStatus.FAILED,
                        error=error_message,
                    )
                )

                current_status = self._mark_failed(current_status)

                return TaskExecutionResult(
                    issue_number=issue_number,
                    status=ExecutionStatus.FAILED,
                    steps=tuple(step_results),
                    attempts=1,
                    error=error_message,
                    lifecycle_status=current_status,
                )

        return TaskExecutionResult(
            issue_number=issue_number,
            status=ExecutionStatus.SUCCEEDED,
            steps=tuple(step_results),
            attempts=1,
            lifecycle_status=current_status,
        )

    @staticmethod
    def _transition_if_needed(
        current: TaskStatus,
        target: TaskStatus,
    ) -> TaskStatus:
        """Perform a valid lifecycle transition when a change is needed."""
        if current == target:
            return current

        return transition_task(current, target)

    @staticmethod
    def _mark_failed(current: TaskStatus) -> TaskStatus:
        """Move a nonterminal task to FAILED when the transition is valid."""
        if current in {TaskStatus.DONE, TaskStatus.FAILED}:
            return current

        try:
            return transition_task(current, TaskStatus.FAILED)
        except InvalidTransitionError:
            return current

    @staticmethod
    def _format_output(output: Any) -> str:
        """Convert stage output into a readable result string."""
        if output is None:
            return ""

        if isinstance(output, str):
            return output

        return str(output)
