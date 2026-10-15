"""Task lifecycle states and validated state transitions."""

from enum import StrEnum


class TaskStatus(StrEnum):
    """Lifecycle states for a CodePilot task."""

    TRIAGED = "TRIAGED"
    EXPLORING = "EXPLORING"
    IMPLEMENTING = "IMPLEMENTING"
    TESTING = "TESTING"
    PR_OPENED = "PR_OPENED"
    DONE = "DONE"
    FAILED = "FAILED"


class InvalidTransitionError(ValueError):
    """Raised when a task attempts an invalid lifecycle transition."""


_ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.TRIAGED: frozenset({TaskStatus.EXPLORING, TaskStatus.FAILED}),
    TaskStatus.EXPLORING: frozenset({TaskStatus.IMPLEMENTING, TaskStatus.FAILED}),
    TaskStatus.IMPLEMENTING: frozenset({TaskStatus.TESTING, TaskStatus.FAILED}),
    TaskStatus.TESTING: frozenset(
        {
            TaskStatus.IMPLEMENTING,
            TaskStatus.PR_OPENED,
            TaskStatus.FAILED,
        }
    ),
    TaskStatus.PR_OPENED: frozenset({TaskStatus.DONE, TaskStatus.FAILED}),
    TaskStatus.DONE: frozenset(),
    TaskStatus.FAILED: frozenset(),
}


def transition_task(
    current: TaskStatus,
    target: TaskStatus,
) -> TaskStatus:
    """Validate a transition and return the resulting task status."""
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise InvalidTransitionError(
            f"Cannot transition task from {current.value} to {target.value}."
        )

    return target
