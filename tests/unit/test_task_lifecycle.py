"""Tests for task lifecycle transitions."""

import pytest

from codepilot.graph.lifecycle import (
    InvalidTransitionError,
    TaskStatus,
    transition_task,
)


def test_happy_path_transitions():
    status = TaskStatus.TRIAGED

    for next_status in (
        TaskStatus.EXPLORING,
        TaskStatus.IMPLEMENTING,
        TaskStatus.TESTING,
        TaskStatus.PR_OPENED,
        TaskStatus.DONE,
    ):
        status = transition_task(status, next_status)

    assert status == TaskStatus.DONE


def test_testing_can_return_to_implementation_for_retry():
    assert transition_task(TaskStatus.TESTING, TaskStatus.IMPLEMENTING) == TaskStatus.IMPLEMENTING


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (TaskStatus.TRIAGED, TaskStatus.DONE),
        (TaskStatus.EXPLORING, TaskStatus.PR_OPENED),
        (TaskStatus.IMPLEMENTING, TaskStatus.DONE),
        (TaskStatus.DONE, TaskStatus.IMPLEMENTING),
        (TaskStatus.FAILED, TaskStatus.EXPLORING),
    ],
)
def test_invalid_transitions_are_rejected(current, target):
    with pytest.raises(InvalidTransitionError):
        transition_task(current, target)


def test_failure_is_allowed_before_completion():
    assert transition_task(TaskStatus.IMPLEMENTING, TaskStatus.FAILED) == TaskStatus.FAILED
