"""Unit tests for the CodePilot issue task queue."""

from queue import Empty

import pytest

from codepilot.agents.task_queue import IssueTaskQueue
from codepilot.github.models import GitHubIssue


@pytest.fixture
def issue() -> GitHubIssue:
    """Return a sample GitHub issue."""
    return GitHubIssue(number=42, title="Fix a typo")


@pytest.fixture
def task_queue() -> IssueTaskQueue:
    """Return an empty task queue."""
    return IssueTaskQueue()


def test_enqueue_accepts_new_issue(
    task_queue: IssueTaskQueue,
    issue: GitHubIssue,
) -> None:
    assert task_queue.enqueue(issue) is True
    assert task_queue.queued_count == 1


def test_enqueue_rejects_duplicate_queued_issue(
    task_queue: IssueTaskQueue,
    issue: GitHubIssue,
) -> None:
    assert task_queue.enqueue(issue) is True
    assert task_queue.enqueue(issue) is False
    assert task_queue.queued_count == 1


def test_get_returns_next_issue(
    task_queue: IssueTaskQueue,
    issue: GitHubIssue,
) -> None:
    task_queue.enqueue(issue)

    result = task_queue.get(timeout=0.1)

    assert result == issue
    assert task_queue.queued_count == 0
    assert task_queue.active_issue_numbers == frozenset({42})


def test_active_issue_cannot_be_queued_again(
    task_queue: IssueTaskQueue,
    issue: GitHubIssue,
) -> None:
    task_queue.enqueue(issue)
    task_queue.get(timeout=0.1)

    assert task_queue.enqueue(issue) is False


def test_complete_releases_active_issue(
    task_queue: IssueTaskQueue,
    issue: GitHubIssue,
) -> None:
    task_queue.enqueue(issue)
    task_queue.get(timeout=0.1)

    assert task_queue.complete(42) is True
    assert task_queue.active_issue_numbers == frozenset()
    assert task_queue.tracked_issue_numbers == frozenset()


def test_completed_issue_can_be_queued_again(
    task_queue: IssueTaskQueue,
    issue: GitHubIssue,
) -> None:
    task_queue.enqueue(issue)
    task_queue.get(timeout=0.1)
    task_queue.complete(42)

    assert task_queue.enqueue(issue) is True


def test_complete_rejects_unknown_or_inactive_issue(
    task_queue: IssueTaskQueue,
) -> None:
    assert task_queue.complete(999) is False


def test_get_times_out_when_queue_is_empty(
    task_queue: IssueTaskQueue,
) -> None:
    with pytest.raises(Empty):
        task_queue.get(timeout=0.01)


def test_tracked_issues_include_queued_and_active(
    task_queue: IssueTaskQueue,
) -> None:
    first = GitHubIssue(number=1, title="Fix a typo")
    second = GitHubIssue(number=2, title="Fix another typo")

    task_queue.enqueue(first)
    task_queue.enqueue(second)
    task_queue.get(timeout=0.1)

    assert task_queue.tracked_issue_numbers == frozenset({1, 2})
    assert task_queue.queued_count == 1
