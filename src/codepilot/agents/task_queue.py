"""Thread-safe queue for GitHub issues awaiting agent processing."""

from __future__ import annotations

from queue import Queue
from threading import Lock

from codepilot.github.models import GitHubIssue


class IssueTaskQueue:
    """Queue issues while preventing duplicate scheduling."""

    def __init__(self) -> None:
        """Initialize the queue and its tracking state."""
        self._queue: Queue[GitHubIssue] = Queue()
        self._lock = Lock()
        self._queued_issue_numbers: set[int] = set()
        self._active_issue_numbers: set[int] = set()

    def enqueue(self, issue: GitHubIssue) -> bool:
        """Enqueue an issue unless it is already queued or active.

        Returns:
            True if the issue was queued; False if it was already tracked.
        """
        with self._lock:
            if issue.number in self._queued_issue_numbers:
                return False

            if issue.number in self._active_issue_numbers:
                return False

            self._queued_issue_numbers.add(issue.number)
            self._queue.put(issue)
            return True

    def get(self, timeout: float | None = None) -> GitHubIssue:
        """Retrieve the next issue and mark it as active."""
        if timeout is None:
            issue = self._queue.get()
        else:
            issue = self._queue.get(timeout=timeout)

        with self._lock:
            self._queued_issue_numbers.remove(issue.number)
            self._active_issue_numbers.add(issue.number)

        return issue

    def complete(self, issue_number: int) -> bool:
        """Mark an active issue as completed.

        Returns:
            True if an active issue was released; False otherwise.
        """
        with self._lock:
            if issue_number not in self._active_issue_numbers:
                return False

            self._active_issue_numbers.remove(issue_number)
            self._queue.task_done()
            return True

    @property
    def queued_count(self) -> int:
        """Return the number of issues waiting to be processed."""
        with self._lock:
            return len(self._queued_issue_numbers)

    @property
    def active_issue_numbers(self) -> frozenset[int]:
        """Return an immutable snapshot of active issue numbers."""
        with self._lock:
            return frozenset(self._active_issue_numbers)

    @property
    def tracked_issue_numbers(self) -> frozenset[int]:
        """Return all queued and active issue numbers."""
        with self._lock:
            return frozenset(self._queued_issue_numbers | self._active_issue_numbers)
