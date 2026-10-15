"""Task-scoped working memory for CodePilot."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass
class WorkingMemory:
    """Maintain mutable state for one active coding task.

    Each instance belongs to a single task. Create a new instance for
    every issue to prevent state from leaking between tasks.
    """

    issue_number: int
    repository: str
    category: str
    issue: dict[str, Any] = field(default_factory=dict)
    repository_map: dict[str, Any] = field(default_factory=dict)
    selected_files: list[str] = field(default_factory=list)
    proposed_diff: str = ""
    test_results: list[dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    status: str = "TRIAGED"
    episodic_context: list[dict[str, Any]] = field(default_factory=list)
    semantic_context: list[dict[str, Any]] = field(default_factory=list)
    _lock: RLock = field(
        default_factory=RLock,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate task identity and initial state."""
        if self.issue_number < 1:
            raise ValueError("issue_number must be positive.")

        if not self.repository.strip():
            raise ValueError("repository must not be empty.")

        if not self.category.strip():
            raise ValueError("category must not be empty.")

        if self.retry_count < 0:
            raise ValueError("retry_count must not be negative.")

    def update_status(self, status: str) -> None:
        """Update the task's recorded lifecycle status."""
        if not status.strip():
            raise ValueError("status must not be empty.")

        with self._lock:
            self.status = status

    def set_repository_map(self, repository_map: dict[str, Any]) -> None:
        """Store repository-map information for the active task."""
        with self._lock:
            self.repository_map = dict(repository_map)

    def set_selected_files(self, files: list[str]) -> None:
        """Replace the selected file paths for the active task."""
        if any(not path.strip() for path in files):
            raise ValueError("Selected file paths must not be empty.")

        with self._lock:
            self.selected_files = list(dict.fromkeys(files))

    def set_proposed_diff(self, proposed_diff: str) -> None:
        """Store the current proposed diff."""
        with self._lock:
            self.proposed_diff = proposed_diff

    def record_test_result(
        self,
        *,
        passed: bool,
        output: str = "",
        retry_count: int | None = None,
    ) -> None:
        """Record a test attempt and optionally update the retry count."""
        if retry_count is not None and retry_count < 0:
            raise ValueError("retry_count must not be negative.")

        with self._lock:
            if retry_count is not None:
                self.retry_count = retry_count

            self.test_results.append(
                {
                    "passed": passed,
                    "output": output,
                    "retry_count": self.retry_count,
                }
            )

    def set_memory_context(
        self,
        *,
        episodic: list[dict[str, Any]] | None = None,
        semantic: list[dict[str, Any]] | None = None,
    ) -> None:
        """Attach retrieved historical experiences and lessons."""
        with self._lock:
            if episodic is not None:
                self.episodic_context = [dict(item) for item in episodic]

            if semantic is not None:
                self.semantic_context = [dict(item) for item in semantic]

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot for explicit subagent handoff."""
        with self._lock:
            return {
                "issue_number": self.issue_number,
                "repository": self.repository,
                "category": self.category,
                "issue": dict(self.issue),
                "repository_map": dict(self.repository_map),
                "selected_files": list(self.selected_files),
                "proposed_diff": self.proposed_diff,
                "test_results": [dict(item) for item in self.test_results],
                "retry_count": self.retry_count,
                "status": self.status,
                "episodic_context": [dict(item) for item in self.episodic_context],
                "semantic_context": [dict(item) for item in self.semantic_context],
            }

    def clear(self) -> None:
        """Clear task-specific working data after task completion or failure."""
        with self._lock:
            self.issue.clear()
            self.repository_map.clear()
            self.selected_files.clear()
            self.proposed_diff = ""
            self.test_results.clear()
            self.retry_count = 0
            self.episodic_context.clear()
            self.semantic_context.clear()
