"""Episodic session tracking and persistent session summaries (Component 5).

During a session every attempted task is tracked with: issue ID, task type,
files modified, outcome, and duration. When the session ends, one structured
summary is appended to ``sessions.jsonl``. At startup the last three summaries
for the repository are read so recently FAILED issues are not retried
automatically.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

SESSIONS_FILENAME = "sessions.jsonl"
RECENT_SESSION_LIMIT = 3
#: Outcomes that make an issue "recently failed". BLOCKED is excluded: the
#: guardrail check is cheap and deterministic, so it is simply re-applied.
RETRY_SUPPRESSING_OUTCOMES = frozenset({"FAILED"})


class SessionTracker:
    """Collect task records for the current session and persist a summary."""

    def __init__(self, *, repository: str, memory_directory: str | Path) -> None:
        if not repository.strip():
            raise ValueError("repository must not be empty.")
        self.repository = repository
        self.session_id = uuid4().hex
        self.started_at = datetime.now(UTC).isoformat()
        self.sessions_file = Path(memory_directory) / SESSIONS_FILENAME
        self._tasks: list[dict[str, Any]] = []
        self._lock = Lock()
        self._written = False

    def record(
        self,
        *,
        issue_number: int,
        title: str,
        task_type: str,
        outcome: str,
        files_modified: Sequence[str] = (),
        duration_seconds: float = 0.0,
        source: str = "github_issue",
        pull_request_url: str | None = None,
    ) -> dict[str, Any]:
        """Record one attempted task in this session."""
        entry = {
            "issue_number": issue_number,
            "title": title[:200],
            "task_type": task_type,
            "source": source,
            "outcome": outcome,
            "files_modified": sorted(set(files_modified)),
            "duration_seconds": round(max(duration_seconds, 0.0), 1),
            "pull_request_url": pull_request_url,
            "finished_at": datetime.now(UTC).isoformat(),
        }
        with self._lock:
            self._tasks.append(entry)
        return dict(entry)

    @property
    def tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(task) for task in self._tasks]

    def build_summary(self) -> dict[str, Any]:
        tasks = self.tasks
        outcomes: dict[str, int] = {}
        for task in tasks:
            outcomes[task["outcome"]] = outcomes.get(task["outcome"], 0) + 1
        return {
            "session_id": self.session_id,
            "repository": self.repository,
            "started_at": self.started_at,
            "ended_at": datetime.now(UTC).isoformat(),
            "task_count": len(tasks),
            "outcomes": outcomes,
            "failed_issue_numbers": sorted(
                {
                    task["issue_number"]
                    for task in tasks
                    if task["source"] == "github_issue"
                    and task["outcome"] in RETRY_SUPPRESSING_OUTCOMES
                }
            ),
            "tasks": tasks,
        }

    def write_summary(self) -> dict[str, Any] | None:
        """Append this session's summary once. Sessions with no tasks are skipped."""
        with self._lock:
            if self._written:
                return None
            self._written = True
        summary = self.build_summary()
        if not summary["tasks"]:
            return None
        self.sessions_file.parent.mkdir(parents=True, exist_ok=True)
        with self.sessions_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        logger.info(
            "Episodic memory: session summary written (%d task(s)) to %s.",
            summary["task_count"],
            self.sessions_file,
        )
        return summary


def load_recent_summaries(
    memory_directory: str | Path,
    *,
    repository: str,
    limit: int = RECENT_SESSION_LIMIT,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` most recent session summaries for ``repository``."""
    sessions_file = Path(memory_directory) / SESSIONS_FILENAME
    if not sessions_file.is_file():
        return []
    summaries: list[dict[str, Any]] = []
    for line in sessions_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            summary = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Ignoring a corrupt line in %s.", sessions_file)
            continue
        if isinstance(summary, dict) and summary.get("repository") == repository:
            summaries.append(summary)
    return summaries[-limit:]


def recently_failed_issue_numbers(summaries: Sequence[dict[str, Any]]) -> set[int]:
    """Collect GitHub issue numbers that FAILED in the given sessions."""
    failed: set[int] = set()
    for summary in summaries:
        for number in summary.get("failed_issue_numbers", []):
            if isinstance(number, int):
                failed.add(number)
    return failed
