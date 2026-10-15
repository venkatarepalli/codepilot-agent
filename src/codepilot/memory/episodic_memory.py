"""Session-scoped episodic memory backed by LangGraph's Store API."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from langgraph.store.memory import InMemoryStore


class EpisodicMemory:
    """Store task outcomes in a LangGraph Store for the current session.

    The Store is intentionally session-scoped: it retains task history while
    the same store/session is in use, but the default InMemoryStore does not
    persist history across process restarts. Long-lived reusable lessons belong
    in SemanticMemory instead.

    ``storage_path`` remains accepted for compatibility with existing startup
    configuration. It is not used for JSON persistence.
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        *,
        store: Any | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize episodic memory with an injectable LangGraph Store."""
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self._store = store if store is not None else InMemoryStore()
        self._session_id = session_id or uuid4().hex
        if not self._session_id.strip():
            raise ValueError("session_id must not be empty.")
        self._namespace = ("codepilot", "episodic", self._session_id)
        self._lock = RLock()

    @property
    def store(self) -> Any:
        """Expose the configured Store for diagnostics and controlled reuse."""
        return self._store

    @property
    def session_id(self) -> str:
        """Return the namespace identifier for this episodic-memory session."""
        return self._session_id

    def record_task(
        self,
        *,
        issue_number: int,
        repository: str,
        category: str,
        status: str,
        summary: str,
    ) -> dict[str, Any]:
        """Record a terminal task outcome and return the stored record."""
        if issue_number < 1:
            raise ValueError("issue_number must be positive.")
        if not repository.strip():
            raise ValueError("repository must not be empty.")
        if not category.strip():
            raise ValueError("category must not be empty.")
        if status not in {"DONE", "FAILED"}:
            raise ValueError("status must be DONE or FAILED.")
        if not summary.strip():
            raise ValueError("summary must not be empty.")

        record: dict[str, Any] = {
            "issue_number": issue_number,
            "repository": repository,
            "category": category,
            "status": status,
            "summary": summary,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        key = f"{record['recorded_at']}:{uuid4().hex}"

        with self._lock:
            self._store.put(self._namespace, key, record)

        return record.copy()

    def get_recent_tasks(
        self,
        *,
        limit: int = 3,
        repository: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the newest task experiences, optionally filtered by repository."""
        if limit < 1:
            raise ValueError("limit must be positive.")

        with self._lock:
            items = self._store.search(self._namespace, limit=10_000)

        records = [dict(item.value) for item in items]
        if repository is not None:
            records = [record for record in records if record.get("repository") == repository]

        records.sort(key=lambda record: record.get("recorded_at", ""), reverse=True)
        return [record.copy() for record in records[:limit]]

    def clear(self) -> None:
        """Delete all task outcomes in this session namespace."""
        with self._lock:
            items = self._store.search(self._namespace, limit=10_000)
            for item in items:
                self._store.delete(self._namespace, item.key)
