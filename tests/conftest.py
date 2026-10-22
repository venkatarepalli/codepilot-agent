"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_session_memory(tmp_path, monkeypatch):
    """Never let tests write episodic session summaries into the real project."""
    monkeypatch.setenv("CODEPILOT_MEMORY_DIR", str(tmp_path / "codepilot-memory"))
