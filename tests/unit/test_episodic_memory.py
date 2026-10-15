"""Tests for session-scoped episodic memory."""

import pytest
from langgraph.store.memory import InMemoryStore

from codepilot.memory.episodic_memory import EpisodicMemory


@pytest.fixture
def memory():
    """Create isolated episodic memory for a test."""
    return EpisodicMemory(
        store=InMemoryStore(),
        session_id="test-session",
    )


def record_task(
    memory,
    issue_number,
    repository="owner/repo",
    status="DONE",
):
    """Record a representative task outcome."""
    return memory.record_task(
        issue_number=issue_number,
        repository=repository,
        category="bug_fix",
        status=status,
        summary=f"Completed issue {issue_number}.",
    )


def test_initialization_creates_session_scoped_store(memory):
    assert memory.store is not None
    assert memory.session_id == "test-session"
    assert memory.get_recent_tasks() == []


def test_record_task_stores_outcome(memory):
    record = record_task(memory, 12)

    assert record["issue_number"] == 12
    assert record["repository"] == "owner/repo"
    assert record["status"] == "DONE"
    assert memory.get_recent_tasks() == [record]


def test_recent_tasks_returns_newest_first(memory):
    for issue_number in (1, 2, 3, 4):
        record_task(memory, issue_number)

    recent = memory.get_recent_tasks()

    assert [record["issue_number"] for record in recent] == [4, 3, 2]


def test_recent_tasks_filters_by_repository(memory):
    record_task(memory, 1, repository="owner/first")
    record_task(memory, 2, repository="owner/second")
    record_task(memory, 3, repository="owner/first")

    recent = memory.get_recent_tasks(repository="owner/first")

    assert [record["issue_number"] for record in recent] == [3, 1]


def test_separate_instances_do_not_share_default_session_store():
    first = EpisodicMemory(session_id="first")
    second = EpisodicMemory(session_id="second")

    record_task(first, 25)

    assert first.get_recent_tasks()[0]["issue_number"] == 25
    assert second.get_recent_tasks() == []


def test_clear_removes_all_records(memory):
    record_task(memory, 5, status="FAILED")

    memory.clear()

    assert memory.get_recent_tasks() == []


@pytest.mark.parametrize("status", ["TRIAGED", "TESTING", "PR_OPENED"])
def test_record_task_rejects_nonterminal_status(memory, status):
    with pytest.raises(ValueError, match="status must be DONE or FAILED"):
        record_task(memory, 1, status=status)


def test_record_task_rejects_invalid_issue_number(memory):
    with pytest.raises(ValueError, match="issue_number must be positive"):
        record_task(memory, 0)


def test_recent_tasks_rejects_invalid_limit(memory):
    with pytest.raises(ValueError, match="limit must be positive"):
        memory.get_recent_tasks(limit=0)
