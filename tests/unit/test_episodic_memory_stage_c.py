"""Tests for LangGraph Store-backed episodic memory."""

import pytest
from langgraph.store.memory import InMemoryStore

from codepilot.memory.episodic_memory import EpisodicMemory


def record(memory, issue_number, repository="example/repo", status="DONE"):
    return memory.record_task(
        issue_number=issue_number,
        repository=repository,
        category="bug_fix",
        status=status,
        summary=f"Task {issue_number} summary",
    )


def test_records_and_returns_recent_tasks_newest_first():
    memory = EpisodicMemory(store=InMemoryStore(), session_id="test-session")
    record(memory, 1)
    record(memory, 2)
    record(memory, 3)

    recent = memory.get_recent_tasks(limit=2)

    assert [item["issue_number"] for item in recent] == [3, 2]


def test_recent_tasks_can_be_filtered_by_repository():
    memory = EpisodicMemory(store=InMemoryStore(), session_id="test-session")
    record(memory, 1, repository="example/one")
    record(memory, 2, repository="example/two")
    record(memory, 3, repository="example/one")

    recent = memory.get_recent_tasks(limit=3, repository="example/one")

    assert [item["issue_number"] for item in recent] == [3, 1]


def test_session_namespaces_are_isolated_when_store_is_shared():
    store = InMemoryStore()
    first = EpisodicMemory(store=store, session_id="session-one")
    second = EpisodicMemory(store=store, session_id="session-two")
    record(first, 1)

    assert len(first.get_recent_tasks()) == 1
    assert second.get_recent_tasks() == []


def test_clear_only_clears_the_current_session():
    store = InMemoryStore()
    first = EpisodicMemory(store=store, session_id="session-one")
    second = EpisodicMemory(store=store, session_id="session-two")
    record(first, 1)
    record(second, 2)

    first.clear()

    assert first.get_recent_tasks() == []
    assert [item["issue_number"] for item in second.get_recent_tasks()] == [2]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"issue_number": 0}, "issue_number must be positive"),
        ({"repository": " "}, "repository must not be empty"),
        ({"category": " "}, "category must not be empty"),
        ({"status": "RUNNING"}, "status must be DONE or FAILED"),
        ({"summary": " "}, "summary must not be empty"),
    ],
)
def test_record_task_validates_inputs(kwargs, message):
    memory = EpisodicMemory(store=InMemoryStore(), session_id="validation")
    values = {
        "issue_number": 42,
        "repository": "example/repo",
        "category": "bug_fix",
        "status": "DONE",
        "summary": "Completed task",
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        memory.record_task(**values)


def test_limit_must_be_positive():
    memory = EpisodicMemory(store=InMemoryStore(), session_id="validation")

    with pytest.raises(ValueError, match="limit must be positive"):
        memory.get_recent_tasks(limit=0)
