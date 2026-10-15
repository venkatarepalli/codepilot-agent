"""Tests for task-scoped working memory."""

import pytest

from codepilot.memory.working_memory import WorkingMemory


@pytest.fixture
def memory():
    return WorkingMemory(
        issue_number=42,
        repository="owner/repo",
        category="bug_fix",
        issue={"title": "Fix validation"},
    )


def test_initial_state(memory):
    snapshot = memory.snapshot()

    assert snapshot["issue_number"] == 42
    assert snapshot["repository"] == "owner/repo"
    assert snapshot["category"] == "bug_fix"
    assert snapshot["status"] == "TRIAGED"
    assert snapshot["retry_count"] == 0


def test_update_status(memory):
    memory.update_status("TESTING")

    assert memory.snapshot()["status"] == "TESTING"


def test_set_repository_map(memory):
    memory.set_repository_map({"src/app.py": "Application entry point"})

    assert memory.snapshot()["repository_map"] == {"src/app.py": "Application entry point"}


def test_set_selected_files_removes_duplicates(memory):
    memory.set_selected_files(["src/app.py", "tests/test_app.py", "src/app.py"])

    assert memory.snapshot()["selected_files"] == [
        "src/app.py",
        "tests/test_app.py",
    ]


def test_set_proposed_diff(memory):
    diff = "--- a/src/app.py\n+++ b/src/app.py\n"

    memory.set_proposed_diff(diff)

    assert memory.snapshot()["proposed_diff"] == diff


def test_record_test_results(memory):
    memory.record_test_result(
        passed=False,
        output="One test failed.",
        retry_count=1,
    )
    memory.record_test_result(
        passed=True,
        output="All tests passed.",
        retry_count=1,
    )

    snapshot = memory.snapshot()

    assert len(snapshot["test_results"]) == 2
    assert snapshot["test_results"][0]["passed"] is False
    assert snapshot["test_results"][1]["passed"] is True
    assert snapshot["retry_count"] == 1


def test_set_memory_context(memory):
    memory.set_memory_context(
        episodic=[{"summary": "Previous fix"}],
        semantic=[{"lesson": "Validate inputs"}],
    )

    snapshot = memory.snapshot()

    assert snapshot["episodic_context"] == [{"summary": "Previous fix"}]
    assert snapshot["semantic_context"] == [{"lesson": "Validate inputs"}]


def test_snapshot_is_independent_of_internal_lists(memory):
    snapshot = memory.snapshot()
    snapshot["selected_files"].append("external.py")
    snapshot["issue"]["title"] = "Changed outside"

    current = memory.snapshot()

    assert current["selected_files"] == []
    assert current["issue"]["title"] == "Fix validation"


def test_clear_removes_task_working_data(memory):
    memory.set_selected_files(["src/app.py"])
    memory.set_proposed_diff("sample diff")
    memory.record_test_result(passed=False, retry_count=2)
    memory.set_memory_context(
        episodic=[{"summary": "Old task"}],
        semantic=[{"lesson": "Old lesson"}],
    )
    memory.update_status("FAILED")

    memory.clear()

    snapshot = memory.snapshot()

    assert snapshot["issue_number"] == 42
    assert snapshot["repository"] == "owner/repo"
    assert snapshot["status"] == "FAILED"
    assert snapshot["issue"] == {}
    assert snapshot["selected_files"] == []
    assert snapshot["proposed_diff"] == ""
    assert snapshot["test_results"] == []
    assert snapshot["retry_count"] == 0
    assert snapshot["episodic_context"] == []
    assert snapshot["semantic_context"] == []


@pytest.mark.parametrize(
    ("issue_number", "repository", "category", "message"),
    [
        (0, "owner/repo", "bug_fix", "issue_number must be positive"),
        (1, " ", "bug_fix", "repository must not be empty"),
        (1, "owner/repo", " ", "category must not be empty"),
    ],
)
def test_invalid_task_identity(
    issue_number,
    repository,
    category,
    message,
):
    with pytest.raises(ValueError, match=message):
        WorkingMemory(
            issue_number=issue_number,
            repository=repository,
            category=category,
        )


def test_reject_empty_status(memory):
    with pytest.raises(ValueError, match="status must not be empty"):
        memory.update_status(" ")


def test_reject_empty_selected_file_path(memory):
    with pytest.raises(ValueError, match="Selected file paths"):
        memory.set_selected_files(["src/app.py", " "])


def test_reject_negative_retry_count(memory):
    with pytest.raises(ValueError, match="retry_count must not be negative"):
        memory.record_test_result(passed=False, retry_count=-1)
