"""Tests for merge-triggered semantic lesson recording."""

from unittest.mock import MagicMock

import pytest

from codepilot.github.merge_monitor import PullRequestMergeMonitor
from codepilot.github.models import GitHubIssue
from codepilot.github.pr_workflow import PullRequestResult


@pytest.fixture
def task_data():
    issue = GitHubIssue(
        number=42, title="Fix a typo", url="https://github.com/example/repo/issues/42"
    )
    coder_result = MagicMock()
    coder_result.issue_number = 42
    pull_request = PullRequestResult(
        branch="codepilot/issue-42-fix-a-typo",
        pull_number=101,
        pull_url="https://github.com/example/repo/pull/101",
        title="[CodePilot] Fix a typo",
        base_branch="main",
    )
    return issue, coder_result, pull_request


@pytest.fixture
def setup_monitor():
    orchestrator = MagicMock()
    orchestrator._pr_workflow.is_merged.return_value = False
    orchestrator.record_merged_pr_lesson.return_value = True
    return orchestrator, PullRequestMergeMonitor(orchestrator=orchestrator)


def register(monitor, task_data):
    issue, coder_result, pull_request = task_data
    monitor.register(
        issue=issue,
        coder_result=coder_result,
        pull_request=pull_request,
        summary="Corrected the typo.",
    )


def test_unmerged_pr_remains_pending(setup_monitor, task_data):
    orchestrator, monitor = setup_monitor
    register(monitor, task_data)

    assert monitor.poll_once() == {101: "pending"}
    assert monitor.pending_pull_numbers == frozenset({101})
    orchestrator.record_merged_pr_lesson.assert_not_called()


def test_merged_pr_records_lesson_and_removes_pending(setup_monitor, task_data):
    orchestrator, monitor = setup_monitor
    orchestrator._pr_workflow.is_merged.return_value = True
    register(monitor, task_data)

    assert monitor.poll_once() == {101: "recorded"}
    issue, coder_result, pull_request = task_data
    orchestrator.record_merged_pr_lesson.assert_called_once_with(
        issue, coder_result, pull_request, summary="Corrected the typo."
    )
    assert monitor.pending_pull_numbers == frozenset()


def test_merge_status_api_error_is_retried(setup_monitor, task_data):
    orchestrator, monitor = setup_monitor
    orchestrator._pr_workflow.is_merged.side_effect = RuntimeError("temporary API failure")
    register(monitor, task_data)

    assert monitor.poll_once() == {101: "retry"}
    assert monitor.pending_pull_numbers == frozenset({101})
    orchestrator.record_merged_pr_lesson.assert_not_called()


def test_lesson_recording_error_keeps_pr_pending(setup_monitor, task_data):
    orchestrator, monitor = setup_monitor
    orchestrator._pr_workflow.is_merged.return_value = True
    orchestrator.record_merged_pr_lesson.side_effect = RuntimeError("storage unavailable")
    register(monitor, task_data)

    assert monitor.poll_once() == {101: "retry"}
    assert monitor.pending_pull_numbers == frozenset({101})


def test_false_record_result_keeps_pr_pending(setup_monitor, task_data):
    orchestrator, monitor = setup_monitor
    orchestrator._pr_workflow.is_merged.return_value = True
    orchestrator.record_merged_pr_lesson.return_value = False
    register(monitor, task_data)

    assert monitor.poll_once() == {101: "not-recorded"}
    assert monitor.pending_pull_numbers == frozenset({101})


def test_register_rejects_mismatched_issue_number(setup_monitor, task_data):
    _, monitor = setup_monitor
    issue, coder_result, pull_request = task_data
    coder_result.issue_number = 7

    with pytest.raises(ValueError, match="does not match"):
        monitor.register(
            issue=issue,
            coder_result=coder_result,
            pull_request=pull_request,
            summary="Corrected the typo.",
        )


def test_register_rejects_blank_summary(setup_monitor, task_data):
    _, monitor = setup_monitor
    issue, coder_result, pull_request = task_data

    with pytest.raises(ValueError, match="summary"):
        monitor.register(
            issue=issue,
            coder_result=coder_result,
            pull_request=pull_request,
            summary="   ",
        )
