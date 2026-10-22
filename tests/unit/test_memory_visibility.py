"""Memory activity is logged at INFO so it is visible in the TUI Agent Logs."""

import logging
from unittest.mock import MagicMock

from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.github.merge_monitor import PullRequestMergeMonitor
from codepilot.github.models import GitHubIssue
from codepilot.github.pr_workflow import PullRequestResult


def test_recorded_lesson_is_logged_at_info(caplog) -> None:
    orchestrator = MagicMock()
    orchestrator._pr_workflow.is_merged.return_value = True
    orchestrator.record_merged_pr_lesson.return_value = True
    monitor = PullRequestMergeMonitor(orchestrator=orchestrator)
    coder_result = MagicMock()
    coder_result.issue_number = 42
    monitor.register(
        issue=GitHubIssue(number=42, title="Fix a typo", url="u"),
        coder_result=coder_result,
        pull_request=PullRequestResult(
            branch="codepilot/issue-42-fix-a-typo",
            pull_number=101,
            pull_url="u/101",
            title="[CodePilot] Fix a typo",
            base_branch="main",
        ),
        summary="Corrected the typo.",
    )

    with caplog.at_level(logging.INFO, logger="codepilot"):
        assert monitor.poll_once() == {101: "recorded"}

    assert "recorded lesson from merged PR #101 (issue #42)" in caplog.text


def test_memory_injection_is_logged_at_info(caplog) -> None:
    episodic = MagicMock()
    episodic.get_recent_tasks.return_value = [{"issue_number": 1}]
    semantic = MagicMock()
    semantic.retrieve_lessons.return_value = [{"lesson_id": "a"}, {"lesson_id": "b"}]
    orchestrator = CodePilotOrchestrator(
        model=object(),
        repository="example/repo",
        agent_factory=lambda **kwargs: MagicMock(),
        episodic_memory=episodic,
        semantic_memory=semantic,
    )

    with caplog.at_level(logging.INFO, logger="codepilot"):
        orchestrator.prepare_task(GitHubIssue(number=9, title="Fix bug", body="x"))

    assert "Memory for issue #9: 2 past lesson(s)" in caplog.text
    assert "1 recent task(s)" in caplog.text
