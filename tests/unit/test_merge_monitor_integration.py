"""Integration tests for publishing and polling merge-monitor wiring."""

from unittest.mock import MagicMock

from codepilot.config.bootstrap import create_orchestrator
from codepilot.config.settings import Settings
from codepilot.github.merge_monitor import PullRequestMergeMonitor
from codepilot.github.models import GitHubIssue
from codepilot.github.polling import GitHubPollingService


def _settings() -> Settings:
    return Settings(
        llm_provider="",
        llm_model="",
        openai_api_key=None,
        anthropic_api_key=None,
        google_api_key=None,
        github_token=None,
        github_repository_owner="example-owner",
        github_repository_name="example-repo",
        github_default_branch="main",
        poll_interval_minutes=5,
        complexity_threshold=7,
        repo_map_token_budget=4000,
        repo_map_top_k=10,
        max_test_retries=3,
        state_dir=".codepilot",
        chroma_dir=".codepilot/chroma",
    )


def _issue() -> GitHubIssue:
    return GitHubIssue(number=42, title="Fix a typo")


def test_bootstrap_attaches_merge_monitor_when_workflow_is_supplied(monkeypatch, tmp_path):
    workflow = MagicMock()
    monkeypatch.setattr("codepilot.config.bootstrap.create_chat_model", lambda: object())
    # The orchestrator's agent factory expects a real chat model. Keep this
    # bootstrap wiring test isolated from provider-specific model interfaces.
    monkeypatch.setattr(
        "codepilot.agents.orchestrator.create_deep_agent",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "codepilot.config.bootstrap.SemanticMemory.from_persistent_directory",
        lambda **kwargs: object(),
    )

    orchestrator = create_orchestrator(
        workspace_root=tmp_path / "workspaces",
        pr_workflow=workflow,
    )

    assert isinstance(orchestrator.merge_monitor, PullRequestMergeMonitor)


def test_polling_checks_merge_monitor_without_changing_issue_polling():
    client = MagicMock()
    issue = _issue()
    client.get_open_issues.return_value = [issue]
    monitor = MagicMock()
    service = GitHubPollingService(client, _settings(), merge_monitor=monitor)

    assert service.poll_once() == [issue]
    monitor.poll_once.assert_called_once_with()
    client.get_open_issues.assert_called_once_with()
