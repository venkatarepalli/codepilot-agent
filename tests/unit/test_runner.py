"""Tests for the CodePilot command-line runner."""

from threading import Event
from unittest.mock import MagicMock

from codepilot import runner
from codepilot.config.settings import Settings
from codepilot.github.models import GitHubIssue


def make_settings() -> Settings:
    return Settings(
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        openai_api_key="test-key",
        anthropic_api_key=None,
        google_api_key=None,
        github_token="test-token",
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


def test_run_validates_configuration_before_connecting(monkeypatch):
    settings = make_settings()
    settings = Settings(
        **{
            **{field: getattr(settings, field) for field in settings.__dataclass_fields__},
            "github_token": None,
        }
    )
    client_factory = MagicMock()
    monkeypatch.setattr(runner, "GitHubClient", client_factory)

    from codepilot.config.settings import ConfigurationError

    try:
        runner.run(settings=settings, stop_event=Event())
    except ConfigurationError:
        pass
    else:
        raise AssertionError("Expected configuration validation to fail")

    client_factory.assert_not_called()


def test_run_connects_components_and_stops_cleanly(monkeypatch, capsys):
    settings = make_settings()
    client = MagicMock()
    toolkit = MagicMock()
    toolkit.get_tools.return_value = ["github-tool"]
    client.toolkit = toolkit
    orchestrator = MagicMock()
    orchestrator.invoke.return_value = {"messages": []}
    polling = MagicMock()
    stop = Event()
    stop.set()

    monkeypatch.setattr(runner, "GitHubClient", lambda _settings: client)
    monkeypatch.setattr(runner, "create_orchestrator", lambda **kwargs: orchestrator)
    monkeypatch.setattr(runner, "GitHubPollingService", lambda *_args: polling)

    assert runner.run(settings=settings, stop_event=stop) == 0
    orchestrator.invoke.assert_not_called()
    polling.run_forever.assert_called_once()
    assert "CodePilot coding loop started" in capsys.readouterr().out


def test_analysis_callback_releases_issue_after_success(monkeypatch):
    settings = make_settings()
    client = MagicMock()
    client.toolkit.get_tools.return_value = []
    orchestrator = MagicMock()
    orchestrator.invoke.return_value = {"messages": []}
    polling = MagicMock()
    monkeypatch.setattr(runner, "GitHubClient", lambda _settings: client)
    monkeypatch.setattr(runner, "create_orchestrator", lambda **kwargs: orchestrator)
    monkeypatch.setattr(runner, "_clone_repository", lambda *_args: None)

    def capture_callback(callback, _stop_event):
        callback(GitHubIssue(number=7, title="Investigate issue"))

    polling.run_forever.side_effect = capture_callback
    monkeypatch.setattr(runner, "GitHubPollingService", lambda *_args: polling)

    runner.run(settings=settings, stop_event=Event())

    orchestrator.run_coding_task.assert_called_once()
    polling.release_issue.assert_called_once_with(7)


def test_analysis_callback_releases_issue_when_agent_fails(monkeypatch):
    settings = make_settings()
    client = MagicMock()
    client.toolkit.get_tools.return_value = []
    orchestrator = MagicMock()
    orchestrator.invoke.side_effect = RuntimeError("agent failed")
    polling = MagicMock()
    monkeypatch.setattr(runner, "GitHubClient", lambda _settings: client)
    monkeypatch.setattr(runner, "create_orchestrator", lambda **kwargs: orchestrator)
    monkeypatch.setattr(runner, "_clone_repository", lambda *_args: None)

    def capture_callback(callback, _stop_event):
        try:
            callback(GitHubIssue(number=8, title="Investigate issue"))
        except RuntimeError:
            pass

    polling.run_forever.side_effect = capture_callback
    monkeypatch.setattr(runner, "GitHubPollingService", lambda *_args: polling)

    runner.run(settings=settings, stop_event=Event())

    polling.release_issue.assert_called_once_with(8)
