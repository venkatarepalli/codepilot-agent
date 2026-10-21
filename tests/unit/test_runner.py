"""Tests for the CodePilot command-line runner."""

import logging
from threading import Event
from types import SimpleNamespace
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
    orchestrator.merge_monitor = MagicMock(name="merge_monitor")
    polling = MagicMock()
    stop = Event()
    stop.set()

    monkeypatch.setattr(runner, "GitHubClient", lambda _settings: client)
    monkeypatch.setattr(runner, "create_orchestrator", lambda **kwargs: orchestrator)
    polling_factory = MagicMock(return_value=polling)
    monkeypatch.setattr(runner, "GitHubPollingService", polling_factory)

    assert runner.run(settings=settings, stop_event=stop) == 0
    polling_factory.assert_called_once_with(
        client, settings, merge_monitor=orchestrator.merge_monitor
    )
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
    monkeypatch.setattr(runner, "GitHubPollingService", lambda *_args, **_kwargs: polling)

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
    monkeypatch.setattr(runner, "GitHubPollingService", lambda *_args, **_kwargs: polling)

    runner.run(settings=settings, stop_event=Event())

    polling.release_issue.assert_called_once_with(8)


def test_interactive_runtime_routes_issue_snapshot_to_tui(monkeypatch):
    settings = make_settings()
    client = MagicMock()
    orchestrator = MagicMock()
    orchestrator.merge_monitor = None
    monkeypatch.setattr(runner, "GitHubClient", lambda _settings: client)
    monkeypatch.setattr(runner, "create_orchestrator", lambda **_kwargs: orchestrator)
    polling = MagicMock()
    polling_factory = MagicMock(return_value=polling)
    monkeypatch.setattr(runner, "GitHubPollingService", polling_factory)

    class TUIProbe:
        issues = None

        def update_issues(self, issues):
            self.issues = issues

        def call_from_thread(self, callback, *args, **kwargs):
            callback(*args, **kwargs)

    runtime = runner.CodePilotRuntime(settings, interactive=True)
    app = TUIProbe()
    runtime.attach_app(app)
    issue = GitHubIssue(number=42, title="Fix a typo")

    runtime._on_issues_snapshot((issue,), frozenset({42}))

    assert app.issues == [{"number": 42, "title": "Fix a typo", "status": "IN PROGRESS"}]
    polling_factory.assert_called_once_with(
        client,
        settings,
        merge_monitor=None,
        on_issues=runtime._on_issues_snapshot,
    )


def test_direct_task_runs_coder_and_does_not_publish_a_fake_issue(monkeypatch, tmp_path):
    settings = make_settings()
    client = MagicMock()
    orchestrator = MagicMock()
    orchestrator.merge_monitor = None
    orchestrator.run_coding_task.return_value = SimpleNamespace(
        status=runner.CoderStatus.SUCCEEDED,
        attempts=1,
        plan="Update calculator behavior.",
        test_result=SimpleNamespace(succeeded=True, return_code=0),
        diff_path=tmp_path / "proposed_diff.txt",
        error=None,
    )
    monkeypatch.setattr(runner, "GitHubClient", lambda _settings: client)
    create_orchestrator_factory = MagicMock(return_value=orchestrator)
    monkeypatch.setattr(
        runner,
        "create_orchestrator",
        create_orchestrator_factory,
    )
    monkeypatch.setattr(runner, "GitHubPollingService", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr(runner, "_clone_repository", lambda _settings, _destination: None)

    runtime = runner.CodePilotRuntime(settings)
    expected_repository = f"{settings.github_repository_owner}/{settings.github_repository_name}"

    assert create_orchestrator_factory.call_args.kwargs["repository"] == expected_repository
    result = runtime.process_direct_task("Fix the calculator's rounding behavior")

    orchestrator.run_coding_task.assert_called_once()
    fake_issue = orchestrator.run_coding_task.call_args.args[0]
    assert fake_issue.title == "Fix the calculator's rounding behavior"
    assert fake_issue.url == ""
    orchestrator.publish_coder_result.assert_not_called()
    assert str(tmp_path / "proposed_diff.txt") in result
    assert "No pull request was opened" in result


def test_skip_issue_discards_oldest_queued_issue_for_this_session(monkeypatch):
    settings = make_settings()
    client = MagicMock()
    orchestrator = MagicMock()
    orchestrator.merge_monitor = None
    polling = MagicMock()
    monkeypatch.setattr(runner, "GitHubClient", lambda _settings: client)
    monkeypatch.setattr(runner, "create_orchestrator", lambda **_kwargs: orchestrator)
    monkeypatch.setattr(runner, "GitHubPollingService", lambda *_args, **_kwargs: polling)

    class TUIProbe:
        issues = None

        def append_log(self, _message, *, source="CodePilot"):
            return None

        def update_issues(self, issues):
            self.issues = issues

    runtime = runner.CodePilotRuntime(settings, interactive=True)
    app = TUIProbe()
    runtime.attach_app(app)
    issue = GitHubIssue(number=42, title="Fix a typo")
    runtime._on_issues_snapshot((issue,), frozenset({issue.number}))
    runtime._enqueue_issue(issue)

    assert app.issues[0]["status"] == "QUEUED"

    runtime.skip_issue()

    assert not runtime._pending_issues
    assert issue.number in runtime._skipped_issue_numbers
    assert issue.number in runtime._attempted_issues
    assert app.issues[0]["status"] == "SKIPPED"
    polling.release_issue.assert_called_once_with(issue.number)


def test_tui_logging_handler_forwards_background_logs_to_ui() -> None:
    """Application logs are forwarded to Agent Logs instead of the terminal."""
    captured: list[tuple[str, str, str]] = []

    class RuntimeProbe:
        def _post_ui(self, method_name, message, *, source):
            captured.append((method_name, message, source))

    handler = runner._TUILogForwardingHandler(RuntimeProbe())
    record = logging.LogRecord(
        name="codepilot.agents.deepagents_coder_adapter",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Requesting code changes from the configured LLM.",
        args=(),
        exc_info=None,
    )

    handler.emit(record)

    assert len(captured) == 1
    method_name, message, source = captured[0]
    assert method_name == "append_log"
    assert "Requesting code changes" in message
    assert source == "deepagents_coder_adapter"


def test_issue_checkout_remains_available_until_pr_publication(monkeypatch):
    """Publishing finishes before the temporary source baseline is deleted."""
    settings = make_settings()
    client = MagicMock()
    orchestrator = MagicMock()
    orchestrator.merge_monitor = None
    checkout_paths = []
    baseline_available_during_publish = []

    result = SimpleNamespace(
        status=runner.CoderStatus.SUCCEEDED,
        attempts=1,
        plan="Update the calculation.",
        test_result=SimpleNamespace(succeeded=True, return_code=0),
        diff_path=None,
        error=None,
    )

    def clone_repository(_settings, destination):
        destination.mkdir(parents=True)
        (destination / "baseline-marker.txt").write_text("baseline", encoding="utf-8")

    def run_coding_task(_issue, repository_path, *, lifecycle_callback=None):
        checkout = runner.Path(repository_path)
        assert checkout.is_dir()
        checkout_paths.append(checkout)
        return result

    def publish_coder_result(*_args, **_kwargs):
        checkout = checkout_paths[-1]
        marker = checkout / "baseline-marker.txt"
        baseline_available_during_publish.append(
            checkout.is_dir()
            and marker.is_file()
            and marker.read_text(encoding="utf-8") == "baseline"
        )
        return SimpleNamespace(
            pull_url="https://github.com/example/repository/pull/7",
            pull_number=7,
        )

    orchestrator.run_coding_task.side_effect = run_coding_task
    orchestrator.publish_coder_result.side_effect = publish_coder_result
    polling = MagicMock()
    monkeypatch.setattr(runner, "GitHubClient", lambda _settings: client)
    monkeypatch.setattr(runner, "create_orchestrator", lambda **_kwargs: orchestrator)
    monkeypatch.setattr(runner, "GitHubPollingService", lambda *_args, **_kwargs: polling)
    monkeypatch.setattr(runner, "_clone_repository", clone_repository)

    runtime = runner.CodePilotRuntime(settings)
    issue = SimpleNamespace(number=42, title="Fix calculator", reporter=None, url="")
    runtime.process_issue(issue)

    orchestrator.publish_coder_result.assert_called_once()
    assert baseline_available_during_publish == [True]
    commit_message = orchestrator.publish_coder_result.call_args.kwargs["commit_message"]

    assert commit_message.startswith("fix(#42): Fix calculator")
    assert "- Change: Fix calculator" in commit_message
    assert "- Why: Address the reported issue." in commit_message
    assert "- Closes #42" in commit_message
    assert len(checkout_paths) == 1
    assert not checkout_paths[0].exists(), (
        "temporary checkout should be cleaned up after publication"
    )
    polling.release_issue.assert_called_once_with(issue.number)


def test_unexpected_issue_error_is_reported_as_failed(monkeypatch, caplog):
    """Unexpected processing errors mark the issue failed and release its claim."""
    caplog.set_level("ERROR", logger="codepilot.runner")

    settings = make_settings()
    client = MagicMock()
    orchestrator = MagicMock()
    orchestrator.merge_monitor = None
    polling = MagicMock()

    monkeypatch.setattr(runner, "GitHubClient", lambda _settings: client)
    monkeypatch.setattr(
        runner,
        "create_orchestrator",
        lambda **_kwargs: orchestrator,
    )
    monkeypatch.setattr(
        runner,
        "GitHubPollingService",
        lambda *_args, **_kwargs: polling,
    )
    monkeypatch.setattr(
        runner,
        "_clone_repository",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("clone failed")),
    )

    runtime = runner.CodePilotRuntime(settings)
    issue = SimpleNamespace(
        number=43,
        title="Fix calculator",
        reporter=None,
        url="https://github.com/example/repository/issues/43",
    )

    runtime.process_issue(issue)

    assert runtime._issue_statuses[issue.number] == "FAILED"
    assert "Coding workflow failed for issue #43" in caplog.text
    assert "clone failed" in caplog.text
    polling.release_issue.assert_called_once_with(issue.number)
    orchestrator.run_coding_task.assert_not_called()
    assert runtime._issue_statuses[issue.number] == runner.TaskStatus.FAILED.value


def test_rejected_publication_is_reported_as_rejected_not_failed(monkeypatch, caplog):
    """A deliberate human rejection is not logged as an unexpected failure."""
    caplog.set_level("INFO", logger="codepilot.runner")

    settings = make_settings()
    client = MagicMock()
    orchestrator = MagicMock()
    orchestrator.merge_monitor = None
    orchestrator.run_coding_task.return_value = SimpleNamespace(
        status=runner.CoderStatus.SUCCEEDED,
        attempts=1,
        plan="Update the calculation.",
        test_result=SimpleNamespace(succeeded=True, return_code=0),
        diff_path=None,
        error=None,
    )
    polling = MagicMock()

    monkeypatch.setattr(runner, "GitHubClient", lambda _settings: client)
    monkeypatch.setattr(runner, "create_orchestrator", lambda **_kwargs: orchestrator)
    monkeypatch.setattr(runner, "GitHubPollingService", lambda *_args, **_kwargs: polling)

    def clone_repository(_settings, destination):
        destination.mkdir(parents=True)

    monkeypatch.setattr(runner, "_clone_repository", clone_repository)
    runtime = runner.CodePilotRuntime(settings)

    def reject_publication(*_args, **_kwargs):
        runtime.pr_workflow._require_approval(
            lambda *_approval_args: False,
            "publish_branch",
            {"issue_number": 42},
        )

    orchestrator.publish_coder_result.side_effect = reject_publication
    issue = SimpleNamespace(
        number=42,
        title="Fix calculator",
        reporter=None,
        url="https://github.com/example/repository/issues/42",
    )

    runtime.process_issue(issue)

    assert runtime._issue_statuses[issue.number] == "REJECTED"
    assert "Publication rejected for issue #42" in caplog.text
    assert "Coding workflow failed for issue #42" not in caplog.text
    assert not any(record.exc_info for record in caplog.records)
    polling.release_issue.assert_called_once_with(issue.number)
