"""Component 2: Repo Map token budget and top-K come from configuration."""

from unittest.mock import MagicMock

import pytest

from codepilot.config import bootstrap


def test_bootstrap_uses_configured_budget_and_top_k(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    constructor = MagicMock(return_value=object())
    monkeypatch.setattr(bootstrap, "create_chat_model", lambda: object())
    monkeypatch.setattr(bootstrap, "CodePilotOrchestrator", constructor)

    bootstrap.create_orchestrator(
        workspace_root=tmp_path / "ws", repo_map_token_budget=1234, repo_map_top_k=4
    )

    assembler = constructor.call_args.kwargs["context_assembler"]
    assert assembler.token_budget == 1234
    assert assembler.keyword_top_k == 4
    assert assembler.semantic_top_k == 4


def test_bootstrap_rejects_non_positive_values(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bootstrap, "create_chat_model", lambda: object())
    with pytest.raises(ValueError):
        bootstrap.create_orchestrator(repo_map_token_budget=0)


def test_runner_forwards_settings_to_bootstrap(monkeypatch) -> None:
    from codepilot import runner
    from codepilot.config.settings import Settings

    captured = {}

    def fake_create_orchestrator(**kwargs):
        captured.update(kwargs)
        orchestrator = MagicMock()
        orchestrator.merge_monitor = None
        return orchestrator

    monkeypatch.setattr(runner, "GitHubClient", lambda _settings: MagicMock())
    monkeypatch.setattr(runner, "create_orchestrator", fake_create_orchestrator)
    monkeypatch.setattr(runner, "GitHubPollingService", MagicMock())
    settings = Settings(
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        openai_api_key="k",
        anthropic_api_key=None,
        google_api_key=None,
        github_token="t",
        github_repository_owner="o",
        github_repository_name="r",
        github_default_branch="main",
        poll_interval_minutes=5,
        complexity_threshold=7,
        repo_map_token_budget=2500,
        repo_map_top_k=6,
        max_test_retries=3,
        state_dir=".codepilot",
        chroma_dir=".codepilot/chroma",
    )

    runner.CodePilotRuntime(settings)

    assert captured["repo_map_token_budget"] == 2500
    assert captured["repo_map_top_k"] == 6
