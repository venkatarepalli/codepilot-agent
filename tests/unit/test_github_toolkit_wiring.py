"""LangChain GitHubToolkit is wired into the Orchestrator with role scoping."""

from unittest.mock import MagicMock

from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.config.settings import Settings
from codepilot.github.client import (
    BLOCKED_TOOLKIT_MODES,
    ORCHESTRATOR_READ_ONLY_MODES,
    PR_AGENT_WRITE_MODES,
    GitHubClient,
)


def _client() -> GitHubClient:
    wrapper = GitHubAPIWrapper.model_construct(
        github=MagicMock(),
        github_repo_instance=MagicMock(),
        github_repository="owner/repo",
        active_branch="main",
        github_base_branch="main",
    )
    return GitHubClient(MagicMock(spec=Settings), api_wrapper=wrapper)


def test_client_builds_real_langchain_github_toolkit() -> None:
    client = _client()
    assert isinstance(client.toolkit, GitHubToolkit)
    assert len(client.tools) > 0


def test_every_toolkit_mode_is_explicitly_classified() -> None:
    # If langchain adds a new GitHub tool, this fails instead of silently
    # leaving it unscoped.
    modes = {tool.mode for tool in _client().tools}
    classified = ORCHESTRATOR_READ_ONLY_MODES | PR_AGENT_WRITE_MODES | BLOCKED_TOOLKIT_MODES
    assert modes <= classified, modes - classified
    assert not ORCHESTRATOR_READ_ONLY_MODES & PR_AGENT_WRITE_MODES


def test_orchestrator_tools_are_read_only_and_include_list_issues() -> None:
    modes = {tool.mode for tool in _client().orchestrator_tools}
    assert "get_issues" in modes
    assert modes <= ORCHESTRATOR_READ_ONLY_MODES
    for dangerous in (
        "delete_file",
        "create_file",
        "update_file",
        "create_pull_request",
        "create_branch",
        "set_active_branch",
    ):
        assert dangerous not in modes


def test_pr_agent_tools_contain_branch_and_pr_creation() -> None:
    modes = {tool.mode for tool in _client().pr_agent_tools}
    assert {"create_branch", "create_pull_request"} <= modes
    assert "delete_file" not in modes


def test_toolkit_tools_are_bound_to_the_real_deep_agent() -> None:
    client = _client()
    orchestrator = CodePilotOrchestrator(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="ok")])),
        tools=client.orchestrator_tools,
    )
    bound = set(orchestrator.agent.nodes["tools"].bound.tools_by_name)
    expected = {tool.name for tool in client.orchestrator_tools}
    assert expected <= bound
    assert "Get_Issues" in bound
    assert "Delete_File" not in bound
    assert "Create_Pull_Request" not in bound


def test_runner_passes_scoped_toolkit_tools_to_orchestrator(monkeypatch) -> None:
    from threading import Event

    from codepilot import runner

    client = MagicMock()
    client.orchestrator_tools = ["read-only-tool"]
    client.tools = ["read-only-tool", "delete-tool"]
    captured = {}

    def fake_create_orchestrator(**kwargs):
        captured.update(kwargs)
        orchestrator = MagicMock()
        orchestrator.merge_monitor = None
        return orchestrator

    monkeypatch.setattr(runner, "GitHubClient", lambda _settings: client)
    monkeypatch.setattr(runner, "create_orchestrator", fake_create_orchestrator)
    monkeypatch.setattr(runner, "GitHubPollingService", MagicMock())
    stop = Event()
    stop.set()

    runner.run(
        settings=Settings(
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
        ),
        stop_event=stop,
    )

    assert captured["tools"] == ["read-only-tool"]
