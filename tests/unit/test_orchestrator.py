"""Unit tests for the CodePilot orchestrator."""

from unittest.mock import MagicMock

import pytest

from codepilot.agents.classifier import IssueCategory
from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.context.context_assembler import ContextAssembler, ContextPackage
from codepilot.context.repo_map import FileSummary, RepoMap
from codepilot.github.models import GitHubIssue
from codepilot.graph.lifecycle import TaskStatus


@pytest.fixture
def issue() -> GitHubIssue:
    """Return a sample bug-fix issue."""
    return GitHubIssue(
        number=42,
        title="Fix login crash",
        body="The application crashes during login.",
        url="https://github.com/example/repo/issues/42",
        labels=("bug",),
        reporter="reporter",
    )


@pytest.fixture
def repo_map() -> RepoMap:
    """Return a small repository map for context integration tests."""
    return RepoMap(
        root="sample-repository",
        directory_tree="src/\\n  login.py",
        files=(
            FileSummary(
                path="src/login.py",
                language="Python",
                summary="Login flow validates credentials and handles exceptions.",
            ),
        ),
    )


def make_orchestrator(**kwargs) -> CodePilotOrchestrator:
    return CodePilotOrchestrator(
        model="test-model",
        agent_factory=MagicMock(return_value=MagicMock()),
        **kwargs,
    )


def test_initializes_deepagents_with_expected_configuration() -> None:
    factory = MagicMock(return_value=MagicMock())
    CodePilotOrchestrator(model="test-model", tools=[], agent_factory=factory)

    factory.assert_called_once()
    kwargs = factory.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["name"] == "codepilot_orchestrator"
    assert "strict safety rules" in kwargs["system_prompt"]


def test_prepare_task_classifies_issue(issue: GitHubIssue) -> None:
    task = make_orchestrator().prepare_task(issue)
    assert task["category"] == IssueCategory.BUG_FIX
    assert task["status"] == TaskStatus.TRIAGED
    assert task["issue"] == issue
    assert task["context"] is None


def test_task_prompt_contains_issue_details(issue: GitHubIssue) -> None:
    prompt = make_orchestrator().build_task_prompt(issue)
    assert "GitHub issue #42" in prompt
    assert "Fix login crash" in prompt
    assert issue.url in prompt
    assert "bug_fix" in prompt
    assert "regression tests" in prompt


def test_invoke_submits_prompt_to_agent(issue: GitHubIssue) -> None:
    agent = MagicMock()
    agent.invoke.return_value = {"messages": []}
    orchestrator = CodePilotOrchestrator(
        model="test-model",
        agent_factory=MagicMock(return_value=agent),
    )

    result = orchestrator.invoke(issue)

    assert result == {"messages": []}
    agent.invoke.assert_called_once()
    payload = agent.invoke.call_args.args[0]
    assert payload["messages"][0]["role"] == "user"
    assert "GitHub issue #42" in payload["messages"][0]["content"]


def test_agent_property_returns_compiled_agent() -> None:
    agent = MagicMock()
    orchestrator = CodePilotOrchestrator(
        model="test-model",
        agent_factory=MagicMock(return_value=agent),
    )
    assert orchestrator.agent is agent


def test_context_assembler_is_included_in_task_prompt(
    issue: GitHubIssue, repo_map: RepoMap
) -> None:
    assembler = ContextAssembler(token_budget=1000)
    provider = MagicMock(return_value=repo_map)
    orchestrator = make_orchestrator(
        context_assembler=assembler,
        repo_map_provider=provider,
    )

    task = orchestrator.prepare_task(issue)

    provider.assert_called_once_with()
    assert isinstance(task["context"], ContextPackage)
    assert "## Repository map" in task["prompt"]
    assert "src/login.py" in task["prompt"]
    assert "Login flow validates credentials" in task["prompt"]
    assert "read source files on demand" in task["prompt"]


def test_context_is_built_using_issue_title_and_description(
    issue: GitHubIssue, repo_map: RepoMap
) -> None:
    assembler = MagicMock()
    assembler.assemble.return_value = ContextPackage(
        repo_map_text="Repository: sample-repository",
        files=(),
        estimated_tokens=10,
        token_budget=1000,
        truncated=False,
    )
    orchestrator = make_orchestrator(
        context_assembler=assembler,
        repo_map_provider=lambda: repo_map,
    )

    orchestrator.prepare_task(issue)

    assembler.assemble.assert_called_once_with(repo_map, issue.title, issue.body)


def test_context_configuration_must_be_provided_as_a_pair() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        make_orchestrator(context_assembler=ContextAssembler())


def test_context_truncation_is_disclosed_in_prompt(issue: GitHubIssue, repo_map: RepoMap) -> None:
    assembler = MagicMock()
    assembler.assemble.return_value = ContextPackage(
        repo_map_text="Repository: sample-repository",
        files=(),
        estimated_tokens=20,
        token_budget=20,
        truncated=True,
    )
    orchestrator = make_orchestrator(
        context_assembler=assembler,
        repo_map_provider=lambda: repo_map,
    )

    prompt = orchestrator.build_task_prompt(issue, context_package=assembler.assemble.return_value)

    assert "context was truncated" in prompt
    assert "on demand" in prompt


def test_invoke_includes_assembled_context(issue: GitHubIssue, repo_map: RepoMap) -> None:
    agent = MagicMock()
    agent.invoke.return_value = {"messages": []}
    orchestrator = CodePilotOrchestrator(
        model="test-model",
        agent_factory=MagicMock(return_value=agent),
        context_assembler=ContextAssembler(token_budget=1000),
        repo_map_provider=lambda: repo_map,
    )

    orchestrator.invoke(issue)

    prompt = agent.invoke.call_args.args[0]["messages"][0]["content"]
    assert "src/login.py" in prompt
