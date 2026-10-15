"""Regression tests for forwarding LangGraph invocation configuration."""

from unittest.mock import MagicMock

from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.github.models import GitHubIssue


def _issue() -> GitHubIssue:
    return GitHubIssue(
        number=42,
        title="Fix login crash",
        body="The application crashes during login.",
        url="https://github.com/example/repo/issues/42",
        labels=("bug",),
        reporter="reporter",
    )


def test_invoke_forwards_langgraph_config() -> None:
    agent = MagicMock()
    agent.invoke.return_value = {"messages": []}
    orchestrator = CodePilotOrchestrator(
        model="test-model",
        agent_factory=MagicMock(return_value=agent),
    )
    config = {"recursion_limit": 25}

    result = orchestrator.invoke(_issue(), config=config)

    assert result == {"messages": []}
    agent.invoke.assert_called_once()
    assert agent.invoke.call_args.kwargs["config"] == config


def test_invoke_without_config_preserves_existing_call_shape() -> None:
    agent = MagicMock()
    agent.invoke.return_value = {"messages": []}
    orchestrator = CodePilotOrchestrator(
        model="test-model",
        agent_factory=MagicMock(return_value=agent),
    )

    orchestrator.invoke(_issue())

    agent.invoke.assert_called_once_with(
        {"messages": [{"role": "user", "content": orchestrator.build_task_prompt(_issue())}]}
    )
