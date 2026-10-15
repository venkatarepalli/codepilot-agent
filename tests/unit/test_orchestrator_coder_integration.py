"""Tests for the orchestrator's DeepAgents-to-CoderAgent integration."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from codepilot.agents.coder_models import CoderResult, CoderStatus
from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.github.models import GitHubIssue


@pytest.fixture
def issue() -> GitHubIssue:
    return GitHubIssue(number=42, title="Fix calculation", body="Return 42.")


def test_run_coding_task_requires_workspace_and_test_command(issue: GitHubIssue) -> None:
    orchestrator = CodePilotOrchestrator(
        model="test-model",
        agent_factory=MagicMock(return_value=MagicMock()),
    )
    with pytest.raises(RuntimeError, match="coder_workspace and test_command"):
        orchestrator.run_coding_task(issue, ".")


def test_run_coding_task_connects_deepagents_adapter_to_coder_agent(
    issue: GitHubIssue, tmp_path: Path
) -> None:
    workspace = Mock()
    agent_factory = MagicMock(return_value=MagicMock())
    orchestrator = CodePilotOrchestrator(
        model="test-model",
        agent_factory=agent_factory,
        coder_workspace=workspace,
        test_command=("pytest", "-q"),
    )
    expected = CoderResult(
        status=CoderStatus.FAILED,
        issue_number=42,
        workspace=Mock(),
        plan="",
        attempts=0,
        error="test sentinel",
    )
    with patch("codepilot.agents.orchestrator.CoderAgent") as coder_type:
        coder_type.return_value.run.return_value = expected
        result = orchestrator.run_coding_task(issue, str(tmp_path))

    assert result is expected
    coder_type.assert_called_once()
    kwargs = coder_type.call_args.kwargs
    assert kwargs["planner"].__self__.__class__.__name__ == "DeepAgentsCoderAdapter"
    assert kwargs["implementer"].__self__ is kwargs["planner"].__self__
    assert kwargs["test_command"] == ("pytest", "-q")
    coder_type.return_value.run.assert_called_once_with(issue, str(tmp_path))
    assert agent_factory.call_count == 2
    coding_agent_kwargs = agent_factory.call_args.kwargs
    assert coding_agent_kwargs["tools"] == []
    assert coding_agent_kwargs["name"] == "codepilot_coder"


def test_workspace_and_test_command_must_be_configured_together() -> None:
    with pytest.raises(ValueError, match="provided together"):
        CodePilotOrchestrator(
            model="test-model",
            agent_factory=MagicMock(return_value=MagicMock()),
            test_command=("pytest", "-q"),
        )
