"""Tests for the orchestrator's DeepAgents-to-CoderAgent integration."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from codepilot.agents.coder_models import CoderResult, CoderStatus
from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.github.models import GitHubIssue
from codepilot.sandbox.execution_models import CommandExecutionResult, CommandStatus


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
    root_agent = MagicMock()
    root_agent.invoke.return_value = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "todo-call",
                        "name": "write_todos",
                        "args": {"todos": ["Explore repository", "Plan implementation"]},
                    }
                ],
                "content": "",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "repo-task-call",
                        "name": "task",
                        "args": {
                            "subagent_type": "repo_explorer",
                            "description": "Identify relevant repository files.",
                        },
                    }
                ],
                "content": "",
            },
            {
                "role": "tool",
                "tool_call_id": "repo-task-call",
                "content": "Relevant file: src/calculator.py",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "planner-task-call",
                        "name": "task",
                        "args": {
                            "subagent_type": "implementation_planner",
                            "description": "Propose a focused implementation and tests.",
                        },
                    }
                ],
                "content": "",
            },
            {
                "role": "tool",
                "tool_call_id": "planner-task-call",
                "content": "Inspect the calculation and run the configured tests.",
            },
            {
                "role": "assistant",
                "content": (
                    "Handoff: Relevant file is src/calculator.py. "
                    "Proposed approach: inspect the calculation and make a focused fix. "
                    "Checks: run pytest."
                ),
            },
        ]
    }

    coding_agent = MagicMock()
    agent_factory = MagicMock(side_effect=[root_agent, coding_agent])
    approval_callback = Mock(return_value=True)
    orchestrator = CodePilotOrchestrator(
        model="test-model",
        repository="owner/repo",
        agent_factory=agent_factory,
        coder_workspace=workspace,
        test_command=("pytest", "-q"),
        approval_callback=approval_callback,
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
    assert kwargs["approval_callback"] is approval_callback
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


def test_publish_coder_result_excludes_internal_workspace_artifacts(
    issue: GitHubIssue, tmp_path: Path
) -> None:
    """Publish only repository changes, not CodePilot-generated working files."""
    coder_workspace = Mock()
    coder_workspace.get_changes.return_value = {
        "added": [
            "working/proposed_diff.txt",
            "working/test-output.txt",
            ".pytest_cache/v/cache/nodeids",
        ],
        "modified": ["src/calculator.py"],
        "deleted": ["working/old-diagnostic.log"],
    }
    calculator_content = (
        "def calculate_total(price: float, quantity: int) -> float:\n"
        '    """Calculate the total price for the requested quantity."""\n'
        "    return price * quantity\n"
    )
    coder_workspace.read_file.return_value = calculator_content
    pr_workflow = Mock()
    orchestrator = CodePilotOrchestrator(
        model="test-model",
        agent_factory=MagicMock(return_value=MagicMock()),
        pr_workflow=pr_workflow,
        coder_workspace=coder_workspace,
        test_command=("pytest", "-q"),
    )
    result = CoderResult(
        status=CoderStatus.SUCCEEDED,
        issue_number=issue.number,
        workspace=Mock(),
        plan="Correct the multiplication operator.",
        attempts=1,
        test_result=CommandExecutionResult(
            status=CommandStatus.SUCCEEDED,
            command=("pytest", "-q"),
            return_code=0,
        ),
        diff_path=tmp_path / "working" / "proposed_diff.txt",
    )

    orchestrator.publish_coder_result(
        issue,
        result,
        commit_message="fix: multiply price by quantity",
        summary="Correct total calculation.",
    )

    coder_workspace.read_file.assert_called_once_with(result.workspace, "src/calculator.py")
    assert pr_workflow.publish.call_args.kwargs["changes"] == {
        "src/calculator.py": calculator_content
    }


def test_publish_coder_result_rejects_artifact_only_change_set(
    issue: GitHubIssue, tmp_path: Path
) -> None:
    """Refuse to publish when a workspace only differs by internal artifacts."""
    coder_workspace = Mock()
    coder_workspace.get_changes.return_value = {
        "added": ["working/proposed_diff.txt", "working/test-output.txt"],
        "modified": [],
        "deleted": [],
    }
    pr_workflow = Mock()
    orchestrator = CodePilotOrchestrator(
        model="test-model",
        agent_factory=MagicMock(return_value=MagicMock()),
        pr_workflow=pr_workflow,
        coder_workspace=coder_workspace,
        test_command=("pytest", "-q"),
    )
    result = CoderResult(
        status=CoderStatus.SUCCEEDED,
        issue_number=issue.number,
        workspace=Mock(),
        plan="No repository source change.",
        attempts=1,
        test_result=CommandExecutionResult(
            status=CommandStatus.SUCCEEDED,
            command=("pytest", "-q"),
            return_code=0,
        ),
        diff_path=tmp_path / "working" / "proposed_diff.txt",
    )

    with pytest.raises(ValueError, match="No source-file changes were detected"):
        orchestrator.publish_coder_result(
            issue,
            result,
            commit_message="fix: test artifact filtering",
            summary="No source changes.",
        )

    pr_workflow.publish.assert_not_called()
