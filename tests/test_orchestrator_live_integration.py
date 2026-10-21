"""Acceptance tests for the live DeepAgents orchestrator integration."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from codepilot.agents.coder_models import CoderResult, CoderStatus
from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.github.models import GitHubIssue


@pytest.fixture
def issue() -> GitHubIssue:
    return GitHubIssue(
        number=42,
        title="Fix calculation",
        body="Return price multiplied by quantity.",
        url="https://github.com/example/repo/issues/42",
        reporter="reporter",
    )


def _delegated_response() -> dict[str, object]:
    """Model an observed root-agent run that uses todos and both specialist agents."""
    return {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "todo-call",
                        "name": "write_todos",
                        "args": {"todos": ["Explore", "Plan"]},
                    }
                ],
                "content": "",
            },
            {"role": "tool", "name": "write_todos", "content": "Todos updated"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "repo-task-call",
                        "name": "task",
                        "args": {
                            "subagent_type": "repo_explorer",
                            "description": (
                                "Identify relevant paths from the provided repository summaries."
                            ),
                        },
                    }
                ],
                "content": "",
            },
            {
                "role": "tool",
                "tool_call_id": "repo-task-call",
                "content": "Relevant path: src/calculator.py",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "planner-task-call",
                        "name": "task",
                        "args": {
                            "subagent_type": "implementation_planner",
                            "description": (
                                "Propose a small, testable change using the explorer findings."
                            ),
                        },
                    }
                ],
                "content": "",
            },
            {
                "role": "tool",
                "tool_call_id": "planner-task-call",
                "content": "Plan: multiply price by quantity and run focused tests.",
            },
            {
                "role": "assistant",
                "content": (
                    "Handoff: inspect src/calculator.py, multiply price by quantity, "
                    "and verify with focused tests."
                ),
            },
        ]
    }


def _make_orchestrator(
    *,
    root_agent: MagicMock,
    coding_agent: MagicMock,
    workspace: Mock,
    agent_factory: MagicMock,
) -> CodePilotOrchestrator:
    agent_factory.side_effect = [root_agent, coding_agent]
    return CodePilotOrchestrator(
        model="test-model",
        agent_factory=agent_factory,
        coder_workspace=workspace,
        test_command=("pytest", "-q"),
    )


def test_root_agent_is_configured_with_named_specialists() -> None:
    root_agent = MagicMock()
    factory = MagicMock(return_value=root_agent)

    CodePilotOrchestrator(model="test-model", agent_factory=factory)

    root_kwargs = factory.call_args.kwargs
    specialists = {item["name"] for item in root_kwargs.get("subagents", [])}
    assert {"repo_explorer", "implementation_planner"} <= specialists
    assert "write_todos" in root_kwargs["system_prompt"]
    assert "task" in root_kwargs["system_prompt"]


def test_live_coding_path_recovers_from_incomplete_root_delegation(
    issue: GitHubIssue,
    tmp_path: Path,
) -> None:
    root_agent = MagicMock()
    root_agent.invoke.side_effect = [
        {
            "messages": [
                {"role": "user", "content": "Initial task"},
                {
                    "role": "assistant",
                    "content": "Repository exploration is complete.",
                },
            ]
        },
        _delegated_response(),
    ]

    coding_agent = MagicMock()
    workspace = Mock()
    factory = MagicMock()

    orchestrator = _make_orchestrator(
        root_agent=root_agent,
        coding_agent=coding_agent,
        workspace=workspace,
        agent_factory=factory,
    )

    orchestrator.prepare_task = Mock(  # type: ignore[method-assign]
        return_value={
            "prompt": "Prepared issue prompt",
            "category": "bug_fix",
        }
    )

    expected = CoderResult(
        status=CoderStatus.FAILED,
        issue_number=issue.number,
        workspace=Mock(),
        plan="",
        attempts=0,
        error="test sentinel",
    )

    with (
        patch("codepilot.agents.orchestrator.DeepAgentsCoderAdapter") as adapter_type,
        patch("codepilot.agents.orchestrator.CoderAgent") as coder_type,
    ):
        coder_type.return_value.run.return_value = expected

        result = orchestrator.run_coding_task(issue, str(tmp_path))

    assert result is expected
    assert root_agent.invoke.call_count == 2

    retry_input = root_agent.invoke.call_args_list[1].args[0]
    retry_messages = retry_input["messages"]

    assert retry_messages[0]["content"] == "Initial task"
    assert "implementation_planner" in retry_messages[-1]["content"]
    assert "continue" in retry_messages[-1]["content"].casefold()

    coder_type.return_value.run.assert_called_once_with(
        issue,
        str(tmp_path),
    )

    adapter_type.assert_called_once()


def test_live_coding_path_uses_prepared_task_and_root_handoff(
    issue: GitHubIssue, tmp_path: Path
) -> None:
    root_agent = MagicMock()
    root_agent.invoke.return_value = _delegated_response()
    coding_agent = MagicMock()
    workspace = Mock()
    factory = MagicMock()
    orchestrator = _make_orchestrator(
        root_agent=root_agent,
        coding_agent=coding_agent,
        workspace=workspace,
        agent_factory=factory,
    )
    prepared_task = {
        "prompt": "PREPARED_TASK_SENTINEL: skill and retrieved context",
        "category": "bug_fix",
    }
    orchestrator.prepare_task = Mock(return_value=prepared_task)  # type: ignore[method-assign]
    expected = CoderResult(
        status=CoderStatus.FAILED,
        issue_number=issue.number,
        workspace=Mock(),
        plan="",
        attempts=0,
        error="test sentinel",
    )

    with (
        patch("codepilot.agents.orchestrator.DeepAgentsCoderAdapter") as adapter_type,
        patch("codepilot.agents.orchestrator.CoderAgent") as coder_type,
    ):
        coder_type.return_value.run.return_value = expected
        result = orchestrator.run_coding_task(issue, str(tmp_path))

    assert result is expected
    orchestrator.prepare_task.assert_called_once_with(
        issue,
        repository_root=tmp_path,
    )
    root_agent.invoke.assert_called_once()
    root_prompt = root_agent.invoke.call_args.args[0]["messages"][0]["content"]
    assert prepared_task["prompt"] in root_prompt
    assert "write_todos" in root_prompt
    assert "repo_explorer" in root_prompt
    assert "implementation_planner" in root_prompt

    coding_agent_system_prompt = factory.call_args_list[1].kwargs["system_prompt"]
    # Issue text, retrieved memories, and specialist output are task data and
    # must not be elevated into the agent's privileged system instructions.
    assert prepared_task["prompt"] not in coding_agent_system_prompt
    assert "src/calculator.py" not in coding_agent_system_prompt
    # The static system prompt legitimately documents the required Handoff
    # format; assert that the *observed handoff content* is not embedded there.
    assert (
        "inspect src/calculator.py, multiply price by quantity, "
        "and verify with focused tests." not in coding_agent_system_prompt
    )

    adapter_kwargs = adapter_type.call_args.kwargs
    task_context = adapter_kwargs["task_context"]
    assert prepared_task["prompt"] in task_context
    assert "src/calculator.py" in task_context
    assert "Handoff:" in task_context
    adapter_type.assert_called_once_with(coding_agent, workspace, task_context=task_context)
    coder_type.return_value.run.assert_called_once_with(issue, str(tmp_path))


def test_live_coding_path_fails_closed_without_real_delegation(
    issue: GitHubIssue, tmp_path: Path
) -> None:
    root_agent = MagicMock()
    root_agent.invoke.return_value = {
        "messages": [{"role": "assistant", "content": "I will fix it directly."}]
    }
    coding_agent = MagicMock()
    workspace = Mock()
    factory = MagicMock()
    orchestrator = _make_orchestrator(
        root_agent=root_agent,
        coding_agent=coding_agent,
        workspace=workspace,
        agent_factory=factory,
    )
    orchestrator.prepare_task = Mock(  # type: ignore[method-assign]
        return_value={"prompt": "Prepared issue prompt", "category": "bug_fix"}
    )

    with pytest.raises(RuntimeError, match="required DeepAgents delegation"):
        orchestrator.run_coding_task(issue, str(tmp_path))

    assert root_agent.invoke.call_count == 3
    assert factory.call_count == 1  # no separate coding agent is created after failed delegation
