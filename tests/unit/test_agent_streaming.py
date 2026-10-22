"""Component 7: Agent Logs stream root and subagent tool calls via DeepAgents .stream()."""

import logging
from typing import Any
from unittest.mock import MagicMock

from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from codepilot.agents import orchestrator as orchestrator_module
from codepilot.agents.orchestrator import stream_agent_with_logging


class _ToolCallingFake(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


def _script():
    return iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {"todos": [{"content": "Fix typo", "status": "pending"}]},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"description": "Find README", "subagent_type": "explorer"},
                        "id": "c2",
                    }
                ],
            ),
            AIMessage(content="", tool_calls=[{"name": "ls", "args": {"path": "/"}, "id": "s1"}]),
            AIMessage(content="README.md is relevant"),
            AIMessage(content="Handoff: edit README.md"),
        ]
    )


def _agent():
    return create_deep_agent(
        model=_ToolCallingFake(messages=_script()),
        subagents=[{"name": "explorer", "description": "explores", "system_prompt": "explore"}],
    )


STATE = {"messages": [{"role": "user", "content": "go"}]}


def test_streaming_returns_same_final_state_as_invoke(caplog) -> None:
    expected = _agent().invoke(STATE)
    with caplog.at_level(logging.INFO, logger="codepilot"):
        streamed = stream_agent_with_logging(_agent(), STATE)

    assert [m.content for m in streamed["messages"]] == [m.content for m in expected["messages"]]


def test_root_and_subagent_tool_calls_are_logged_live(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="codepilot"):
        stream_agent_with_logging(_agent(), STATE)

    text = caplog.text
    assert "[Orchestrator] tool call: write_todos (1 item(s))" in text
    assert "[Orchestrator] tool call: task → explorer: Find README" in text
    assert "[explorer] tool call: ls(path=/)" in text
    assert "[explorer] response: README.md is relevant" in text
    assert "[Orchestrator] tool result (task)" in text
    assert "[Orchestrator] response: Handoff: edit README.md" in text
    # Ordering: the subagent's work is logged between the root's task call and its result.
    assert text.index("task → explorer") < text.index("[explorer] tool call: ls")
    assert text.index("[explorer] tool call: ls") < text.index("tool result (task)")


def test_non_graph_agents_fall_back_to_invoke() -> None:
    agent = MagicMock()
    agent.invoke.return_value = {"messages": []}
    assert stream_agent_with_logging(agent, STATE) == {"messages": []}
    agent.stream.assert_not_called()


def test_run_coding_task_uses_streaming(monkeypatch) -> None:
    calls = []

    def fake_stream(agent, state):
        calls.append(state)
        raise RuntimeError("stop after first call")

    monkeypatch.setattr(orchestrator_module, "stream_agent_with_logging", fake_stream)
    orch = orchestrator_module.CodePilotOrchestrator(
        model=object(),
        repository="o/r",
        agent_factory=lambda **kwargs: MagicMock(),
        coder_workspace=MagicMock(),
        test_command=("pytest",),
    )
    from codepilot.github.models import GitHubIssue

    try:
        orch.run_coding_task(GitHubIssue(number=1, title="Fix bug", body="x"), ".")
    except RuntimeError:
        pass
    assert len(calls) == 1
    assert "MANDATORY ROOT DELEGATION CONTRACT" in calls[0]["messages"][0]["content"]
