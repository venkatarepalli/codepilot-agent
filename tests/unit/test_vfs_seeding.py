"""Component 2: relevant repo files are loadable on demand via read_file in the agents' VFS."""

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from codepilot.agents import orchestrator as orchestrator_module
from codepilot.agents.orchestrator import CodePilotOrchestrator, stream_agent_with_logging


def _context(*paths: str) -> SimpleNamespace:
    return SimpleNamespace(
        repo_map_text="Repository map\n- README.md",
        files=[SimpleNamespace(path=path) for path in paths],
    )


def _repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("Instalation steps:\n", encoding="utf-8")
    (tmp_path / "src" / "calculator.py").write_text("def calculate_total(): ...\n", "utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-secret\n", encoding="utf-8")
    (tmp_path / "deploy.pem").write_text("PRIVATE\n", encoding="utf-8")
    return tmp_path


def test_relevant_files_are_seeded_and_secrets_never_are(tmp_path, caplog) -> None:
    root = _repo(tmp_path)
    context = _context(
        "src/calculator.py", "README.md", ".env", "deploy.pem", "../outside.txt", "missing.py"
    )
    with caplog.at_level(logging.INFO, logger="codepilot"):
        files = CodePilotOrchestrator._initial_virtual_files(context, repository_root=root)

    assert set(files) == {"/repo-map.md", "/src/calculator.py", "/README.md"}
    assert files["/README.md"]["content"] == ["Instalation steps:"]
    assert "2 relevant file(s)" in caplog.text


def test_without_repository_root_only_the_repo_map_is_seeded() -> None:
    files = CodePilotOrchestrator._initial_virtual_files(_context("README.md"))
    assert set(files) == {"/repo-map.md"}


def test_seeding_is_bounded(tmp_path) -> None:
    for index in range(20):
        (tmp_path / f"f{index}.py").write_text("x = 1\n", encoding="utf-8")
    context = _context(*[f"f{index}.py" for index in range(20)])
    files = CodePilotOrchestrator._initial_virtual_files(context, repository_root=tmp_path)
    assert len(files) == 1 + CodePilotOrchestrator.VFS_MAX_FILES


class _ToolCallingFake(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


def test_real_deep_agent_can_read_seeded_file_on_demand(tmp_path, caplog) -> None:
    files = CodePilotOrchestrator._initial_virtual_files(
        _context("README.md"), repository_root=_repo(tmp_path)
    )
    model = _ToolCallingFake(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read_file", "args": {"file_path": "/README.md"}, "id": "r1"}
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )
    agent = create_deep_agent(model=model)
    with caplog.at_level(logging.INFO, logger="codepilot"):
        state = stream_agent_with_logging(
            agent, {"messages": [{"role": "user", "content": "read"}], "files": files}
        )

    tool_result = next(m for m in state["messages"] if getattr(m, "type", "") == "tool")
    assert "Instalation steps" in tool_result.content
    assert "not found" not in tool_result.content


def test_recovery_attempt_keeps_seeded_files(monkeypatch, tmp_path) -> None:
    calls: list[dict] = []

    def fake_stream(agent, state):
        calls.append(state)
        return {"messages": []}  # never satisfies the delegation contract

    monkeypatch.setattr(orchestrator_module, "stream_agent_with_logging", fake_stream)
    orch = CodePilotOrchestrator(
        model=object(),
        repository="o/r",
        agent_factory=lambda **kwargs: MagicMock(),
        coder_workspace=MagicMock(),
        test_command=("pytest",),
    )
    seeded = {"/repo-map.md": {"content": ["map"]}, "/README.md": {"content": ["x"]}}
    monkeypatch.setattr(
        CodePilotOrchestrator, "_initial_virtual_files", classmethod(lambda cls, *a, **k: seeded)
    )
    from codepilot.github.models import GitHubIssue

    try:
        orch.run_coding_task(GitHubIssue(number=1, title="Fix bug", body="x"), str(tmp_path))
    except RuntimeError:
        pass

    assert len(calls) == 3  # initial + two recovery attempts
    assert all(call["files"] == seeded for call in calls)


def test_stream_does_not_replay_history_messages(caplog) -> None:
    history = AIMessage(
        content="",
        id="old-1",
        tool_calls=[{"name": "task", "args": {"subagent_type": "repo_explorer"}, "id": "t0"}],
    )
    model = _ToolCallingFake(messages=iter([AIMessage(content="Handoff: ok")]))
    agent = create_deep_agent(model=model)
    with caplog.at_level(logging.INFO, logger="codepilot"):
        stream_agent_with_logging(
            agent,
            {
                "messages": [
                    {"role": "user", "content": "start"},
                    history,
                    {"role": "tool", "content": "r", "tool_call_id": "t0"},
                    {"role": "user", "content": "continue"},
                ]
            },
        )
    assert "task → repo_explorer" not in caplog.text
    assert "[Orchestrator] response: Handoff: ok" in caplog.text
