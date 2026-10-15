"""Focused tests for the DeepAgents-to-CoderAgent adapter."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from codepilot.agents.deepagents_coder_adapter import DeepAgentsCoderAdapter
from codepilot.github.models import GitHubIssue


@pytest.fixture
def issue() -> GitHubIssue:
    return GitHubIssue(number=42, title="Fix calculation", body="Return 42.")


@pytest.fixture
def workspace(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    return SimpleNamespace(path=root, issue_number=42, source_repository=root)


def response(content: str) -> dict:
    return {"messages": [{"role": "assistant", "content": content}]}


def test_planner_uses_issue_and_workspace_context(issue, workspace) -> None:
    agent = Mock()
    agent.invoke.return_value = response("1. Inspect answer().\n2. Change result to 42.")
    adapter = DeepAgentsCoderAdapter(agent, Mock())

    plan = adapter.planner(issue, workspace)

    assert "Change result to 42" in plan
    prompt = agent.invoke.call_args.args[0]["messages"][0]["content"]
    assert "Fix calculation" in prompt
    assert "app.py" in prompt
    assert "return 1" in prompt


def test_implementer_applies_valid_change_through_workspace(issue, workspace) -> None:
    agent = Mock()
    change_set = {"changes": [{"path": "app.py", "content": "def answer():\n    return 42\n"}]}
    agent.invoke.return_value = response(json.dumps(change_set))
    coder_workspace = Mock()
    adapter = DeepAgentsCoderAdapter(agent, coder_workspace)

    adapter.implementer(issue, workspace, "Change the return value.", 1, None)

    coder_workspace.write_file.assert_called_once_with(
        workspace, "app.py", "def answer():\n    return 42\n"
    )


def test_implementer_rejects_empty_change_list(issue, workspace):
    agent = Mock()
    agent.invoke.return_value = response('{"changes": []}')
    coder_workspace = Mock()

    adapter = DeepAgentsCoderAdapter(agent, coder_workspace)

    with pytest.raises(
        ValueError,
        match="DeepAgents returned an empty change set",
    ):
        adapter.implementer(
            issue,
            workspace,
            "Change the return value.",
            1,
            None,
        )

    coder_workspace.write_file.assert_not_called()


@pytest.mark.parametrize(
    "path",
    [
        "../outside.py",
        "C:/outside.py",
        "/outside.py",
        "working/proposed_diff.txt",
        ".env",
        "service.key",
        "credentials.json",
    ],
)
def test_implementer_rejects_unsafe_paths(issue, workspace, path) -> None:
    agent = Mock()
    change_set = {"changes": [{"path": path, "content": "unsafe"}]}
    agent.invoke.return_value = response(json.dumps(change_set))
    coder_workspace = Mock()
    adapter = DeepAgentsCoderAdapter(agent, coder_workspace)

    with pytest.raises(ValueError):
        adapter.implementer(issue, workspace, "Plan", 1, None)
    coder_workspace.write_file.assert_not_called()


def test_implementer_rejects_invalid_json_without_writing(issue, workspace) -> None:
    agent = Mock()
    agent.invoke.return_value = response("Here is the code: app.py changed")
    coder_workspace = Mock()
    adapter = DeepAgentsCoderAdapter(agent, coder_workspace)

    with pytest.raises(ValueError, match="valid JSON"):
        adapter.implementer(issue, workspace, "Plan", 1, None)
    coder_workspace.write_file.assert_not_called()


def test_parse_change_set_extracts_json_after_explanatory_text():
    response_text = (
        "I updated the implementation. Here is the corrected file:\n"
        "```python\n"
        "def calculate_total(price, quantity):\n"
        "    return price * quantity\n"
        "```\n"
        "JSON response follows:\n"
        '{"changes":[{"path":"src/calculator.py",'
        '"content":"def calculate_total(price, quantity):\\n'
        '    return price * quantity\\n"}]}'
    )

    result = DeepAgentsCoderAdapter._parse_change_set(response_text)

    assert len(result["changes"]) == 1
    assert result["changes"][0]["path"] == "src/calculator.py"
    assert "return price * quantity" in result["changes"][0]["content"]


def test_context_does_not_include_environment_files(issue, tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "app.py").write_text("safe", encoding="utf-8")
    (root / ".env").write_text("API_KEY=secret", encoding="utf-8")
    agent = Mock()
    agent.invoke.return_value = response("Plan")
    adapter = DeepAgentsCoderAdapter(agent, Mock())

    adapter.planner(issue, SimpleNamespace(path=root))

    prompt = agent.invoke.call_args.args[0]["messages"][0]["content"]
    assert "safe" in prompt
    assert "API_KEY" not in prompt
