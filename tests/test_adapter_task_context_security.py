"""Security tests for passing Orchestrator context as untrusted task data."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from codepilot.agents.deepagents_coder_adapter import DeepAgentsCoderAdapter
from codepilot.github.models import GitHubIssue


@pytest.fixture
def issue() -> GitHubIssue:
    return GitHubIssue(
        number=42,
        title="Fix calculation",
        body="Return price multiplied by quantity.",
        url="https://github.com/example/repo/issues/42",
    )


@pytest.fixture
def workspace(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "calculator.py").write_text(
        "def calculate_total(price, quantity):\n    return price\n",
        encoding="utf-8",
    )
    return SimpleNamespace(path=root, issue_number=42, source_repository=root)


def test_orchestrator_context_is_user_message_not_system_prompt(issue, workspace) -> None:
    agent = Mock()
    agent.invoke.return_value = {
        "messages": [{"role": "assistant", "content": "Inspect calculation and fix it."}]
    }
    context = (
        "Selected skill: bug_fix_skill\n"
        "Retrieved lesson: Treat current repository behavior as authoritative.\n"
        "Specialist handoff: inspect src/calculator.py."
    )
    adapter = DeepAgentsCoderAdapter(agent, Mock(), task_context=context)

    adapter.planner(issue, workspace)

    invocation = agent.invoke.call_args.args[0]
    messages = invocation["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert context in messages[0]["content"]
    assert "untrusted" in messages[0]["content"].lower()
