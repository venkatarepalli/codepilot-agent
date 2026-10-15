"""Tests for Skills System integration with the CodePilot Orchestrator."""

import pytest

from codepilot.agents.classifier import IssueCategory
from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.github.models import GitHubIssue
from codepilot.skills.skill_registry import get_skill


@pytest.fixture
def orchestrator() -> CodePilotOrchestrator:
    """Create an orchestrator with a minimal mock-compatible agent factory."""

    class DummyAgent:
        def invoke(self, payload):
            return payload

    def agent_factory(**kwargs):
        return DummyAgent()

    return CodePilotOrchestrator(
        model=object(),
        repository="example/repo",
        agent_factory=agent_factory,
    )


@pytest.mark.parametrize(
    ("category", "title", "body"),
    [
        (IssueCategory.BUG_FIX, "Fix: incorrect calculation", "Fix the bug."),
        (
            IssueCategory.FEATURE_ADDITION,
            "Add a new feature",
            "Implement the requested capability.",
        ),
        (
            IssueCategory.DEPENDENCY_UPDATE,
            "Update dependency",
            "Upgrade dependency to a compatible version.",
        ),
        (
            IssueCategory.DOCUMENTATION,
            "Update README",
            "Improve the documentation.",
        ),
        (
            IssueCategory.CONFIG_CHANGE,
            "Update configuration",
            "Add a configuration option.",
        ),
    ],
)
def test_task_prompt_contains_selected_skill_guidance(
    orchestrator: CodePilotOrchestrator,
    category: IssueCategory,
    title: str,
    body: str,
) -> None:
    """The task prompt should contain guidance from the selected skill."""
    issue = GitHubIssue(
        number=123,
        title=title,
        body=body,
        labels=[category.value],
    )

    prompt = orchestrator.build_task_prompt(issue)
    skill = get_skill(category)

    assert f"Task category: {category.value}" in prompt
    assert f"Selected skill: {skill.name}" in prompt
    assert skill.instructions in prompt

    for step in skill.workflow_steps:
        assert step in prompt

    for action in skill.forbidden_actions:
        assert action in prompt


def test_task_prompt_preserves_issue_details(
    orchestrator: CodePilotOrchestrator,
) -> None:
    """Skill integration should preserve the original issue details."""
    issue = GitHubIssue(
        number=456,
        title="Fix: incorrect calculation",
        body="The calculated total is incorrect.",
        labels=["bug_fix"],
        url="https://github.com/example/repo/issues/456",
        reporter="example-user",
    )

    prompt = orchestrator.build_task_prompt(issue)

    assert "GitHub issue #456" in prompt
    assert "Fix: incorrect calculation" in prompt
    assert "The calculated total is incorrect." in prompt
    assert "https://github.com/example/repo/issues/456" in prompt
    assert "example-user" in prompt


def test_prepare_task_preserves_category_and_initial_status(
    orchestrator: CodePilotOrchestrator,
) -> None:
    """Task preparation should preserve classification and lifecycle state."""
    issue = GitHubIssue(
        number=789,
        title="Update configuration",
        body="Add a configuration option.",
        labels=["config_change"],
    )

    task = orchestrator.prepare_task(issue)

    assert task["category"] == IssueCategory.CONFIG_CHANGE
    assert task["status"].value == "TRIAGED"
    assert "Selected skill: config_change_skill" in task["prompt"]
