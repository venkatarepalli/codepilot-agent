"""The Active Task panel shows the skill the Orchestrator selects."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from codepilot import runner
from codepilot.github.models import GitHubIssue


def test_skill_name_matches_orchestrator_selection() -> None:
    issue = GitHubIssue(
        number=5,
        title="Fix typo in README installation section",
        body='The README says "instalation" instead of "installation".',
        labels=("ai-assignable",),
    )
    assert runner._skill_name_for(issue) == "documentation_skill"
    assert runner._skill_name_for(SimpleNamespace()) == "—"


def test_active_task_updates_carry_current_skill() -> None:
    runtime = runner.CodePilotRuntime.__new__(runner.CodePilotRuntime)
    runtime._current_skill = "bug_fix_skill"
    runtime._post_ui = MagicMock()

    runtime._update_task("Issue #1: x", "IMPLEMENTING", agent="Coder")

    payload = runtime._post_ui.call_args.args[1]
    assert payload["skill"] == "bug_fix_skill"
