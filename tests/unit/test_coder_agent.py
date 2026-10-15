"""Tests for the Coder Agent implementation workflow."""

import sys
from pathlib import Path

import pytest

from codepilot.agents.coder import CoderAgent
from codepilot.agents.coder_models import CoderStatus
from codepilot.github.models import GitHubIssue
from codepilot.sandbox.coder_workspace import CoderWorkspace
from codepilot.sandbox.execution_models import CommandStatus
from codepilot.sandbox.executor import SandboxExecutor
from codepilot.sandbox.policy import SandboxPolicy
from codepilot.sandbox.workspace import WorkspaceManager


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create a minimal source repository."""
    root = tmp_path / "repository"
    root.mkdir()

    (root / "app.py").write_text(
        "answer = 1\n",
        encoding="utf-8",
    )

    return root


@pytest.fixture
def coder_workspace(tmp_path: Path) -> CoderWorkspace:
    """Build the workspace integration for tests."""
    workspace_root = tmp_path / "workspaces"
    workspace_root.mkdir()

    manager = WorkspaceManager(workspace_root)

    return CoderWorkspace(
        manager,
        SandboxExecutor(
            SandboxPolicy(workspace_root),
            timeout_seconds=2.0,
        ),
    )


@pytest.fixture
def issue() -> GitHubIssue:
    """Create a sample GitHub issue."""
    return GitHubIssue(
        number=123,
        title="Update the answer",
        body="Change the answer to 42.",
        labels=(),
    )


def make_agent(
    workspace: CoderWorkspace,
    *,
    planner,
    implementer,
    test_code: str = "raise SystemExit(0)",
) -> CoderAgent:
    """Create a Coder Agent with a deterministic test command."""
    return CoderAgent(
        workspace,
        planner=planner,
        implementer=implementer,
        test_command=[
            sys.executable,
            "-c",
            test_code,
        ],
    )


def test_successful_implementation_generates_diff(
    coder_workspace: CoderWorkspace,
    repository: Path,
    issue: GitHubIssue,
) -> None:
    """Passing tests produce a saved proposed diff."""

    def implementer(issue, workspace, plan, attempt, previous_result):
        coder_workspace.write_file(
            workspace,
            "app.py",
            "answer = 42\n",
        )

    agent = make_agent(
        coder_workspace,
        planner=lambda issue, workspace: "Update answer to 42.",
        implementer=implementer,
    )

    result = agent.run(issue, repository)

    assert result.status == CoderStatus.SUCCEEDED, f"Coder Agent failed: {result.error}"
    assert result.attempts == 1
    assert result.test_result is not None
    assert result.test_result.status == CommandStatus.SUCCEEDED
    assert result.diff_path is not None
    assert result.diff_path.is_file()

    diff = result.diff_path.read_text(encoding="utf-8")

    assert "-answer = 1" in diff
    assert "+answer = 42" in diff

    assert (repository / "app.py").read_text(encoding="utf-8") == "answer = 1\n"


def test_failed_test_is_retried(
    coder_workspace: CoderWorkspace,
    repository: Path,
    issue: GitHubIssue,
) -> None:
    """The next implementation attempt receives the prior test result."""
    received_results = []

    def implementer(issue, workspace, plan, attempt, previous_result):
        received_results.append(previous_result)

        if attempt == 1:
            coder_workspace.write_file(
                workspace,
                "app.py",
                "answer = 1\n",
            )
        else:
            coder_workspace.write_file(
                workspace,
                "app.py",
                "answer = 42\n",
            )

    agent = make_agent(
        coder_workspace,
        planner=lambda issue, workspace: "Update the answer.",
        implementer=implementer,
        test_code=(
            "from pathlib import Path; "
            "text = Path('app.py').read_text(); "
            "raise SystemExit(0 if '42' in text else 1)"
        ),
    )

    result = agent.run(issue, repository)

    assert result.status == CoderStatus.SUCCEEDED
    assert result.attempts == 2
    assert received_results[0] is None
    assert received_results[1] is not None
    assert received_results[1].status == CommandStatus.FAILED


def test_stops_after_three_failed_attempts(
    coder_workspace: CoderWorkspace,
    repository: Path,
    issue: GitHubIssue,
) -> None:
    """The implementation workflow must not exceed three attempts."""
    attempts = []

    def implementer(issue, workspace, plan, attempt, previous_result):
        attempts.append(attempt)

    agent = make_agent(
        coder_workspace,
        planner=lambda issue, workspace: "Do the task.",
        implementer=implementer,
        test_code="raise SystemExit(1)",
    )

    result = agent.run(issue, repository)

    assert result.status == CoderStatus.FAILED
    assert result.attempts == 3
    assert attempts == [1, 2, 3]
    assert result.test_result is not None
    assert result.test_result.status == CommandStatus.FAILED


def test_planning_failure_does_not_start_implementation(
    coder_workspace: CoderWorkspace,
    repository: Path,
    issue: GitHubIssue,
) -> None:
    """A failed planner returns a structured failure."""
    calls = []

    agent = make_agent(
        coder_workspace,
        planner=lambda issue, workspace: "",
        implementer=lambda *args: calls.append(args),
    )

    result = agent.run(issue, repository)

    assert result.status == CoderStatus.FAILED
    assert result.attempts == 0
    assert "Planning failed" in result.error
    assert calls == []


def test_implementation_exception_returns_failure(
    coder_workspace: CoderWorkspace,
    repository: Path,
    issue: GitHubIssue,
) -> None:
    """Implementation exceptions are captured as structured failures."""

    def broken_implementer(issue, workspace, plan, attempt, previous_result):
        raise RuntimeError("simulated implementation failure")

    agent = make_agent(
        coder_workspace,
        planner=lambda issue, workspace: "Implement the change.",
        implementer=broken_implementer,
    )

    result = agent.run(issue, repository)

    assert result.status == CoderStatus.FAILED
    assert result.attempts == 1
    assert "simulated implementation failure" in result.error


def test_empty_change_set_retries_implementation(
    coder_workspace: CoderWorkspace,
    repository: Path,
    issue: GitHubIssue,
) -> None:
    """An empty change set triggers a retry with explicit feedback."""
    attempts = []
    received_results = []

    def implementer(issue, workspace, plan, attempt, previous_result):
        attempts.append(attempt)
        received_results.append(previous_result)

        if attempt == 1:
            raise ValueError(
                "DeepAgents returned an empty change set. "
                "The model did not propose any file changes."
            )

        coder_workspace.write_file(
            workspace,
            "app.py",
            "answer = 42\n",
        )

    agent = make_agent(
        coder_workspace,
        planner=lambda issue, workspace: "Update the answer to 42.",
        implementer=implementer,
    )

    result = agent.run(issue, repository)

    assert result.status == CoderStatus.SUCCEEDED, f"Coder Agent failed: {result.error}"
    assert result.attempts == 2
    assert attempts == [1, 2]

    # The first attempt has no previous failure assessment.
    assert received_results[0] is None

    # The second attempt receives feedback about the empty change set.
    assert received_results[1] is not None
    assert "empty change set" in received_results[1].summary.lower()
    assert "non-empty" in received_results[1].summary.lower()

    assert result.diff_path is not None
    assert result.diff_path.is_file()

    # The original repository must remain unchanged.
    assert (repository / "app.py").read_text(encoding="utf-8") == "answer = 1\n"


def test_empty_plan_is_rejected(
    coder_workspace: CoderWorkspace,
    repository: Path,
    issue: GitHubIssue,
) -> None:
    """The planner must return meaningful instructions."""
    agent = make_agent(
        coder_workspace,
        planner=lambda issue, workspace: "   ",
        implementer=lambda *args: None,
    )

    result = agent.run(issue, repository)

    assert result.status == CoderStatus.FAILED
    assert "non-empty implementation plan" in result.error


def test_test_command_failure_is_reported(
    coder_workspace: CoderWorkspace,
    repository: Path,
    issue: GitHubIssue,
) -> None:
    """An executor exception is returned as a task failure."""

    def broken_execute(*args, **kwargs):
        raise RuntimeError("executor unavailable")

    coder_workspace.execute = broken_execute

    agent = make_agent(
        coder_workspace,
        planner=lambda issue, workspace: "Implement the change.",
        implementer=lambda *args: None,
    )

    result = agent.run(issue, repository)

    assert result.status == CoderStatus.FAILED
    assert "Test execution failed" in result.error
    assert result.attempts == 1
