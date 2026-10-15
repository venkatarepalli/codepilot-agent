"""Tests for publishing successful Coder Agent results as pull requests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codepilot.agents.coder_models import CoderResult, CoderStatus
from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.github.models import GitHubIssue
from codepilot.github.pr_workflow import PullRequestResult
from codepilot.memory.semantic_memory import TaskLesson
from codepilot.sandbox.coder_workspace import CoderWorkspace
from codepilot.sandbox.execution_models import (
    CommandExecutionResult,
    CommandStatus,
)
from codepilot.sandbox.workspace import WorkspaceInfo


@pytest.fixture
def issue() -> GitHubIssue:
    """Provide an issue with a URL and reporter."""
    return GitHubIssue(
        number=42,
        title="Fix calculation bug",
        body="Correct the calculation.",
        url="https://github.com/example-owner/example-repo/issues/42",
        labels=("bug",),
        reporter="reporter",
    )


@pytest.fixture
def workspace_info(tmp_path: Path) -> WorkspaceInfo:
    """Provide isolated workspace metadata."""
    source = tmp_path / "repository"
    source.mkdir()

    workspace_path = tmp_path / "workspaces" / "issue-42"
    workspace_path.mkdir(parents=True)

    return WorkspaceInfo(
        issue_number=42,
        path=workspace_path,
        source_repository=source,
    )


@pytest.fixture
def coder_workspace() -> MagicMock:
    """Mock the controlled workspace interface."""
    workspace = MagicMock(spec=CoderWorkspace)
    workspace.get_changes.return_value = {
        "added": ["src/new_module.py"],
        "modified": ["src/app.py"],
        "deleted": [],
    }
    workspace.read_file.side_effect = lambda _workspace, path: {
        "src/app.py": "def calculate():\n    return 42\n",
        "src/new_module.py": "ANSWER = 42\n",
    }[str(path)]
    return workspace


@pytest.fixture
def pull_request_workflow() -> MagicMock:
    """Return a mocked PR workflow."""
    workflow = MagicMock()
    workflow.is_merged.return_value = False
    workflow.publish.return_value = PullRequestResult(
        branch="codepilot/issue-42-fix-calculation-bug",
        pull_number=123,
        pull_url="https://github.com/example-owner/example-repo/pull/123",
        title="[CodePilot] Fix calculation bug",
        base_branch="main",
    )
    return workflow


@pytest.fixture
def semantic_memory() -> MagicMock:
    """Provide mocked semantic memory."""
    return MagicMock()


@pytest.fixture
def orchestrator(
    coder_workspace: MagicMock,
    pull_request_workflow: MagicMock,
    semantic_memory: MagicMock,
) -> CodePilotOrchestrator:
    """Create an orchestrator with mocked publishing dependencies."""
    return CodePilotOrchestrator(
        model="test-model",
        repository="example-owner/example-repo",
        agent_factory=MagicMock(return_value=MagicMock()),
        coder_workspace=coder_workspace,
        test_command=["pytest", "-q"],
        pr_workflow=pull_request_workflow,
        semantic_memory=semantic_memory,
    )


@pytest.fixture
def successful_result(workspace_info: WorkspaceInfo) -> CoderResult:
    """Provide a successful coding result with passing test evidence."""
    diff_path = workspace_info.path / "working" / "proposed_diff.txt"
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(
        "--- a/src/app.py\n+++ b/src/app.py\n",
        encoding="utf-8",
    )

    return CoderResult(
        status=CoderStatus.SUCCEEDED,
        issue_number=42,
        workspace=workspace_info,
        plan="Correct the calculation.",
        attempts=1,
        test_result=CommandExecutionResult(
            status=CommandStatus.SUCCEEDED,
            command=("pytest", "-q"),
            return_code=0,
            stdout="Tests passed.",
        ),
        diff_path=diff_path,
    )


def test_publish_coder_result_passes_complete_file_contents(
    orchestrator: CodePilotOrchestrator,
    issue: GitHubIssue,
    successful_result: CoderResult,
    coder_workspace: MagicMock,
    pull_request_workflow: MagicMock,
) -> None:
    """Publish complete workspace contents rather than the unified diff."""
    result = orchestrator.publish_coder_result(
        issue,
        successful_result,
        commit_message="fix(#42): correct calculation",
        summary="Correct the calculation.",
    )

    assert result.pull_number == 123

    kwargs = pull_request_workflow.publish.call_args.kwargs

    assert kwargs["issue_number"] == 42
    assert kwargs["issue_title"] == issue.title
    assert kwargs["issue_url"] == issue.url
    assert kwargs["reviewer"] == issue.reporter
    assert kwargs["changes"] == {
        "src/app.py": "def calculate():\n    return 42\n",
        "src/new_module.py": "ANSWER = 42\n",
    }

    assert kwargs["changes"]["src/app.py"] != successful_result.diff_path.read_text(
        encoding="utf-8"
    )

    assert coder_workspace.read_file.call_count == 2


def test_failed_coder_result_is_not_published(
    orchestrator: CodePilotOrchestrator,
    issue: GitHubIssue,
    successful_result: CoderResult,
    pull_request_workflow: MagicMock,
) -> None:
    """Reject unsuccessful coding results."""
    failed_result = CoderResult(
        status=CoderStatus.FAILED,
        issue_number=successful_result.issue_number,
        workspace=successful_result.workspace,
        error="Tests failed.",
    )

    with pytest.raises(ValueError, match="failed"):
        orchestrator.publish_coder_result(
            issue,
            failed_result,
            commit_message="fix(#42): correct calculation",
            summary="Correct the calculation.",
        )

    pull_request_workflow.publish.assert_not_called()


def test_mismatched_issue_is_not_published(
    orchestrator: CodePilotOrchestrator,
    issue: GitHubIssue,
    successful_result: CoderResult,
    pull_request_workflow: MagicMock,
) -> None:
    """Reject a coding result belonging to a different issue."""
    mismatched_result = CoderResult(
        status=CoderStatus.SUCCEEDED,
        issue_number=43,
        workspace=successful_result.workspace,
        plan=successful_result.plan,
        attempts=1,
        test_result=successful_result.test_result,
        diff_path=successful_result.diff_path,
    )

    with pytest.raises(ValueError, match="does not match"):
        orchestrator.publish_coder_result(
            issue,
            mismatched_result,
            commit_message="fix(#42): correct calculation",
            summary="Correct the calculation.",
        )

    pull_request_workflow.publish.assert_not_called()


def test_deleted_files_are_rejected_without_publishing(
    orchestrator: CodePilotOrchestrator,
    issue: GitHubIssue,
    successful_result: CoderResult,
    coder_workspace: MagicMock,
    pull_request_workflow: MagicMock,
) -> None:
    """Never silently omit deleted files from a proposed PR."""
    coder_workspace.get_changes.return_value = {
        "added": [],
        "modified": [],
        "deleted": ["src/obsolete.py"],
    }

    with pytest.raises(NotImplementedError, match="deleted files"):
        orchestrator.publish_coder_result(
            issue,
            successful_result,
            commit_message="fix(#42): correct calculation",
            summary="Correct the calculation.",
        )

    pull_request_workflow.publish.assert_not_called()


def test_merged_pr_records_semantic_lesson(
    orchestrator: CodePilotOrchestrator,
    issue: GitHubIssue,
    successful_result: CoderResult,
    pull_request_workflow: MagicMock,
    semantic_memory: MagicMock,
) -> None:
    """Store a lesson only after GitHub confirms the PR was merged."""
    pull_request_workflow.is_merged.return_value = True
    pr_result = pull_request_workflow.publish.return_value

    stored = orchestrator.record_merged_pr_lesson(
        issue,
        successful_result,
        pr_result,
        summary="Corrected the calculation and added a regression test.",
    )

    assert stored is True
    pull_request_workflow.is_merged.assert_called_once_with(pr_result.pull_number)
    semantic_memory.store_lesson.assert_called_once()
    lesson = semantic_memory.store_lesson.call_args.args[0]
    assert isinstance(lesson, TaskLesson)
    assert lesson.lesson_id == f"pull-request-{pr_result.pull_number}"
    assert lesson.repository == "example-owner/example-repo"
    assert lesson.problem.startswith(issue.title)
    assert lesson.resolution == "Corrected the calculation and added a regression test."
    assert lesson.outcome == "merged"


def test_unmerged_pr_does_not_record_semantic_lesson(
    orchestrator: CodePilotOrchestrator,
    issue: GitHubIssue,
    successful_result: CoderResult,
    pull_request_workflow: MagicMock,
    semantic_memory: MagicMock,
) -> None:
    """Do not learn from a PR that is still open or otherwise unmerged."""
    pull_request_workflow.is_merged.return_value = False

    stored = orchestrator.record_merged_pr_lesson(
        issue,
        successful_result,
        pull_request_workflow.publish.return_value,
        summary="Corrected the calculation.",
    )

    assert stored is False
    semantic_memory.store_lesson.assert_not_called()


def test_failed_coder_result_does_not_record_semantic_lesson(
    orchestrator: CodePilotOrchestrator,
    issue: GitHubIssue,
    successful_result: CoderResult,
    pull_request_workflow: MagicMock,
    semantic_memory: MagicMock,
) -> None:
    """Reject failed work before checking or recording a PR lesson."""
    failed_result = CoderResult(
        status=CoderStatus.FAILED,
        issue_number=successful_result.issue_number,
        workspace=successful_result.workspace,
        error="Tests failed.",
    )

    with pytest.raises(ValueError, match="failed"):
        orchestrator.record_merged_pr_lesson(
            issue,
            failed_result,
            pull_request_workflow.publish.return_value,
            summary="This must not be stored.",
        )

    pull_request_workflow.is_merged.assert_not_called()
    semantic_memory.store_lesson.assert_not_called()


def test_semantic_memory_not_configured_is_safe_noop(
    coder_workspace: MagicMock,
    pull_request_workflow: MagicMock,
    issue: GitHubIssue,
    successful_result: CoderResult,
) -> None:
    """Skip lesson storage when semantic memory is disabled."""
    orchestrator = CodePilotOrchestrator(
        model="test-model",
        repository="example-owner/example-repo",
        agent_factory=MagicMock(return_value=MagicMock()),
        coder_workspace=coder_workspace,
        test_command=["pytest", "-q"],
        pr_workflow=pull_request_workflow,
    )

    stored = orchestrator.record_merged_pr_lesson(
        issue,
        successful_result,
        pull_request_workflow.publish.return_value,
        summary="Corrected the calculation.",
    )

    assert stored is False
    pull_request_workflow.is_merged.assert_not_called()


def test_rechecking_merged_pr_uses_same_lesson_id(
    orchestrator: CodePilotOrchestrator,
    issue: GitHubIssue,
    successful_result: CoderResult,
    pull_request_workflow: MagicMock,
    semantic_memory: MagicMock,
) -> None:
    """Use a deterministic lesson ID so Chroma upsert avoids duplicate records."""
    pull_request_workflow.is_merged.return_value = True
    pr_result = pull_request_workflow.publish.return_value

    for _ in range(2):
        assert (
            orchestrator.record_merged_pr_lesson(
                issue,
                successful_result,
                pr_result,
                summary="Corrected the calculation.",
            )
            is True
        )

    lesson_ids = [call.args[0].lesson_id for call in semantic_memory.store_lesson.call_args_list]
    assert lesson_ids == [f"pull-request-{pr_result.pull_number}"] * 2
