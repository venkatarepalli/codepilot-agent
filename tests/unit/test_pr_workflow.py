"""Unit tests for the pull-request workflow."""

from unittest.mock import MagicMock

import pytest
from github import InputGitTreeElement

from codepilot.github.pr_workflow import (
    PullRequestResult,
    PullRequestWorkflow,
    PullRequestWorkflowError,
)


@pytest.fixture
def github_client() -> MagicMock:
    """Create a mocked GitHub client and repository."""
    client = MagicMock()
    repository = client.repository

    base_ref = MagicMock()
    base_ref.object.sha = "base-sha"
    repository.get_git_ref.return_value = base_ref

    parent_commit = MagicMock()
    parent_commit.tree = MagicMock()
    repository.get_git_commit.return_value = parent_commit

    blob = MagicMock()
    blob.sha = "blob-sha"
    repository.create_git_blob.return_value = blob

    tree = MagicMock()
    tree.sha = "tree-sha"
    repository.create_git_tree.return_value = tree

    commit = MagicMock()
    commit.sha = "new-commit-sha"
    repository.create_git_commit.return_value = commit

    pull_request = MagicMock()
    pull_request.number = 42
    pull_request.html_url = "https://github.com/example-owner/example-repo/pull/42"
    client.create_pull_request.return_value = pull_request

    return client


@pytest.fixture
def approvals() -> MagicMock:
    """Approve requested operations by default."""
    return MagicMock(return_value=True)


@pytest.fixture
def workflow(
    github_client: MagicMock,
    approvals: MagicMock,
) -> PullRequestWorkflow:
    """Create a workflow using mocked GitHub operations."""
    return PullRequestWorkflow(
        github_client=github_client,
        default_branch="main",
        approval_callback=approvals,
    )


@pytest.fixture
def publish_arguments() -> dict:
    """Return a valid issue and change set."""
    return {
        "issue_number": 12,
        "issue_title": "Fix login validation",
        "issue_url": "https://github.com/example-owner/example-repo/issues/12",
        "changes": {"src/login.py": "def login():\n    return True\n"},
        "commit_message": "fix: correct login validation",
        "summary": "Correct login validation.",
        "test_results": "Tests passed.",
        "reviewer": "reporter",
    }


def test_publish_creates_branch_commit_and_pull_request(
    workflow: PullRequestWorkflow,
    github_client: MagicMock,
    approvals: MagicMock,
    publish_arguments: dict,
) -> None:
    """Publish a commit and open a correctly configured pull request."""
    result = workflow.publish(**publish_arguments)

    assert isinstance(result, PullRequestResult)
    assert result.branch == "codepilot/issue-12-fix-login-validation"
    assert result.pull_number == 42
    assert result.pull_url.endswith("/pull/42")
    assert result.title == "[CodePilot] Fix login validation"
    assert result.base_branch == "main"

    repository = github_client.repository

    repository.get_git_ref.assert_called_once_with("heads/main")
    repository.get_git_commit.assert_called_once_with("base-sha")
    repository.create_git_blob.assert_called_once_with(
        "def login():\n    return True\n",
        "utf-8",
    )
    repository.create_git_tree.assert_called_once()

    tree_entries = repository.create_git_tree.call_args.args[0]
    assert len(tree_entries) == 1
    assert isinstance(tree_entries[0], InputGitTreeElement)

    # A free-form "fix: ..." message is normalized to the required format.
    repository.create_git_commit.assert_called_once_with(
        "fix(#12): correct login validation\n\n"
        "- Change: correct login validation (src/login.py)\n"
        "- Why: Resolve issue #12: Fix login validation\n"
        "- Closes #12",
        repository.create_git_tree.return_value,
        [repository.get_git_commit.return_value],
    )

    repository.create_git_ref.assert_called_once_with(
        ref="refs/heads/codepilot/issue-12-fix-login-validation",
        sha="new-commit-sha",
    )

    github_client.create_pull_request.assert_called_once()
    pr_arguments = github_client.create_pull_request.call_args.kwargs

    assert pr_arguments["title"] == "[CodePilot] Fix login validation"
    assert pr_arguments["head"] == result.branch
    assert pr_arguments["base"] == "main"
    assert pr_arguments["reviewer"] == "reporter"

    approval_operations = [call.args[0] for call in approvals.call_args_list]
    assert "publish_branch" in approval_operations
    assert "create_pull_request_against_protected_branch" in approval_operations

    for approval_call in approvals.call_args_list:
        details = approval_call.args[1]
        assert details["changed_files"] == ["src/login.py"]
        assert details["files_changed"] == len(details["changed_files"])


def test_approval_details_list_every_file_in_the_publication_payload(
    workflow: PullRequestWorkflow,
    approvals: MagicMock,
    publish_arguments: dict,
) -> None:
    """Approval details must identify the exact files being published."""
    publish_arguments["changes"] = {
        "tests/test_login.py": "def test_login():\n    assert True\n",
        "src/login.py": "def login():\n    return True\n",
    }

    workflow.publish(**publish_arguments)

    for approval_call in approvals.call_args_list:
        details = approval_call.args[1]
        assert details["changed_files"] == ["src/login.py", "tests/test_login.py"]
        assert details["files_changed"] == 2
        assert details["files_changed"] == len(details["changed_files"])


def test_publish_includes_summary_tests_and_issue_link(
    workflow: PullRequestWorkflow,
    github_client: MagicMock,
    publish_arguments: dict,
) -> None:
    """Include the implementation summary and test evidence in the PR body."""
    workflow.publish(**publish_arguments)

    body = github_client.create_pull_request.call_args.kwargs["body"]

    assert "Correct login validation." in body
    assert "Tests passed." in body
    assert "Closes #12" in body
    assert publish_arguments["issue_url"] in body
    assert "`src/login.py`" in body


def test_publish_requires_approval_before_branch_creation(
    workflow: PullRequestWorkflow,
    github_client: MagicMock,
    approvals: MagicMock,
    publish_arguments: dict,
) -> None:
    """Do not create remote Git objects when human approval is denied."""
    approvals.return_value = False

    with pytest.raises(PullRequestWorkflowError, match="publish_branch"):
        workflow.publish(**publish_arguments)

    repository = github_client.repository
    repository.create_git_blob.assert_not_called()
    repository.create_git_tree.assert_not_called()
    repository.create_git_commit.assert_not_called()
    repository.create_git_ref.assert_not_called()
    github_client.create_pull_request.assert_not_called()


def test_publish_stops_when_approval_callback_raises(
    workflow: PullRequestWorkflow,
    github_client: MagicMock,
    approvals: MagicMock,
    publish_arguments: dict,
) -> None:
    """An unavailable approval UI fails closed before any GitHub object is written."""
    approvals.side_effect = RuntimeError("approval UI unavailable")

    with pytest.raises(PullRequestWorkflowError, match="Approval callback failed"):
        workflow.publish(**publish_arguments)

    repository = github_client.repository
    repository.create_git_blob.assert_not_called()
    repository.create_git_tree.assert_not_called()
    repository.create_git_commit.assert_not_called()
    repository.create_git_ref.assert_not_called()
    github_client.create_pull_request.assert_not_called()


def test_publish_requires_approval_for_more_than_five_files(
    workflow: PullRequestWorkflow,
    approvals: MagicMock,
    publish_arguments: dict,
) -> None:
    """Require approval before publishing a change set containing six files."""
    publish_arguments["changes"] = {
        f"src/module_{index}.py": f"value = {index}\n" for index in range(6)
    }

    workflow.publish(**publish_arguments)

    operations = [call.args[0] for call in approvals.call_args_list]
    assert "publish_changes_over_five_files" in operations


def test_publish_stops_when_large_change_approval_is_denied(
    workflow: PullRequestWorkflow,
    github_client: MagicMock,
    approvals: MagicMock,
    publish_arguments: dict,
) -> None:
    """Stop before GitHub writes if approval for a large change set is denied."""
    approvals.return_value = False
    publish_arguments["changes"] = {
        f"src/module_{index}.py": f"value = {index}\n" for index in range(6)
    }

    with pytest.raises(
        PullRequestWorkflowError,
        match="publish_changes_over_five_files",
    ):
        workflow.publish(**publish_arguments)

    repository = github_client.repository
    repository.create_git_blob.assert_not_called()
    repository.create_git_tree.assert_not_called()
    repository.create_git_commit.assert_not_called()
    repository.create_git_ref.assert_not_called()
    github_client.create_pull_request.assert_not_called()


def test_publish_requires_approval_for_master_branch(
    workflow: PullRequestWorkflow,
    approvals: MagicMock,
    publish_arguments: dict,
) -> None:
    """Request protected-branch approval for a master target."""
    publish_arguments["base_branch"] = "master"

    workflow.publish(**publish_arguments)

    operations = [call.args[0] for call in approvals.call_args_list]
    assert "create_pull_request_against_protected_branch" in operations


def test_publish_stops_when_protected_branch_approval_is_denied(
    workflow: PullRequestWorkflow,
    github_client: MagicMock,
    approvals: MagicMock,
    publish_arguments: dict,
) -> None:
    """Do not open a PR when approval for the protected target is denied."""

    def approve_except_protected_branch(
        operation: str,
        details: dict,
    ) -> bool:
        return operation != "create_pull_request_against_protected_branch"

    approvals.side_effect = approve_except_protected_branch

    with pytest.raises(
        PullRequestWorkflowError,
        match="create_pull_request_against_protected_branch",
    ):
        workflow.publish(**publish_arguments)

    # Approval is requested before any remote Git object is created.
    repository = github_client.repository
    repository.create_git_blob.assert_not_called()
    repository.create_git_tree.assert_not_called()
    repository.create_git_commit.assert_not_called()
    repository.create_git_ref.assert_not_called()
    github_client.create_pull_request.assert_not_called()


def test_publish_rejects_invalid_issue_number(
    workflow: PullRequestWorkflow,
    publish_arguments: dict,
) -> None:
    """Reject invalid issue numbers."""
    publish_arguments["issue_number"] = 0

    with pytest.raises(ValueError, match="positive integer"):
        workflow.publish(**publish_arguments)


@pytest.mark.parametrize(
    "path",
    [
        "../outside.py",
        "/outside.py",
        "src/../outside.py",
        "src//outside.py",
        "C:/outside.py",
    ],
)
def test_publish_rejects_unsafe_paths(
    workflow: PullRequestWorkflow,
    github_client: MagicMock,
    publish_arguments: dict,
    path: str,
) -> None:
    """Reject absolute paths and traversal attempts."""
    publish_arguments["changes"] = {path: "value = 1\n"}

    with pytest.raises(ValueError, match="Invalid repository-relative path"):
        workflow.publish(**publish_arguments)

    github_client.repository.create_git_blob.assert_not_called()


def test_publish_rejects_empty_changes(
    workflow: PullRequestWorkflow,
    publish_arguments: dict,
) -> None:
    """Reject an empty change set."""
    publish_arguments["changes"] = {}

    with pytest.raises(ValueError, match="At least one file change"):
        workflow.publish(**publish_arguments)


def test_publish_wraps_github_api_failures(
    workflow: PullRequestWorkflow,
    github_client: MagicMock,
    publish_arguments: dict,
) -> None:
    """Report GitHub API failures as workflow errors."""
    github_client.repository.get_git_ref.side_effect = RuntimeError("GitHub unavailable")

    with pytest.raises(
        PullRequestWorkflowError,
        match="Could not publish the pull request",
    ):
        workflow.publish(**publish_arguments)

    github_client.create_pull_request.assert_not_called()


def test_publish_reports_no_metadata_warnings_on_success(
    workflow: PullRequestWorkflow,
    publish_arguments: dict,
) -> None:
    """A fully configured PR should have no metadata warnings."""
    result = workflow.publish(**publish_arguments)

    assert result.metadata_warnings == ()


def test_publish_reports_label_failure_without_failing_pr_creation(
    workflow: PullRequestWorkflow,
    github_client: MagicMock,
    publish_arguments: dict,
) -> None:
    """Label failure must be visible without losing the created PR."""
    github_client.create_pull_request.side_effect = None
    pull_request = github_client.create_pull_request.return_value
    pull_request.number = 42
    pull_request.html_url = "https://github.com/example-owner/example-repo/pull/42"
    pull_request.metadata_warnings = ("Failed to apply required labels.",)

    def create_pull_request_with_warning(**kwargs):
        kwargs["warning_collector"].append("Failed to apply required labels.")
        return pull_request

    github_client.create_pull_request.side_effect = create_pull_request_with_warning

    result = workflow.publish(**publish_arguments)

    assert result.pull_number == 42
    assert result.metadata_warnings == ("Failed to apply required labels.",)
    github_client.create_pull_request.assert_called_once()


def test_publish_reports_reviewer_failure_without_failing_pr_creation(
    workflow: PullRequestWorkflow,
    github_client: MagicMock,
    publish_arguments: dict,
) -> None:
    """Reviewer-request failure must be visible without losing the created PR."""
    pull_request = github_client.create_pull_request.return_value
    pull_request.number = 42
    pull_request.html_url = "https://github.com/example-owner/example-repo/pull/42"
    pull_request.metadata_warnings = ("Failed to request reviewer 'reporter'.",)

    def create_pull_request_with_warning(**kwargs):
        kwargs["warning_collector"].append("Failed to request reviewer 'reporter'.")

        return pull_request

    github_client.create_pull_request.side_effect = create_pull_request_with_warning

    result = workflow.publish(**publish_arguments)

    assert result.pull_number == 42
    assert result.metadata_warnings == ("Failed to request reviewer 'reporter'.",)
    github_client.create_pull_request.assert_called_once()


def test_publish_preserves_multiple_metadata_warnings(
    workflow: PullRequestWorkflow,
    github_client: MagicMock,
    publish_arguments: dict,
) -> None:
    """Multiple metadata failures should be reported together."""
    pull_request = github_client.create_pull_request.return_value
    pull_request.number = 42
    pull_request.html_url = "https://github.com/example-owner/example-repo/pull/42"
    pull_request.metadata_warnings = (
        "Failed to apply required labels.",
        "Failed to request reviewer 'reporter'.",
    )

    def create_pull_request_with_warnings(**kwargs):
        kwargs["warning_collector"].extend(
            [
                "Failed to apply required labels.",
                "Failed to request reviewer 'reporter'.",
            ]
        )
        return pull_request

    github_client.create_pull_request.side_effect = create_pull_request_with_warnings

    result = workflow.publish(**publish_arguments)

    assert result.metadata_warnings == (
        "Failed to apply required labels.",
        "Failed to request reviewer 'reporter'.",
    )
    github_client.create_pull_request.assert_called_once()
