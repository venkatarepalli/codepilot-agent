"""PR title prefix and structured commit message are enforced by the PR workflow."""

from unittest.mock import MagicMock

import pytest

from codepilot.github.pr_workflow import (
    PullRequestWorkflow,
    build_structured_commit_message,
    enforce_structured_commit_message,
    format_pull_request_title,
    is_structured_commit_message,
)

COMPLIANT = (
    "fix(#7): handle empty input\n\n"
    "- Change: return 0 for empty list\n"
    "- Why: sum() crashed on []\n"
    "- Closes #7"
)


@pytest.mark.parametrize(
    "raw",
    [
        "Fix bug",
        "[CodePilot] Fix bug",
        "[codepilot]   Fix   bug",
        "[CodePilot] [CodePilot] Fix bug",
    ],
)
def test_title_has_prefix_exactly_once(raw: str) -> None:
    assert format_pull_request_title(raw) == "[CodePilot] Fix bug"


def test_title_rejects_prefix_only() -> None:
    with pytest.raises(ValueError):
        format_pull_request_title("[CodePilot]  ")


def test_compliant_message_is_recognised_and_left_unchanged() -> None:
    assert is_structured_commit_message(COMPLIANT, 7)
    assert (
        enforce_structured_commit_message(
            COMPLIANT, issue_number=7, issue_title="t", changed_files=["a.py"]
        )
        == COMPLIANT
    )


@pytest.mark.parametrize(
    "bad",
    [
        "fix: handle empty input",  # no issue ref, no body
        COMPLIANT.replace("#7)", "#8)"),  # wrong issue number
        COMPLIANT.replace("- Closes #7", "Closes #7"),  # closes not a bullet
        COMPLIANT.replace("\n\n", "\n", 1),  # no blank line
        COMPLIANT.replace("handle empty input", "x" * 80),  # header too long
    ],
)
def test_non_compliant_messages_are_rebuilt(bad: str) -> None:
    assert not is_structured_commit_message(bad, 7)
    fixed = enforce_structured_commit_message(
        bad, issue_number=7, issue_title="Empty input crash", changed_files=["calc.py"]
    )
    assert is_structured_commit_message(fixed, 7), fixed


def test_builder_output_is_compliant_and_header_bounded() -> None:
    message = build_structured_commit_message(
        issue_number=123, summary="y" * 200, what_changed="w", why=""
    )
    assert is_structured_commit_message(message, 123)
    assert len(message.splitlines()[0]) <= 72


def test_publish_commits_structured_message_and_prefixed_title() -> None:
    client = MagicMock()
    repository = client.repository
    repository.get_git_ref.return_value.object.sha = "base"
    repository.create_git_blob.return_value.sha = "blob"
    repository.create_git_commit.return_value.sha = "c"
    client.create_pull_request.return_value = MagicMock(number=5, html_url="u/5")
    approvals = []

    def approve(operation, details):
        approvals.append((operation, dict(details)))
        return True

    workflow = PullRequestWorkflow(
        github_client=client, default_branch="main", approval_callback=approve
    )
    result = workflow.publish(
        issue_number=9,
        issue_title="[CodePilot] Fix README typo",
        issue_url="https://x/issues/9",
        changes={"README.md": "hi\n"},
        commit_message="Fixed the typo lol",
        summary="s",
        test_results="ok",
    )

    committed = repository.create_git_commit.call_args.args[0]
    assert is_structured_commit_message(committed, 9), committed
    assert result.title == "[CodePilot] Fix README typo"
    assert client.create_pull_request.call_args.kwargs["title"] == result.title
    # The human sees the exact title and commit message before approving.
    assert approvals[0][1]["pull_request_title"] == result.title
    assert approvals[0][1]["commit_message"] == committed
