"""Unit tests for CodePilot issue selection."""

import pytest

from codepilot.github.issue_selector import select_eligible_issues
from codepilot.github.models import GitHubIssue


def make_issue(
    number: int,
    *,
    title: str = "Fix a typo",
    body: str = "",
    labels: tuple[str, ...] = (),
    assignee: str | None = None,
) -> GitHubIssue:
    """Create a test issue."""
    return GitHubIssue(
        number=number,
        title=title,
        body=body,
        labels=labels,
        assignee=assignee,
    )


def test_selects_issue_with_ai_assignable_label_regardless_of_complexity() -> None:
    issue = make_issue(
        1,
        title="Redesign architecture and database migration",
        labels=("ai-assignable",),
    )

    result = select_eligible_issues(
        [issue],
        complexity_threshold=7,
    )

    assert result == [issue]


def test_selects_unassigned_issue_below_complexity_threshold() -> None:
    issue = make_issue(2, title="Fix a typo")

    result = select_eligible_issues(
        [issue],
        complexity_threshold=7,
    )

    assert result == [issue]


def test_excludes_issue_at_complexity_threshold() -> None:
    issue = make_issue(
        3,
        title="Refactor architecture and redesign database migration",
        body=(
            "Improve authentication, authorization, security, concurrency, "
            "performance, integration, API, multiple modules, backward "
            "compatibility, breaking change, edge cases and test coverage."
        ),
    )

    result = select_eligible_issues(
        [issue],
        complexity_threshold=10,
    )

    assert result == []


def test_excludes_assigned_issue_without_ai_assignable_label() -> None:
    issue = make_issue(4, assignee="developer")

    result = select_eligible_issues(
        [issue],
        complexity_threshold=7,
    )

    assert result == []


def test_excludes_issue_already_in_progress() -> None:
    issue = make_issue(5, labels=("ai-assignable",))

    result = select_eligible_issues(
        [issue],
        complexity_threshold=7,
        in_progress_issue_numbers={5},
    )

    assert result == []


def test_preserves_input_order() -> None:
    issues = [
        make_issue(8),
        make_issue(7),
        make_issue(9),
    ]

    result = select_eligible_issues(
        issues,
        complexity_threshold=7,
    )

    assert [issue.number for issue in result] == [8, 7, 9]


@pytest.mark.parametrize("threshold", [0, 11, -1])
def test_rejects_invalid_complexity_threshold(threshold: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 10"):
        select_eligible_issues(
            [],
            complexity_threshold=threshold,
        )
