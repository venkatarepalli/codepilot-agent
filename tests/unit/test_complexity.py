"""Unit tests for CodePilot complexity estimation."""

import pytest

from codepilot.agents.complexity import estimate_complexity
from codepilot.github.models import GitHubIssue


def make_issue(title: str, body: str = "") -> GitHubIssue:
    """Create a GitHub issue for complexity tests."""
    return GitHubIssue(
        number=1,
        title=title,
        body=body,
    )


def test_simple_issue_receives_base_score() -> None:
    issue = make_issue("Fix a typo")

    assert estimate_complexity(issue) == 2


def test_matching_complexity_signals_increase_score() -> None:
    issue = make_issue("Improve database migration")

    # Base 2 + database 2 + migration 2.
    assert estimate_complexity(issue) == 6


def test_title_and_body_are_both_considered() -> None:
    issue = make_issue(
        "Update login",
        "Implement authentication and authorization.",
    )

    # Base 2 + authentication 2 + authorization 2.
    assert estimate_complexity(issue) == 6


def test_matching_is_case_insensitive() -> None:
    issue = make_issue("Improve DATABASE performance")

    # Base 2 + database 2 + performance 2.
    assert estimate_complexity(issue) == 6


def test_score_never_exceeds_ten() -> None:
    issue = make_issue(
        "Refactor architecture and redesign database migration",
        (
            "Improve authentication, authorization, security, concurrency, "
            "performance, integration, API, multiple modules, backward "
            "compatibility, breaking change, edge cases and test coverage."
        ),
    )

    assert estimate_complexity(issue) == 10


@pytest.mark.parametrize(
    ("title", "body"),
    [
        ("Fix a typo", ""),
        ("Improve database performance", ""),
        ("Redesign the architecture", "Consider backward compatibility."),
    ],
)
def test_estimation_is_deterministic(title: str, body: str) -> None:
    issue = make_issue(title, body)

    assert estimate_complexity(issue) == estimate_complexity(issue)
