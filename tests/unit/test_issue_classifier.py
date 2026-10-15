"""Tests for deterministic issue classification."""

import pytest

from codepilot.agents.classifier import IssueCategory, classify_issue
from codepilot.github.models import GitHubIssue


@pytest.mark.parametrize(
    ("title", "body", "expected"),
    [
        (
            "Fix login failure",
            "Users encounter a crash after login.",
            IssueCategory.BUG_FIX,
        ),
        (
            "Add export support",
            "Implement a new capability for CSV export.",
            IssueCategory.FEATURE_ADDITION,
        ),
        (
            "Dependency update",
            "Upgrade dependency to the supported version.",
            IssueCategory.DEPENDENCY_UPDATE,
        ),
        (
            "Update README",
            "Document the installation process.",
            IssueCategory.DOCUMENTATION,
        ),
        (
            "Update configuration",
            "Add an environment variable for the service.",
            IssueCategory.CONFIG_CHANGE,
        ),
    ],
)
def test_classifies_issue_by_content(title, body, expected):
    issue = GitHubIssue(number=1, title=title, body=body)

    assert classify_issue(issue) == expected


def test_explicit_category_label_takes_precedence():
    issue = GitHubIssue(
        number=2,
        title="Fix crash",
        labels=("documentation",),
    )

    assert classify_issue(issue) == IssueCategory.DOCUMENTATION


def test_category_labels_accept_hyphens_and_spaces():
    hyphenated = GitHubIssue(
        number=3,
        title="Example",
        labels=("bug-fix",),
    )
    spaced = GitHubIssue(
        number=4,
        title="Example",
        labels=("dependency update",),
    )

    assert classify_issue(hyphenated) == IssueCategory.BUG_FIX
    assert classify_issue(spaced) == IssueCategory.DEPENDENCY_UPDATE


def test_ambiguous_issue_defaults_to_feature_addition():
    issue = GitHubIssue(
        number=5,
        title="Review the current workflow",
        body="Please investigate the process.",
    )

    assert classify_issue(issue) == IssueCategory.FEATURE_ADDITION
