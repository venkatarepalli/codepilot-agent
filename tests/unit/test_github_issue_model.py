"""Tests for the normalized GitHub issue model."""

import pytest

from codepilot.github.models import GitHubIssue


def test_issue_can_be_created():
    issue = GitHubIssue(
        number=12,
        title="Fix login failure",
        body="Login fails after a session expires.",
        labels=("bug", "backend"),
        reporter="contributor",
    )

    assert issue.number == 12
    assert issue.title == "Fix login failure"
    assert issue.reporter == "contributor"
    assert issue.labels == ("bug", "backend")


def test_issue_rejects_invalid_number():
    with pytest.raises(ValueError, match="Issue number"):
        GitHubIssue(number=0, title="Invalid issue")


def test_issue_rejects_empty_title():
    with pytest.raises(ValueError, match="title"):
        GitHubIssue(number=1, title="   ")


def test_issue_normalizes_labels():
    issue = GitHubIssue(
        number=1,
        title="Example",
        labels=(" bug ", "", "backend"),
    )

    assert issue.labels == ("bug", "backend")


def test_issue_is_immutable():
    issue = GitHubIssue(number=1, title="Example")

    with pytest.raises(AttributeError):
        issue.title = "Changed title"
