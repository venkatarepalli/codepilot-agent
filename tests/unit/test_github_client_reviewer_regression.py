"""Regression tests for best-effort PR labels and reviewer assignment."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from codepilot.github.client import GitHubClient


def make_client(*, pull_request_author: str = "venkat"):
    """Create a GitHub client with mocked repository and pull request objects."""
    api_wrapper = MagicMock()
    toolkit = MagicMock()
    pull_request = MagicMock()
    pull_request.number = 2
    pull_request.user = SimpleNamespace(login=pull_request_author)
    api_wrapper.github_repo_instance.create_pull.return_value = pull_request

    client = GitHubClient(
        SimpleNamespace(),
        api_wrapper=api_wrapper,
        toolkit=toolkit,
    )
    return client, api_wrapper.github_repo_instance, pull_request


def create_pr(client: GitHubClient, *, reviewer: str | None, labels: tuple[str, ...] = ()):
    """Call the PR client with stable, valid fields for regression tests."""
    return client.create_pull_request(
        title="[CodePilot] Fix total",
        body="Correct the total calculation.",
        head="codepilot/issue-1-fix-total",
        base="main",
        labels=labels,
        reviewer=reviewer,
    )


def test_create_pull_request_does_not_request_review_from_pr_author() -> None:
    """The PR author must never be requested to review their own PR."""
    client, repository, pull_request = make_client(pull_request_author="venkat")

    result = create_pr(client, reviewer="venkat")

    assert result is pull_request
    repository.create_pull.assert_called_once()
    pull_request.create_review_request.assert_not_called()


def test_create_pull_request_requests_a_different_reviewer() -> None:
    """A non-author reviewer is still requested normally."""
    client, _repository, pull_request = make_client(pull_request_author="venkat")

    result = create_pr(client, reviewer="reviewer-two")

    assert result is pull_request
    pull_request.create_review_request.assert_called_once_with(reviewers=["reviewer-two"])


def test_reviewer_request_failure_does_not_hide_created_pr(caplog) -> None:
    """An optional reviewer API failure must not turn an existing PR into failure."""
    client, repository, pull_request = make_client(pull_request_author="venkat")
    pull_request.create_review_request.side_effect = RuntimeError("review request rejected")

    with caplog.at_level(logging.WARNING, logger="codepilot.github.client"):
        result = create_pr(client, reviewer="reviewer-two")

    assert repository.create_pull.call_count == 1
    assert result is pull_request
    assert "Pull request #2 was created" in caplog.text
    assert "reviewer-two" in caplog.text
    assert "review request rejected" in caplog.text


def test_label_failure_does_not_hide_created_pr(caplog) -> None:
    """An optional label API failure must not erase successful PR creation."""
    client, repository, pull_request = make_client(pull_request_author="venkat")
    pull_request.add_to_labels.side_effect = RuntimeError("label update rejected")

    with caplog.at_level(logging.WARNING, logger="codepilot.github.client"):
        result = create_pr(
            client,
            reviewer=None,
            labels=("codepilot-generated", "needs-review"),
        )

    assert repository.create_pull.call_count == 1
    assert result is pull_request
    assert "Pull request #2 was created" in caplog.text
    assert "label" in caplog.text.lower()
    assert "label update rejected" in caplog.text
