"""Regression coverage for duplicate issue processing with an open CodePilot PR."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from codepilot.config.settings import Settings
from codepilot.github.client import GitHubClient
from codepilot.github.models import GitHubIssue
from codepilot.github.polling import GitHubPollingService


def _settings() -> Settings:
    return Settings(
        llm_provider=None,
        llm_model=None,
        openai_api_key=None,
        anthropic_api_key=None,
        google_api_key=None,
        github_token=None,
        github_repository_owner="example-owner",
        github_repository_name="example-repo",
        github_default_branch="main",
        poll_interval_minutes=5,
        complexity_threshold=7,
        repo_map_token_budget=4000,
        repo_map_top_k=10,
        max_test_retries=3,
        state_dir=".codepilot",
        chroma_dir=".codepilot/chroma",
    )


def _pull_request(title: str, branch: str) -> SimpleNamespace:
    return SimpleNamespace(title=title, head=SimpleNamespace(ref=branch))


def _client_for_repository(repository: MagicMock) -> GitHubClient:
    client = GitHubClient.__new__(GitHubClient)
    client._api_wrapper = SimpleNamespace(github_repo_instance=repository)
    client._toolkit = MagicMock()
    return client


def test_client_finds_issue_numbers_from_open_codepilot_pr_branches() -> None:
    repository = MagicMock()
    repository.get_pulls.return_value = [
        _pull_request(
            "[CodePilot] Fix calculate_total to multiply price by quantity",
            "codepilot/issue-1-fix-calculate-total",
        ),
        _pull_request(
            "[CodePilot] Fix another issue",
            "codepilot/issue-12-fix-another-issue",
        ),
        _pull_request("Unrelated PR", "feature/issue-4"),
        _pull_request("[CodePilot] Unrelated branch", "feature/cleanup"),
    ]
    client = _client_for_repository(repository)

    assert client.get_open_codepilot_pr_issue_numbers() == {1, 12}
    repository.get_pulls.assert_called_once_with(state="open")


def test_polling_skips_only_issues_with_an_existing_open_codepilot_pr() -> None:
    issue_with_pr = GitHubIssue(number=1, title="Fix calculate_total")
    issue_without_pr = GitHubIssue(number=2, title="Fix a typo")
    client = MagicMock()
    client.get_open_issues.return_value = [issue_with_pr, issue_without_pr]
    client.get_open_codepilot_pr_issue_numbers.return_value = {1}
    service = GitHubPollingService(client, _settings())

    assert service.poll_once() == [issue_without_pr]
    assert service.in_progress_issue_numbers == frozenset({2})
    client.get_open_issues.assert_called_once_with()
    client.get_open_codepilot_pr_issue_numbers.assert_called_once_with()
