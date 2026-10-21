"""Unit tests for the GitHub polling service."""

from unittest.mock import MagicMock

import pytest

from codepilot.config.settings import Settings
from codepilot.github.models import GitHubIssue
from codepilot.github.polling import GitHubPollingService


@pytest.fixture
def settings() -> Settings:
    """Return test settings without external credentials."""
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


@pytest.fixture
def issue() -> GitHubIssue:
    """Return a simple issue eligible for processing."""
    return GitHubIssue(
        number=42,
        title="Fix a typo",
    )


@pytest.fixture
def service(
    settings: Settings,
    issue: GitHubIssue,
) -> GitHubPollingService:
    """Return a polling service backed by a mocked GitHub client."""
    client = MagicMock()
    client.get_open_issues.return_value = [issue]

    return GitHubPollingService(client, settings)


def test_poll_once_returns_eligible_issues(
    service: GitHubPollingService,
    issue: GitHubIssue,
) -> None:
    assert service.poll_once() == [issue]


def test_poll_once_claims_returned_issues(
    service: GitHubPollingService,
) -> None:
    service.poll_once()

    assert service.in_progress_issue_numbers == frozenset({42})


def test_repeated_poll_does_not_return_same_issue(
    service: GitHubPollingService,
) -> None:
    assert len(service.poll_once()) == 1
    assert service.poll_once() == []


def test_released_issue_can_be_selected_again(
    service: GitHubPollingService,
    issue: GitHubIssue,
) -> None:
    service.poll_once()
    service.release_issue(issue.number)

    assert service.poll_once() == [issue]


def test_release_unknown_issue_is_safe(
    service: GitHubPollingService,
) -> None:
    service.release_issue(999)

    assert service.in_progress_issue_numbers == frozenset()


def test_poll_once_uses_github_client(
    service: GitHubPollingService,
) -> None:
    service.poll_once()

    service._client.get_open_issues.assert_called_once_with()


def test_poll_once_publishes_full_issue_snapshot_and_claimed_numbers(
    settings: Settings,
    issue: GitHubIssue,
) -> None:
    client = MagicMock()
    client.get_open_issues.return_value = [issue]
    observer = MagicMock()
    service = GitHubPollingService(client, settings, on_issues=observer)

    assert service.poll_once() == [issue]

    observer.assert_called_once_with((issue,), frozenset({issue.number}))


def test_releasing_issue_refreshes_observer_snapshot(
    settings: Settings,
    issue: GitHubIssue,
) -> None:
    client = MagicMock()
    client.get_open_issues.return_value = [issue]
    observer = MagicMock()
    service = GitHubPollingService(client, settings, on_issues=observer)

    service.poll_once()
    service.release_issue(issue.number)

    assert observer.call_count == 2
    assert observer.call_args.args == ((issue,), frozenset())


def test_run_forever_stops_when_stop_event_is_set(
    service: GitHubPollingService,
) -> None:
    from threading import Event

    stop_event = Event()
    stop_event.set()
    callback = MagicMock()

    service.run_forever(callback, stop_event)

    callback.assert_not_called()


def test_scheduling_failure_releases_issue(
    service: GitHubPollingService,
    issue: GitHubIssue,
) -> None:
    from threading import Event

    stop_event = Event()
    stop_event.set()

    def fail_to_schedule(_issue: GitHubIssue) -> None:
        raise RuntimeError("Queue unavailable")

    # Allow the first polling iteration, then stop before the next one.
    service._settings = MagicMock(
        poll_interval_minutes=0,
        complexity_threshold=7,
    )

    # Setting the event in the callback ends the loop after this iteration.
    def callback(_issue: GitHubIssue) -> None:
        fail_to_schedule(_issue)
        stop_event.set()

    service.run_forever(callback, stop_event)

    assert issue.number not in service.in_progress_issue_numbers
