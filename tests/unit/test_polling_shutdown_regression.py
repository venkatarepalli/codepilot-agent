"""Regression tests for graceful polling shutdown."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

from codepilot.github.polling import GitHubPollingService


def test_polling_stops_after_keyboard_interrupt_from_issue_handler():
    issue = SimpleNamespace(number=7, title="Test issue")
    client = Mock()
    settings = SimpleNamespace(complexity_threshold=10, poll_interval_minutes=5)
    service = GitHubPollingService(client, settings)
    stop = Event()
    service.poll_once = Mock(return_value=[issue])

    def interrupt(_issue):
        raise KeyboardInterrupt

    service.run_forever(interrupt, stop)
    assert stop.is_set()
    assert service.in_progress_issue_numbers == frozenset()


def test_polling_does_not_poll_when_stop_event_is_already_set():
    client = Mock()
    settings = SimpleNamespace(complexity_threshold=10, poll_interval_minutes=5)
    service = GitHubPollingService(client, settings)
    stop = Event()
    stop.set()

    service.run_forever(Mock(), stop)
    client.get_open_issues.assert_not_called()
