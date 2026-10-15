"""Poll GitHub issues and optionally check registered pull requests for merges."""
from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Event
from typing import Any

from codepilot.config.settings import Settings
from codepilot.github.issue_selector import select_eligible_issues
from codepilot.github.models import GitHubIssue

logger = logging.getLogger(__name__)


class GitHubPollingService:
    """Poll eligible issues and, when configured, process merge-monitor updates."""

    def __init__(
        self,
        client: Any,
        settings: Settings,
        *,
        merge_monitor: Any | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self._merge_monitor = merge_monitor
        self._in_progress_issue_numbers: set[int] = set()

    @property
    def in_progress_issue_numbers(self) -> frozenset[int]:
        """Return the issue numbers currently claimed for processing."""
        return frozenset(self._in_progress_issue_numbers)

    def poll_once(self) -> list[GitHubIssue]:
        """Check pending PR merges, then return newly eligible GitHub issues."""
        if self._merge_monitor is not None:
            try:
                outcomes = self._merge_monitor.poll_once()
                logger.debug("Pull-request merge-monitor outcomes: %s", outcomes)
            except Exception:
                logger.exception("Pull-request merge-monitor polling failed")
        issues = self._client.get_open_issues()
        eligible = select_eligible_issues(
            issues,
            complexity_threshold=self._settings.complexity_threshold,
            in_progress_issue_numbers=self._in_progress_issue_numbers,
        )
        for issue in eligible:
            self._in_progress_issue_numbers.add(issue.number)
        return eligible

    def release_issue(self, issue_number: int) -> None:
        """Release an issue claim so it can be selected again."""
        self._in_progress_issue_numbers.discard(issue_number)

    def run_forever(
        self,
        on_issue: Callable[[GitHubIssue], None],
        stop_event: Event,
    ) -> None:
        """Poll at the configured interval until the stop event is set."""
        while not stop_event.is_set():
            try:
                eligible_issues = self.poll_once()
                for issue in eligible_issues:
                    if stop_event.is_set():
                        self.release_issue(issue.number)
                        break
                    try:
                        on_issue(issue)
                    except KeyboardInterrupt:
                        self.release_issue(issue.number)
                        stop_event.set()
                        raise
                    except Exception:
                        logger.exception("Could not schedule issue #%s", issue.number)
                        self.release_issue(issue.number)
            except KeyboardInterrupt:
                stop_event.set()
                break
            except Exception:
                logger.exception("GitHub polling iteration failed")
            if stop_event.wait(self._settings.poll_interval_minutes * 60):
                break
