"""Poll GitHub issues and optionally check registered pull requests for merges."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from threading import Event, Lock
from typing import Any

from codepilot.config.settings import Settings
from codepilot.github.issue_selector import select_eligible_issues
from codepilot.github.models import GitHubIssue

logger = logging.getLogger(__name__)

IssuesObserver = Callable[[Sequence[GitHubIssue], frozenset[int]], None]


class GitHubPollingService:
    """Poll eligible issues and, when configured, process merge-monitor updates."""

    def __init__(
        self,
        client: Any,
        settings: Settings,
        *,
        merge_monitor: Any | None = None,
        on_issues: IssuesObserver | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self._merge_monitor = merge_monitor
        self._on_issues = on_issues
        self._in_progress_issue_numbers: set[int] = set()
        self._last_open_issues: list[GitHubIssue] = []
        self._state_lock = Lock()

    @property
    def in_progress_issue_numbers(self) -> frozenset[int]:
        """Return the issue numbers currently claimed for processing."""
        with self._state_lock:
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
        get_open_pr_issue_numbers = getattr(
            self._client, "get_open_codepilot_pr_issue_numbers", None
        )
        if callable(get_open_pr_issue_numbers):
            open_pr_issue_numbers = set(get_open_pr_issue_numbers())
        else:
            open_pr_issue_numbers = set()

        issues_without_open_pr = [
            issue for issue in issues if issue.number not in open_pr_issue_numbers
        ]
        for issue in issues:
            if issue.number in open_pr_issue_numbers:
                logger.info(
                    "Skipping issue #%s because an open CodePilot pull request already exists.",
                    issue.number,
                )

        with self._state_lock:
            # Keep the full GitHub snapshot for the TUI; filter only the work queue.
            self._last_open_issues = list(issues)
            eligible = select_eligible_issues(
                issues_without_open_pr,
                complexity_threshold=self._settings.complexity_threshold,
                in_progress_issue_numbers=self._in_progress_issue_numbers,
            )
            for issue in eligible:
                self._in_progress_issue_numbers.add(issue.number)
        self._notify_issues_observer()
        return eligible

    def release_issue(self, issue_number: int) -> None:
        """Release an issue claim so it can be selected again."""
        with self._state_lock:
            self._in_progress_issue_numbers.discard(issue_number)
        self._notify_issues_observer()

    def _notify_issues_observer(self) -> None:
        """Publish a read-only snapshot for consumers such as the Textual TUI."""
        if self._on_issues is None:
            return
        with self._state_lock:
            issues = tuple(self._last_open_issues)
            in_progress_issue_numbers = frozenset(self._in_progress_issue_numbers)
        try:
            self._on_issues(issues, in_progress_issue_numbers)
        except Exception:
            # A presentation-layer failure must not break GitHub polling.
            logger.exception("Could not publish the GitHub issue snapshot")

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
