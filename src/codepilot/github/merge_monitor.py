"""Poll registered CodePilot pull requests and record lessons after merge."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import RLock
from typing import Any

from codepilot.agents.coder_models import CoderResult
from codepilot.github.models import GitHubIssue
from codepilot.github.pr_workflow import PullRequestResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PendingPullRequest:
    """Task data needed to record a lesson once its PR is merged."""

    issue: GitHubIssue
    coder_result: CoderResult
    pull_request: PullRequestResult
    summary: str


class PullRequestMergeMonitor:
    """Check tracked PRs and record semantic lessons only after confirmed merge.

    The monitor is deliberately dependency-injected. Register a PR after it is
    published, then call ``poll_once`` from the application's polling loop.
    GitHub/API errors leave the PR registered for a later retry.
    """

    def __init__(self, *, orchestrator: Any) -> None:
        if not callable(getattr(orchestrator, "record_merged_pr_lesson", None)):
            raise TypeError("orchestrator must provide record_merged_pr_lesson().")
        if not callable(getattr(orchestrator, "_pr_workflow", None)) and not hasattr(
            orchestrator, "_pr_workflow"
        ):
            raise TypeError("orchestrator must have a configured PullRequestWorkflow.")
        self._orchestrator = orchestrator
        self._pending: dict[int, PendingPullRequest] = {}
        self._lock = RLock()

    @property
    def pending_pull_numbers(self) -> frozenset[int]:
        """Return the pull request numbers currently being monitored."""
        with self._lock:
            return frozenset(self._pending)

    def register(
        self,
        *,
        issue: GitHubIssue,
        coder_result: CoderResult,
        pull_request: PullRequestResult,
        summary: str,
    ) -> None:
        """Register a published PR; registering it twice replaces the same entry."""
        if pull_request.pull_number <= 0:
            raise ValueError("pull_request.pull_number must be positive.")
        if not summary.strip():
            raise ValueError("summary must not be empty.")
        if issue.number != coder_result.issue_number:
            raise ValueError("CoderResult issue number does not match the supplied issue.")
        with self._lock:
            self._pending[pull_request.pull_number] = PendingPullRequest(
                issue=issue,
                coder_result=coder_result,
                pull_request=pull_request,
                summary=summary.strip(),
            )

    def poll_once(self) -> dict[int, str]:
        """Check each pending PR once; return per-PR outcomes for observability."""
        with self._lock:
            pending = list(self._pending.items())

        outcomes: dict[int, str] = {}
        workflow = self._orchestrator._pr_workflow
        for pull_number, task in pending:
            try:
                merged = workflow.is_merged(pull_number)
            except Exception:
                logger.exception("Could not check merge status for pull request #%s", pull_number)
                outcomes[pull_number] = "retry"
                continue

            if not merged:
                outcomes[pull_number] = "pending"
                continue

            try:
                recorded = self._orchestrator.record_merged_pr_lesson(
                    task.issue,
                    task.coder_result,
                    task.pull_request,
                    summary=task.summary,
                )
            except Exception:
                logger.exception("Could not record lesson for merged pull request #%s", pull_number)
                outcomes[pull_number] = "retry"
                continue

            # A False result means semantic memory was not configured. Keep the
            # item registered so it can be retried if configuration changes.
            if not recorded:
                outcomes[pull_number] = "not-recorded"
                continue

            with self._lock:
                self._pending.pop(pull_number, None)
            outcomes[pull_number] = "recorded"

        return outcomes
