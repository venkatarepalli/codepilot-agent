"""Select GitHub issues that CodePilot is eligible to process."""

from collections.abc import Collection, Iterable

from codepilot.agents.complexity import estimate_complexity
from codepilot.github.models import GitHubIssue

ASSIGNABLE_LABEL = "ai-assignable"


def select_eligible_issues(
    issues: Iterable[GitHubIssue],
    *,
    complexity_threshold: int,
    in_progress_issue_numbers: Collection[int] = (),
) -> list[GitHubIssue]:
    """Select eligible issues using labels, complexity, and task status.

    Rules:
    - Issues already being processed are excluded.
    - Issues labelled 'ai-assignable' are eligible regardless of complexity.
    - Other issues must be unassigned and have an estimated complexity
      strictly below the configured threshold.
    - Pull requests are excluded by GitHubClient before reaching this function.

    The input order is preserved.
    """
    if not 1 <= complexity_threshold <= 10:
        raise ValueError("Complexity threshold must be between 1 and 10.")

    in_progress = set(in_progress_issue_numbers)
    eligible: list[GitHubIssue] = []

    for issue in issues:
        if issue.number in in_progress:
            continue

        labels = {label.strip().lower() for label in issue.labels}

        if ASSIGNABLE_LABEL in labels:
            eligible.append(issue)
            continue

        if issue.assignee is not None:
            continue

        score = estimate_complexity(issue)

        if score < complexity_threshold:
            eligible.append(issue)

    return eligible
