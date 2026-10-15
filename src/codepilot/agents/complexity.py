"""Estimate GitHub issue complexity using transparent heuristic rules."""

from codepilot.github.models import GitHubIssue

MIN_COMPLEXITY = 1
MAX_COMPLEXITY = 10

# Each matching signal adds to the initial score.
COMPLEXITY_SIGNALS: tuple[tuple[str, int], ...] = (
    ("refactor", 2),
    ("architecture", 3),
    ("redesign", 3),
    ("migration", 2),
    ("database", 2),
    ("authentication", 2),
    ("authorization", 2),
    ("security", 2),
    ("concurrency", 3),
    ("performance", 2),
    ("integration", 2),
    ("api", 1),
    ("multiple modules", 2),
    ("backward compatibility", 2),
    ("breaking change", 3),
    ("edge cases", 1),
    ("test coverage", 1),
)


def estimate_complexity(issue: GitHubIssue) -> int:
    """Estimate issue complexity on a scale from 1 to 10.

    The estimate starts at 2 and increases for recognized complexity
    signals in the issue title and body. The final score is capped at 10.

    This is a heuristic, not an objective measure of engineering effort.
    """
    text = f"{issue.title}\n{issue.body}".lower()
    score = 2

    for signal, weight in COMPLEXITY_SIGNALS:
        if signal in text:
            score += weight

    return min(MAX_COMPLEXITY, max(MIN_COMPLEXITY, score))
