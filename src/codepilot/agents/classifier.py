"""Classify GitHub issues into CodePilot task categories."""

from enum import StrEnum

from codepilot.github.models import GitHubIssue


class IssueCategory(StrEnum):
    """Supported categories of GitHub issues."""

    BUG_FIX = "bug_fix"
    FEATURE_ADDITION = "feature_addition"
    DEPENDENCY_UPDATE = "dependency_update"
    DOCUMENTATION = "documentation"
    CONFIG_CHANGE = "config_change"


def classify_issue(issue: GitHubIssue) -> IssueCategory:
    """Classify an issue using its labels, title, and body.

    Explicit category labels take precedence over keyword-based inference.
    When the issue is ambiguous, default to feature addition.
    """
    label_categories = {category.value: category for category in IssueCategory}

    for label in issue.labels:
        normalized_label = label.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized_label in label_categories:
            return label_categories[normalized_label]

    text = f"{issue.title}\n{issue.body}".lower()

    rules = [
        (
            IssueCategory.DEPENDENCY_UPDATE,
            (
                "dependency update",
                "dependencies update",
                "upgrade dependency",
                "update dependency",
                "bump version",
                "upgrade package",
                "update package",
            ),
        ),
        (
            IssueCategory.DOCUMENTATION,
            (
                "documentation",
                "readme",
                "docs:",
                "document this",
                "update docs",
                "typo",
            ),
        ),
        (
            IssueCategory.CONFIG_CHANGE,
            (
                "configuration",
                "config file",
                "environment variable",
                ".env.example",
                "settings file",
            ),
        ),
        (
            IssueCategory.BUG_FIX,
            (
                "bug",
                "defect",
                "does not work",
                "doesn't work",
                "unexpected error",
                "crash",
                "fix:",
                "regression",
            ),
        ),
        (
            IssueCategory.FEATURE_ADDITION,
            (
                "feature",
                "enhancement",
                "add support",
                "implement",
                "new capability",
                "new functionality",
            ),
        ),
    ]

    for category, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return category

    return IssueCategory.FEATURE_ADDITION
