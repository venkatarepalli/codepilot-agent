"""Data models for GitHub issues handled by CodePilot."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class GitHubIssue:
    """A normalized representation of a GitHub issue."""

    number: int
    title: str
    body: str = ""
    url: str = ""
    labels: tuple[str, ...] = field(default_factory=tuple)
    reporter: str | None = None
    assignee: str | None = None

    def __post_init__(self) -> None:
        """Validate required issue fields."""
        if self.number <= 0:
            raise ValueError("Issue number must be greater than zero.")

        if not self.title.strip():
            raise ValueError("Issue title must not be empty.")

        object.__setattr__(
            self,
            "labels",
            tuple(label.strip() for label in self.labels if label.strip()),
        )
