"""Data models for sandbox security decisions."""

from dataclasses import dataclass
from enum import StrEnum


class SecurityDecision(StrEnum):
    """Possible outcomes of a sandbox security check."""

    ALLOW = "allow"
    BLOCK = "block"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass(frozen=True, slots=True)
class SecurityCheckResult:
    """Result of validating a proposed sandbox operation."""

    decision: SecurityDecision
    reason: str

    def __post_init__(self) -> None:
        """Reject incomplete security decisions."""
        if not self.reason.strip():
            raise ValueError("A security decision must include a reason.")

    @property
    def allowed(self) -> bool:
        """Return whether the operation may proceed automatically."""
        return self.decision == SecurityDecision.ALLOW

    @property
    def requires_human_approval(self) -> bool:
        """Return whether human approval is required."""
        return self.decision == SecurityDecision.REQUIRES_APPROVAL
