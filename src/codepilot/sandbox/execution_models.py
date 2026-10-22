"""Structured models for sandbox command execution."""

from dataclasses import dataclass
from enum import StrEnum


class CommandStatus(StrEnum):
    """Possible outcomes of a sandbox command."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    REQUIRES_APPROVAL = "requires_approval"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class CommandExecutionResult:
    """Immutable result of executing or rejecting a command."""

    status: CommandStatus
    command: tuple[str, ...]
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate execution-result consistency."""
        if not isinstance(self.command, tuple):
            raise TypeError("command must be a tuple of strings.")

        if not all(isinstance(arg, str) for arg in self.command):
            raise TypeError("Every command argument must be a string.")

        if self.duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative.")

        if self.status == CommandStatus.SUCCEEDED:
            if self.return_code != 0:
                raise ValueError("A successful command must have exit code 0.")

        if self.status == CommandStatus.FAILED:
            if self.return_code is None or self.return_code == 0:
                raise ValueError("A failed command must have a non-zero exit code.")

        if (
            self.status
            in {
                CommandStatus.BLOCKED,
                CommandStatus.REQUIRES_APPROVAL,
                CommandStatus.TIMED_OUT,
            }
            and not self.reason.strip()
        ):
            raise ValueError(
                "Blocked, approval-required, and timed-out results must include a reason."
            )

    @property
    def succeeded(self) -> bool:
        """Return whether the command completed successfully."""
        return self.status == CommandStatus.SUCCEEDED
