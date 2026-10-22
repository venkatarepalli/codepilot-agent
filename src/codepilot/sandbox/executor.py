"""Controlled subprocess execution for CodePilot."""

import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from codepilot.sandbox.execution_models import (
    CommandExecutionResult,
    CommandStatus,
)
from codepilot.sandbox.models import SecurityDecision
from codepilot.sandbox.policy import SandboxPolicy

ApprovalCallback = Callable[[str, Mapping[str, Any]], bool]


class SandboxExecutor:
    """Execute commands only after validating them against sandbox policy."""

    def __init__(
        self,
        policy: SandboxPolicy,
        *,
        timeout_seconds: float = 60.0,
        max_output_characters: int = 100_000,
        approval_callback: ApprovalCallback | None = None,
    ) -> None:
        """Configure command execution limits."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        if max_output_characters <= 0:
            raise ValueError("max_output_characters must be positive.")

        self.policy = policy
        self.timeout_seconds = timeout_seconds
        self.max_output_characters = max_output_characters
        self.approval_callback = approval_callback

    def execute(
        self,
        command: Sequence[str],
        *,
        working_directory: str | Path,
    ) -> CommandExecutionResult:
        """Validate and execute one command without invoking a shell."""
        arguments = tuple(command)

        if not arguments or any(not arg.strip() for arg in arguments):
            return CommandExecutionResult(
                status=CommandStatus.BLOCKED,
                command=arguments,
                reason="The command must contain non-empty arguments.",
            )

        if not all(isinstance(arg, str) for arg in arguments):
            return CommandExecutionResult(
                status=CommandStatus.BLOCKED,
                command=tuple(str(arg) for arg in arguments),
                reason="Every command argument must be a string.",
            )

        policy_result = self.policy.check_command(
            arguments,
            working_directory=working_directory,
        )

        if policy_result.decision == SecurityDecision.BLOCK:
            return CommandExecutionResult(
                status=CommandStatus.BLOCKED,
                command=arguments,
                reason=policy_result.reason,
            )

        if policy_result.decision == SecurityDecision.REQUIRES_APPROVAL:
            # Preserve the fail-closed behavior when no interactive approval
            # callback is configured. In TUI mode, a positive decision allows
            # this one validated command to proceed; denials never reach subprocess.
            if self.approval_callback is None:
                return CommandExecutionResult(
                    status=CommandStatus.REQUIRES_APPROVAL,
                    command=arguments,
                    reason=policy_result.reason,
                )

            operation = (
                "execute_git_push"
                if self.policy.is_git_push_command(arguments)
                else "execute_command_requires_approval"
            )
            details = {
                "command": list(arguments),
                "working_directory": str(Path(working_directory).resolve()),
                "reason": policy_result.reason,
            }
            try:
                approved = self.approval_callback(operation, details) is True
            except Exception as exc:
                return CommandExecutionResult(
                    status=CommandStatus.REQUIRES_APPROVAL,
                    command=arguments,
                    reason=(
                        "Human-approval callback failed; command was not executed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            if not approved:
                return CommandExecutionResult(
                    status=CommandStatus.REQUIRES_APPROVAL,
                    command=arguments,
                    reason=f"Human approval was not granted. {policy_result.reason}",
                )

        started_at = time.monotonic()

        try:
            completed = subprocess.run(
                arguments,
                cwd=Path(working_directory).resolve(),
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                stdin=subprocess.DEVNULL,
            )

        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started_at

            return CommandExecutionResult(
                status=CommandStatus.TIMED_OUT,
                command=arguments,
                stdout=self._decode_output(exc.stdout),
                stderr=self._decode_output(exc.stderr),
                duration_seconds=duration,
                reason=(f"Command exceeded the {self.timeout_seconds}-second execution timeout."),
            )

        except (OSError, ValueError) as exc:
            duration = time.monotonic() - started_at

            return CommandExecutionResult(
                status=CommandStatus.FAILED,
                command=arguments,
                return_code=-1,
                duration_seconds=duration,
                reason=f"{type(exc).__name__}: {exc}",
                stderr=str(exc),
            )

        duration = time.monotonic() - started_at

        stdout = self._truncate(completed.stdout)
        stderr = self._truncate(completed.stderr)

        if completed.returncode == 0:
            return CommandExecutionResult(
                status=CommandStatus.SUCCEEDED,
                command=arguments,
                return_code=0,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )

        return CommandExecutionResult(
            status=CommandStatus.FAILED,
            command=arguments,
            return_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            reason=f"Command exited with status {completed.returncode}.",
        )

    def _truncate(self, output: str) -> str:
        """Limit captured output returned to the agent."""
        if len(output) <= self.max_output_characters:
            return output

        omitted = len(output) - self.max_output_characters

        return (
            output[: self.max_output_characters]
            + f"\n[Output truncated; {omitted} characters omitted.]"
        )

    @staticmethod
    def _decode_output(output: str | bytes | None) -> str:
        """Normalize subprocess timeout output to text."""
        if output is None:
            return ""

        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")

        return output
