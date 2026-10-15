"""Tests for controlled sandbox command execution."""

import sys
from pathlib import Path

import pytest

from codepilot.sandbox.execution_models import CommandStatus
from codepilot.sandbox.executor import SandboxExecutor
from codepilot.sandbox.policy import SandboxPolicy


@pytest.fixture
def executor(tmp_path: Path) -> SandboxExecutor:
    """Create an executor with a temporary sandbox."""
    root = tmp_path / "sandbox"
    root.mkdir()

    return SandboxExecutor(
        SandboxPolicy(root),
        timeout_seconds=2.0,
    )


@pytest.fixture
def sandbox_root(executor: SandboxExecutor) -> Path:
    """Return the configured sandbox root."""
    return executor.policy.sandbox_root


def test_executes_successful_command(
    executor: SandboxExecutor,
    sandbox_root: Path,
) -> None:
    """Successful subprocesses return structured output."""
    result = executor.execute(
        [sys.executable, "-c", "print('CodePilot sandbox')"],
        working_directory=sandbox_root,
    )

    assert result.status == CommandStatus.SUCCEEDED
    assert result.return_code == 0
    assert result.stdout.strip() == "CodePilot sandbox"
    assert result.succeeded


def test_returns_nonzero_exit_code(
    executor: SandboxExecutor,
    sandbox_root: Path,
) -> None:
    """Non-zero exit codes are represented as failures."""
    result = executor.execute(
        [sys.executable, "-c", "raise SystemExit(7)"],
        working_directory=sandbox_root,
    )

    assert result.status == CommandStatus.FAILED
    assert result.return_code == 7
    assert not result.succeeded


def test_blocks_prohibited_command(
    executor: SandboxExecutor,
    sandbox_root: Path,
) -> None:
    """Prohibited commands must not reach subprocess execution."""
    result = executor.execute(
        ["curl", "https://example.com"],
        working_directory=sandbox_root,
    )

    assert result.status == CommandStatus.BLOCKED
    assert result.return_code is None


def test_does_not_execute_command_requiring_approval(
    executor: SandboxExecutor,
    sandbox_root: Path,
) -> None:
    """Human-approval operations are returned without execution."""
    result = executor.execute(
        ["git", "push", "origin", "main"],
        working_directory=sandbox_root,
    )

    assert result.status == CommandStatus.REQUIRES_APPROVAL
    assert result.return_code is None


def test_rejects_working_directory_outside_sandbox(
    executor: SandboxExecutor,
    tmp_path: Path,
) -> None:
    """Commands cannot run from outside the sandbox."""
    result = executor.execute(
        [sys.executable, "--version"],
        working_directory=tmp_path,
    )

    assert result.status == CommandStatus.BLOCKED


def test_enforces_execution_timeout(
    tmp_path: Path,
) -> None:
    """Commands exceeding the timeout return a timeout result."""
    root = tmp_path / "sandbox"
    root.mkdir()

    executor = SandboxExecutor(
        SandboxPolicy(root),
        timeout_seconds=0.2,
    )

    result = executor.execute(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(5)",
        ],
        working_directory=root,
    )

    assert result.status == CommandStatus.TIMED_OUT
    assert "timeout" in result.reason.lower()


def test_truncates_large_output(
    tmp_path: Path,
) -> None:
    """Large output is truncated before being returned to the agent."""
    root = tmp_path / "sandbox"
    root.mkdir()

    executor = SandboxExecutor(
        SandboxPolicy(root),
        max_output_characters=20,
    )

    result = executor.execute(
        [
            sys.executable,
            "-c",
            "print('A' * 100)",
        ],
        working_directory=root,
    )

    assert result.status == CommandStatus.SUCCEEDED
    assert len(result.stdout) < 100
    assert "Output truncated" in result.stdout


def test_rejects_invalid_timeout_configuration(
    tmp_path: Path,
) -> None:
    """Timeouts must be positive."""
    root = tmp_path / "sandbox"
    root.mkdir()

    with pytest.raises(ValueError, match="timeout_seconds"):
        SandboxExecutor(
            SandboxPolicy(root),
            timeout_seconds=0,
        )


def test_rejects_invalid_output_limit(
    tmp_path: Path,
) -> None:
    """Output limits must be positive."""
    root = tmp_path / "sandbox"
    root.mkdir()

    with pytest.raises(ValueError, match="max_output_characters"):
        SandboxExecutor(
            SandboxPolicy(root),
            max_output_characters=0,
        )
