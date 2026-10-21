"""Tests for sandbox security policy."""

from pathlib import Path

import pytest

from codepilot.sandbox.models import SecurityDecision
from codepilot.sandbox.policy import SandboxPolicy


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[SandboxPolicy, Path]:
    """Create a temporary sandbox for testing."""
    root = tmp_path / "sandbox"
    root.mkdir()

    return SandboxPolicy(root), root


@pytest.mark.parametrize(
    "command",
    [
        ["git", "push", "origin", "main"],
        ["git", "-C", "repo", "push", "origin", "main"],
        ["git", "--git-dir=.git", "push", "origin", "main"],
        ["git", "-c", "credential.helper=store", "push", "origin", "main"],
        [r"C:\Program Files\Git\cmd\git.exe", "push", "origin", "main"],
    ],
)
def test_requires_approval_for_git_push_with_global_options(
    sandbox: tuple[SandboxPolicy, Path],
    command: list[str],
) -> None:
    """Git global options must not bypass the push approval gate."""
    policy, root = sandbox

    result = policy.check_command(command, working_directory=root)

    assert result.decision == SecurityDecision.REQUIRES_APPROVAL


@pytest.mark.parametrize(
    "filename",
    [
        ".env.local",
        ".env.production",
        ".env.development",
        ".env.test",
        ".env.staging",
    ],
)
def test_environment_file_variants_require_approval(
    sandbox: tuple[SandboxPolicy, Path],
    filename: str,
) -> None:
    """Environment-specific configuration files require approval."""
    policy, root = sandbox

    result = policy.check_path(
        root / filename,
        operation="read",
    )

    assert result.decision == SecurityDecision.REQUIRES_APPROVAL


@pytest.mark.parametrize(
    "filename",
    [
        ".env.example",
        ".env.sample",
    ],
)
def test_environment_example_files_are_not_sensitive(
    sandbox: tuple[SandboxPolicy, Path],
    filename: str,
) -> None:
    """Documented environment examples should remain accessible."""
    policy, root = sandbox

    result = policy.check_path(
        root / filename,
        operation="read",
    )

    assert result.decision == SecurityDecision.ALLOW


@pytest.mark.parametrize(
    "command",
    [
        ["curl", "https://example.com"],
        ["wget", "https://example.com"],
        ["pip", "install", "some-package"],
        ["python", "-m", "pip", "install", "some-package"],
        ["python", "-m", "pip", "install", "--upgrade", "some-package"],
        ["rm", "-rf", "some-directory"],
        ["rm", "-fr", "some-directory"],
        ["rm", "--recursive", "--force", "some-directory"],
    ],
)
def test_blocks_prohibited_commands(
    sandbox: tuple[SandboxPolicy, Path],
    command: list[str],
) -> None:
    """Prohibited commands must never be automatically permitted."""
    policy, root = sandbox

    result = policy.check_command(
        command,
        working_directory=root,
    )

    assert result.decision == SecurityDecision.BLOCK


def test_allows_a_safe_command(
    sandbox: tuple[SandboxPolicy, Path],
) -> None:
    """A safe command in the sandbox should be permitted."""
    policy, root = sandbox

    result = policy.check_command(
        ["python", "--version"],
        working_directory=root,
    )

    assert result.decision == SecurityDecision.ALLOW


def test_requires_approval_for_git_push(
    sandbox: tuple[SandboxPolicy, Path],
) -> None:
    """Pushing to a Git remote requires explicit human approval."""
    policy, root = sandbox

    result = policy.check_command(
        ["git", "push", "origin", "main"],
        working_directory=root,
    )

    assert result.decision == SecurityDecision.REQUIRES_APPROVAL
    assert result.requires_human_approval


@pytest.mark.parametrize(
    "filename",
    [
        ".env",
        "application.secret",
        "private.pem",
        "signing.key",
        "database_credentials.json",
    ],
)
def test_sensitive_files_require_approval(
    sandbox: tuple[SandboxPolicy, Path],
    filename: str,
) -> None:
    """Sensitive paths must not be accessed automatically."""
    policy, root = sandbox

    result = policy.check_path(
        root / filename,
        operation="read",
    )

    assert result.decision == SecurityDecision.REQUIRES_APPROVAL


@pytest.mark.parametrize(
    "operation",
    ["read", "write", "delete"],
)
def test_sensitive_file_operations_require_approval(
    sandbox: tuple[SandboxPolicy, Path],
    operation: str,
) -> None:
    """Sensitive files must be protected for every operation."""
    policy, root = sandbox

    result = policy.check_path(
        root / ".env",
        operation=operation,
    )

    assert result.decision == SecurityDecision.REQUIRES_APPROVAL


def test_blocks_paths_outside_sandbox(
    sandbox: tuple[SandboxPolicy, Path],
    tmp_path: Path,
) -> None:
    """Paths outside the configured root must be rejected."""
    policy, _ = sandbox

    result = policy.check_path(
        tmp_path / "outside.txt",
        operation="read",
    )

    assert result.decision == SecurityDecision.BLOCK


def test_blocks_working_directory_outside_sandbox(
    sandbox: tuple[SandboxPolicy, Path],
    tmp_path: Path,
) -> None:
    """Commands cannot run with a working directory outside the root."""
    policy, _ = sandbox

    result = policy.check_command(
        ["python", "--version"],
        working_directory=tmp_path,
    )

    assert result.decision == SecurityDecision.BLOCK


def test_rejects_shell_control_operators(
    sandbox: tuple[SandboxPolicy, Path],
) -> None:
    """Commands must not chain arbitrary shell operations."""
    policy, root = sandbox

    result = policy.check_command(
        ["python", "--version", "&&", "python", "another_script.py"],
        working_directory=root,
    )

    assert result.decision == SecurityDecision.BLOCK


def test_rejects_empty_command(
    sandbox: tuple[SandboxPolicy, Path],
) -> None:
    """An empty command must not be accepted."""
    policy, root = sandbox

    result = policy.check_command(
        [],
        working_directory=root,
    )

    assert result.decision == SecurityDecision.BLOCK


def test_rejects_unsupported_filesystem_operation(
    sandbox: tuple[SandboxPolicy, Path],
) -> None:
    """Unknown operation types must fail closed."""
    policy, root = sandbox

    result = policy.check_path(
        root / "example.txt",
        operation="change_permissions",
    )

    assert result.decision == SecurityDecision.BLOCK
