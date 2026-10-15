"""Tests for the CoderWorkspace integration layer."""

import sys
from pathlib import Path

import pytest

from codepilot.sandbox.coder_workspace import CoderWorkspace
from codepilot.sandbox.execution_models import CommandStatus
from codepilot.sandbox.executor import SandboxExecutor
from codepilot.sandbox.policy import SandboxPolicy
from codepilot.sandbox.workspace import WorkspaceInfo, WorkspaceManager


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create a small source repository."""
    root = tmp_path / "repository"
    root.mkdir()

    (root / "README.md").write_text(
        "# CodePilot test repository\n",
        encoding="utf-8",
    )
    (root / "app.py").write_text(
        "answer = 42\n",
        encoding="utf-8",
    )

    return root


@pytest.fixture
def coder_workspace(tmp_path: Path) -> CoderWorkspace:
    """Create the integrated workspace interface."""
    workspace_root = tmp_path / "workspaces"
    workspace_root.mkdir()

    manager = WorkspaceManager(workspace_root)
    executor = SandboxExecutor(
        SandboxPolicy(workspace_root),
        timeout_seconds=2.0,
    )

    return CoderWorkspace(manager, executor)


@pytest.fixture
def task_workspace(
    coder_workspace: CoderWorkspace,
    repository: Path,
) -> WorkspaceInfo:
    """Create an issue-specific workspace."""
    return coder_workspace.create(repository, 123)


def test_creates_workspace(
    task_workspace: WorkspaceInfo,
    repository: Path,
) -> None:
    """The workspace must be separate from the source repository."""
    assert task_workspace.path.is_dir()
    assert task_workspace.path != repository
    assert (task_workspace.path / "app.py").is_file()


def test_reads_workspace_file(
    coder_workspace: CoderWorkspace,
    task_workspace: WorkspaceInfo,
) -> None:
    """The integration layer delegates file reading correctly."""
    content = coder_workspace.read_file(
        task_workspace,
        "app.py",
    )

    assert content == "answer = 42\n"


def test_writes_workspace_file(
    coder_workspace: CoderWorkspace,
    task_workspace: WorkspaceInfo,
) -> None:
    """The integration layer delegates file writing correctly."""
    coder_workspace.write_file(
        task_workspace,
        "src/example.py",
        "answer = 100\n",
    )

    assert (
        coder_workspace.read_file(
            task_workspace,
            "src/example.py",
        )
        == "answer = 100\n"
    )


def test_resolve_path_returns_path_inside_workspace(
    coder_workspace: CoderWorkspace,
    task_workspace: WorkspaceInfo,
) -> None:
    """Resolve a relative path through the public workspace interface."""
    resolved_path = coder_workspace.resolve_path(
        task_workspace,
        "working/proposed_diff.txt",
    )

    assert resolved_path == (task_workspace.path / "working" / "proposed_diff.txt")


def test_executes_command_in_workspace(
    coder_workspace: CoderWorkspace,
    task_workspace: WorkspaceInfo,
) -> None:
    """Commands run through the controlled executor."""
    result = coder_workspace.execute(
        task_workspace,
        [
            sys.executable,
            "-c",
            "print('workspace ready')",
        ],
    )

    assert result.status == CommandStatus.SUCCEEDED
    assert result.stdout.strip() == "workspace ready"


def test_executes_command_in_subdirectory(
    coder_workspace: CoderWorkspace,
    task_workspace: WorkspaceInfo,
) -> None:
    """Commands may use an existing subdirectory as their working directory."""
    (task_workspace.path / "src").mkdir()

    result = coder_workspace.execute(
        task_workspace,
        [
            sys.executable,
            "-c",
            "import os; print(os.getcwd())",
        ],
        working_directory="src",
    )

    assert result.status == CommandStatus.SUCCEEDED
    assert Path(result.stdout.strip()) == task_workspace.path / "src"


def test_blocks_prohibited_command(
    coder_workspace: CoderWorkspace,
    task_workspace: WorkspaceInfo,
) -> None:
    """The integration layer must preserve executor policy enforcement."""
    result = coder_workspace.execute(
        task_workspace,
        ["curl", "https://example.com"],
    )

    assert result.status == CommandStatus.BLOCKED


def test_does_not_execute_git_push(
    coder_workspace: CoderWorkspace,
    task_workspace: WorkspaceInfo,
) -> None:
    """Git pushes remain subject to human approval."""
    result = coder_workspace.execute(
        task_workspace,
        ["git", "push", "origin", "main"],
    )

    assert result.status == CommandStatus.REQUIRES_APPROVAL


def test_rejects_working_directory_outside_workspace(
    coder_workspace: CoderWorkspace,
    task_workspace: WorkspaceInfo,
) -> None:
    """The integration layer must reject directory traversal."""
    with pytest.raises(ValueError, match="escapes the workspace"):
        coder_workspace.execute(
            task_workspace,
            [sys.executable, "--version"],
            working_directory="../../outside",
        )


def test_rejects_nonexistent_working_directory(
    coder_workspace: CoderWorkspace,
    task_workspace: WorkspaceInfo,
) -> None:
    """Commands cannot use a nonexistent working directory."""
    with pytest.raises(NotADirectoryError):
        coder_workspace.execute(
            task_workspace,
            [sys.executable, "--version"],
            working_directory="missing",
        )


def test_tracks_changes(
    coder_workspace: CoderWorkspace,
    task_workspace: WorkspaceInfo,
) -> None:
    """Changes are reported relative to the original repository."""
    coder_workspace.write_file(
        task_workspace,
        "new_module.py",
        "answer = 7\n",
    )

    changes = coder_workspace.get_changes(task_workspace)

    assert changes["added"] == ["new_module.py"]
    assert changes["modified"] == []
    assert changes["deleted"] == []


def test_cleanup_removes_workspace(
    coder_workspace: CoderWorkspace,
    task_workspace: WorkspaceInfo,
) -> None:
    """Cleanup removes the issue workspace."""
    workspace_path = task_workspace.path

    coder_workspace.cleanup(task_workspace)

    assert not workspace_path.exists()


def test_cleanup_preserves_original_repository(
    coder_workspace: CoderWorkspace,
    repository: Path,
    task_workspace: WorkspaceInfo,
) -> None:
    """Cleaning up a task must not remove the source repository."""
    coder_workspace.cleanup(task_workspace)

    assert repository.is_dir()
    assert (repository / "app.py").read_text(encoding="utf-8") == "answer = 42\n"
