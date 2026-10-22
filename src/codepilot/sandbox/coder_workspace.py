"""Unified workspace interface for CodePilot's Coder Agent."""

from collections.abc import Sequence
from pathlib import Path

from codepilot.sandbox.execution_models import CommandExecutionResult
from codepilot.sandbox.executor import SandboxExecutor
from codepilot.sandbox.workspace import WorkspaceInfo, WorkspaceManager


class CoderWorkspace:
    """Coordinate workspace management and controlled execution."""

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        executor: SandboxExecutor,
    ) -> None:
        """Initialize the CoderWorkspace integration layer."""
        self._workspace_manager = workspace_manager
        self._executor = executor

    def create(
        self,
        repository_path: str | Path,
        issue_number: int,
    ) -> WorkspaceInfo:
        """Create a dedicated workspace for a GitHub issue."""
        return self._workspace_manager.create_workspace(
            repository_path,
            issue_number,
        )

    def resolve_path(
        self,
        workspace: WorkspaceInfo,
        relative_path: str | Path,
    ) -> Path:
        """Resolve a path while enforcing workspace boundaries."""
        return self._workspace_manager.resolve_path(
            workspace,
            relative_path,
        )

    def read_file(
        self,
        workspace: WorkspaceInfo,
        relative_path: str | Path,
    ) -> str:
        """Read a file through the workspace manager."""
        return self._workspace_manager.read_file(
            workspace,
            relative_path,
        )

    def write_file(
        self,
        workspace: WorkspaceInfo,
        relative_path: str | Path,
        content: str,
    ) -> Path:
        """Write a file through the workspace manager."""
        return self._workspace_manager.write_file(
            workspace,
            relative_path,
            content,
        )

    def get_changes(
        self,
        workspace: WorkspaceInfo,
    ) -> dict[str, list[str]]:
        """Return added, modified, and deleted repository-relative files."""
        return self._workspace_manager.get_changes(workspace)

    def execute(
        self,
        workspace: WorkspaceInfo,
        command: Sequence[str],
        *,
        working_directory: str | Path = ".",
    ) -> CommandExecutionResult:
        """Execute a command inside the issue workspace."""
        resolved_directory = self.resolve_path(
            workspace,
            working_directory,
        )

        if not resolved_directory.is_dir():
            raise NotADirectoryError(f"Working directory does not exist: {resolved_directory}")

        return self._executor.execute(
            command,
            working_directory=resolved_directory,
        )

    def cleanup(self, workspace: WorkspaceInfo) -> None:
        """Remove the issue workspace through the manager."""
        self._workspace_manager.cleanup(workspace)
