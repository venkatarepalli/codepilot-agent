"""Manage isolated working copies for CodePilot tasks."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

_DEFAULT_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "node_modules",
        "dist",
        "build",
    }
)


@dataclass(frozen=True, slots=True)
class WorkspaceInfo:
    """Describe a CodePilot task workspace."""

    issue_number: int
    path: Path
    source_repository: Path

    def __post_init__(self) -> None:
        """Validate workspace metadata."""
        if self.issue_number <= 0:
            raise ValueError("issue_number must be positive.")


class WorkspaceManager:
    """Create, access, track, and remove task workspaces."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        excluded_directories: frozenset[str] = (_DEFAULT_EXCLUDED_DIRECTORIES),
    ) -> None:
        """Initialize the workspace manager."""
        self.workspace_root = Path(workspace_root).resolve()
        self.excluded_directories = excluded_directories

        self.workspace_root.mkdir(parents=True, exist_ok=True)

        if not self.workspace_root.is_dir():
            raise ValueError("workspace_root must be a directory.")

    def create_workspace(
        self,
        repository_path: str | Path,
        issue_number: int,
    ) -> WorkspaceInfo:
        """Create a sanitized working copy for a GitHub issue."""
        if issue_number <= 0:
            raise ValueError("issue_number must be positive.")

        source = Path(repository_path).resolve(strict=True)

        if not source.is_dir():
            raise ValueError("repository_path must be a directory.")

        if self._is_within(source, self.workspace_root):
            raise ValueError("The source repository cannot be inside the workspace root.")

        base_workspace_path = self.workspace_root / f"issue-{issue_number}"

        # Claim a distinct directory atomically. The first attempt keeps the
        # familiar issue-N name; retries get a UUID suffix, preserving earlier
        # workspaces and avoiding a check-then-rename overwrite race.
        while True:
            if not base_workspace_path.exists():
                workspace_path = base_workspace_path
            else:
                workspace_path = self.workspace_root / (f"issue-{issue_number}-run-{uuid4().hex}")

            try:
                workspace_path.mkdir()
            except FileExistsError:
                # Another worker may have claimed the same candidate; retry
                # with a fresh candidate rather than reuse or overwrite it.
                continue
            break

        try:
            self._copy_repository(source, workspace_path)
        except Exception:
            shutil.rmtree(workspace_path, ignore_errors=True)
            raise

        return WorkspaceInfo(
            issue_number=issue_number,
            path=workspace_path.resolve(),
            source_repository=source,
        )

    def resolve_path(
        self,
        workspace: WorkspaceInfo,
        relative_path: str | Path,
    ) -> Path:
        """Resolve a path while enforcing the workspace boundary."""
        workspace_path = self._validate_workspace(workspace)

        requested_path = Path(relative_path)

        if requested_path.is_absolute():
            raise ValueError("Only relative workspace paths are permitted.")

        resolved_path = (workspace_path / requested_path).resolve(strict=False)

        if not self._is_within(resolved_path, workspace_path):
            raise ValueError("The requested path escapes the workspace.")

        return resolved_path

    def read_file(
        self,
        workspace: WorkspaceInfo,
        relative_path: str | Path,
    ) -> str:
        """Read a permitted text file from the workspace."""
        path = self.resolve_path(workspace, relative_path)
        self._check_sensitive_path(path, workspace.path)

        if not path.is_file():
            raise FileNotFoundError(path)

        return path.read_text(encoding="utf-8")

    def write_file(
        self,
        workspace: WorkspaceInfo,
        relative_path: str | Path,
        content: str,
    ) -> Path:
        """Write a text file inside the workspace."""
        path = self.resolve_path(workspace, relative_path)
        self._check_sensitive_path(path, workspace.path)

        if not isinstance(content, str):
            raise TypeError("content must be a string.")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        return path

    def get_changes(
        self,
        workspace: WorkspaceInfo,
    ) -> dict[str, list[str]]:
        """Compare the workspace with its original repository."""
        workspace_path = self._validate_workspace(workspace)
        source_path = workspace.source_repository

        source_files = self._list_relative_files(source_path)
        workspace_files = self._list_relative_files(workspace_path)

        added = sorted(workspace_files - source_files)
        deleted = sorted(source_files - workspace_files)

        modified = sorted(
            relative_path
            for relative_path in source_files & workspace_files
            if self._files_differ(
                source_path / relative_path,
                workspace_path / relative_path,
            )
        )

        return {
            "added": added,
            "modified": modified,
            "deleted": deleted,
        }

    def cleanup(self, workspace: WorkspaceInfo) -> None:
        """Remove a workspace after validating its location."""
        workspace_path = self._validate_workspace(workspace)

        if workspace_path == self.workspace_root:
            raise ValueError("Cannot remove the workspace root.")

        if not workspace_path.exists():
            return

        shutil.rmtree(workspace_path)

    def _copy_repository(
        self,
        source: Path,
        destination: Path,
    ) -> None:
        """Copy permitted repository content into staging."""
        for entry in source.iterdir():
            if entry.name in self.excluded_directories:
                continue

            if self._is_sensitive_name(entry.name):
                continue

            if entry.is_symlink():
                # Avoid importing links that could escape the workspace.
                continue

            target = destination / entry.name

            if entry.is_dir():
                shutil.copytree(
                    entry,
                    target,
                    ignore=self._copy_ignore,
                    symlinks=False,
                )
            elif entry.is_file():
                shutil.copy2(entry, target)

    def _copy_ignore(
        self,
        directory: str,
        names: list[str],
    ) -> set[str]:
        """Exclude unsafe, sensitive, or unnecessary entries from copied directories."""
        ignored: set[str] = set()
        parent = Path(directory)

        for name in names:
            entry = parent / name

            # Never follow symbolic links when copying nested directories.
            if entry.is_symlink():
                ignored.add(name)
            elif name in self.excluded_directories:
                ignored.add(name)
            elif self._is_sensitive_name(name):
                ignored.add(name)

        return ignored

    @staticmethod
    def _is_sensitive_name(name: str) -> bool:
        """Identify filenames excluded from automatic workspace copies."""
        normalized = name.lower()

        return (
            normalized == ".env"
            or normalized.startswith(".env.")
            or normalized.endswith((".secret", ".pem", ".key"))
            or "credentials" in normalized
        )

    def _check_sensitive_path(
        self,
        path: Path,
        workspace_path: Path,
    ) -> None:
        """Reject automatic access to sensitive files."""
        try:
            relative_path = path.relative_to(workspace_path)
        except ValueError as exc:
            raise ValueError("The requested path escapes the workspace.") from exc

        if any(self._is_sensitive_name(part) for part in relative_path.parts):
            raise PermissionError("Sensitive files cannot be accessed automatically.")

    def _validate_workspace(
        self,
        workspace: WorkspaceInfo,
    ) -> Path:
        """Ensure workspace metadata refers to a managed workspace."""
        resolved = workspace.path.resolve(strict=False)

        if not self._is_within(resolved, self.workspace_root):
            raise ValueError("Workspace is outside the configured root.")

        if resolved == self.workspace_root:
            raise ValueError("A task workspace cannot be the workspace root.")

        expected_name = f"issue-{workspace.issue_number}"
        retry_prefix = f"{expected_name}-run-"
        retry_id = (
            resolved.name[len(retry_prefix) :] if resolved.name.startswith(retry_prefix) else ""
        )
        is_retry_workspace = len(retry_id) == 32 and all(
            character in "0123456789abcdef" for character in retry_id
        )

        if resolved.name != expected_name and not is_retry_workspace:
            raise ValueError("Workspace metadata does not match its path.")

        if not resolved.is_dir():
            raise FileNotFoundError(resolved)

        return resolved

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        """Return whether a resolved path is contained within a root."""
        try:
            path.relative_to(root)
        except ValueError:
            return False

        return True

    @classmethod
    def _list_relative_files(cls, root: Path) -> set[str]:
        """List ordinary files while excluding unsafe entries."""
        results: set[str] = set()

        for current_root, directories, files in os.walk(
            root,
            followlinks=False,
        ):
            current = Path(current_root)

            directories[:] = [
                directory
                for directory in directories
                if directory not in _DEFAULT_EXCLUDED_DIRECTORIES
                and not cls._is_sensitive_name(directory)
                and not (current / directory).is_symlink()
            ]

            for filename in files:
                path = current / filename

                if cls._is_sensitive_name(filename) or path.is_symlink():
                    continue

                results.add(path.relative_to(root).as_posix())

        return results

    @staticmethod
    def _files_differ(first: Path, second: Path) -> bool:
        """Compare files without assuming they contain text."""
        if first.stat().st_size != second.stat().st_size:
            return True

        return first.read_bytes() != second.read_bytes()
