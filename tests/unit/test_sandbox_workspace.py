"""Tests for CodePilot workspace management."""

from pathlib import Path

import pytest

from codepilot.sandbox.workspace import WorkspaceManager


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create a small sample repository."""
    root = tmp_path / "repository"
    root.mkdir()

    (root / "README.md").write_text("# Example\n", encoding="utf-8")
    (root / "app.py").write_text("print('hello')\n", encoding="utf-8")

    (root / ".env").write_text("SECRET=value\n", encoding="utf-8")
    (root / "credentials.json").write_text(
        '{"password": "secret"}',
        encoding="utf-8",
    )

    (root / ".git").mkdir()
    (root / ".git" / "config").write_text(
        "[core]\n",
        encoding="utf-8",
    )

    return root


@pytest.fixture
def manager(tmp_path: Path) -> WorkspaceManager:
    """Create a workspace manager."""
    return WorkspaceManager(tmp_path / "workspaces")


@pytest.fixture
def workspace(
    manager: WorkspaceManager,
    repository: Path,
):
    """Create a workspace for issue 123."""
    return manager.create_workspace(repository, 123)


def test_creates_isolated_workspace(
    manager: WorkspaceManager,
    repository: Path,
    workspace,
) -> None:
    """Workspace creation must not modify the source repository."""
    assert workspace.path.is_dir()
    assert workspace.path != repository
    assert (workspace.path / "README.md").is_file()
    assert (repository / "README.md").is_file()


def test_excludes_git_metadata_and_secrets(workspace) -> None:
    """Git metadata and sensitive files are excluded from copies."""
    assert not (workspace.path / ".git").exists()
    assert not (workspace.path / ".env").exists()
    assert not (workspace.path / "credentials.json").exists()


def test_repeated_issue_gets_isolated_workspace_without_overwriting_previous(
    manager: WorkspaceManager,
    repository: Path,
) -> None:
    """Retrying an issue preserves its old workspace and creates a fresh copy."""
    first = manager.create_workspace(repository, 123)
    manager.write_file(first, "app.py", "first attempt's working change\n")

    second = manager.create_workspace(repository, 123)

    assert first.path != second.path
    assert first.path.name == "issue-123"
    assert second.path.name.startswith("issue-123-run-")
    assert manager.read_file(first, "app.py") == "first attempt's working change\n"
    assert manager.read_file(second, "app.py") == "print('hello')\n"
    assert manager.get_changes(first)["modified"] == ["app.py"]
    assert manager.get_changes(second) == {"added": [], "modified": [], "deleted": []}

    manager.cleanup(second)
    assert not second.path.exists()
    assert first.path.is_dir()


def test_rejects_workspace_metadata_with_invalid_retry_suffix(
    manager: WorkspaceManager,
    repository: Path,
) -> None:
    """Retry workspace names must use the expected issue prefix and UUID form."""
    from codepilot.sandbox.workspace import WorkspaceInfo

    retry_path = manager.workspace_root / "issue-123-run-not-a-uuid"
    retry_path.mkdir()
    forged = WorkspaceInfo(
        issue_number=123,
        path=retry_path,
        source_repository=repository,
    )

    with pytest.raises(ValueError, match="does not match its path"):
        manager.read_file(forged, "app.py")


def test_rejects_invalid_issue_number(
    manager: WorkspaceManager,
    repository: Path,
) -> None:
    """Issue numbers must be positive."""
    with pytest.raises(ValueError, match="issue_number"):
        manager.create_workspace(repository, 0)


def test_rejects_source_inside_workspace_root(
    manager: WorkspaceManager,
) -> None:
    """A workspace must not be copied into itself."""
    source = manager.workspace_root / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="inside the workspace root"):
        manager.create_workspace(source, 456)


def test_resolves_relative_path(
    manager: WorkspaceManager,
    workspace,
) -> None:
    """Relative paths resolve inside the workspace."""
    path = manager.resolve_path(workspace, "app.py")

    assert path == workspace.path / "app.py"


def test_rejects_path_traversal(
    manager: WorkspaceManager,
    workspace,
) -> None:
    """Relative paths cannot escape the workspace."""
    with pytest.raises(ValueError, match="escapes the workspace"):
        manager.resolve_path(workspace, "../../outside.txt")


def test_rejects_absolute_path(
    manager: WorkspaceManager,
    workspace,
    tmp_path: Path,
) -> None:
    """Absolute paths are not accepted by workspace operations."""
    with pytest.raises(ValueError, match="relative"):
        manager.resolve_path(workspace, tmp_path / "outside.txt")


def test_reads_and_writes_files(
    manager: WorkspaceManager,
    workspace,
) -> None:
    """Workspace files can be read and safely updated."""
    manager.write_file(workspace, "src/example.py", "answer = 42\n")

    content = manager.read_file(workspace, "src/example.py")

    assert content == "answer = 42\n"


@pytest.mark.parametrize(
    "filename",
    [
        ".env",
        ".env.local",
        "private.secret",
        "private.pem",
        "private.key",
        "database_credentials.json",
    ],
)
def test_rejects_sensitive_file_access(
    manager: WorkspaceManager,
    workspace,
    filename: str,
) -> None:
    """Sensitive filenames cannot be read or written automatically."""
    with pytest.raises(PermissionError, match="Sensitive files"):
        manager.read_file(workspace, filename)

    with pytest.raises(PermissionError, match="Sensitive files"):
        manager.write_file(workspace, filename, "unsafe")


def test_tracks_added_modified_and_deleted_files(
    manager: WorkspaceManager,
    workspace,
) -> None:
    """Workspace differences are reported relative to the source."""
    (workspace.path / "app.py").write_text(
        "print('updated')\n",
        encoding="utf-8",
    )

    (workspace.path / "README.md").unlink()

    (workspace.path / "new.py").write_text(
        "answer = 42\n",
        encoding="utf-8",
    )

    changes = manager.get_changes(workspace)

    assert changes["modified"] == ["app.py"]
    assert changes["deleted"] == ["README.md"]
    assert changes["added"] == ["new.py"]


def test_cleanup_removes_workspace(
    manager: WorkspaceManager,
    workspace,
) -> None:
    """Cleanup removes only the managed task workspace."""
    manager.cleanup(workspace)

    assert not workspace.path.exists()


def test_cleanup_does_not_remove_source_repository(
    manager: WorkspaceManager,
    repository: Path,
    workspace,
) -> None:
    """Workspace cleanup leaves the original repository untouched."""
    manager.cleanup(workspace)

    assert repository.is_dir()
    assert (repository / "app.py").is_file()


def test_rejects_workspace_metadata_outside_managed_root(
    manager: WorkspaceManager,
    workspace,
    tmp_path: Path,
) -> None:
    """Forged workspace metadata cannot target an arbitrary directory."""
    from codepilot.sandbox.workspace import WorkspaceInfo

    outside = tmp_path / "outside"
    outside.mkdir()

    forged = WorkspaceInfo(
        issue_number=123,
        path=outside,
        source_repository=workspace.source_repository,
    )

    with pytest.raises(ValueError, match="outside the configured root"):
        manager.cleanup(forged)


def test_nested_symlinks_are_not_copied(
    manager: WorkspaceManager,
    repository: Path,
    tmp_path: Path,
) -> None:
    """Nested symbolic links must not import external files."""
    source_directory = repository / "src"
    source_directory.mkdir()

    external_file = tmp_path / "external_data.txt"
    external_file.write_text(
        "external content",
        encoding="utf-8",
    )

    symlink = source_directory / "external_data.txt"

    try:
        symlink.symlink_to(external_file)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symbolic links are unavailable: {exc}")

    workspace = manager.create_workspace(repository, 456)

    copied_file = workspace.path / "src" / "external_data.txt"

    assert not copied_file.exists()
    assert not copied_file.is_symlink()


def test_copy_ignore_excludes_nested_symlinks(
    manager: WorkspaceManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested symbolic links must be excluded from directory copies."""
    source_directory = tmp_path / "source"
    source_directory.mkdir()

    linked_path = source_directory / "external_data.txt"

    original_is_symlink = Path.is_symlink

    def mock_is_symlink(path: Path) -> bool:
        if path == linked_path:
            return True

        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", mock_is_symlink)

    ignored = manager._copy_ignore(
        str(source_directory),
        ["external_data.txt", "app.py"],
    )

    assert ignored == {"external_data.txt"}
