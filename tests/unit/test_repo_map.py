"""Tests for CodePilot repository map generation."""

from pathlib import Path

import pytest

from codepilot.context.repo_map import (
    RepoMapCache,
    build_repo_map,
    estimate_token_count,
)


def test_builds_map_with_file_summaries(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    source.write_text(
        '"""Example module."""\n\ndef calculate_total():\n    return 10\n',
        encoding="utf-8",
    )
    repo_map = build_repo_map(tmp_path)
    assert repo_map.root == tmp_path.name
    assert len(repo_map.files) == 1
    assert repo_map.files[0].path == "example.py"
    assert repo_map.files[0].language == "Python"
    assert "Example module." in repo_map.files[0].summary
    assert "calculate_total" in repo_map.files[0].summary


def test_excludes_virtual_environments_and_generated_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")
    excluded = tmp_path / ".venv" / "Lib" / "hidden.py"
    excluded.parent.mkdir(parents=True)
    excluded.write_text("SECRET = True", encoding="utf-8")
    repo_map = build_repo_map(tmp_path)
    paths = {file.path for file in repo_map.files}
    assert "app.py" in paths
    assert not any(".venv" in path for path in paths)


def test_excludes_sensitive_files_without_reading_them(
    tmp_path: Path,
) -> None:
    sensitive_files = [
        ".env",
        ".env.production",
        "application.secret",
        "server.pem",
        "private.key",
        "database_credentials.json",
    ]
    for name in sensitive_files:
        (tmp_path / name).write_text("DO NOT READ", encoding="utf-8")
    (tmp_path / "safe.py").write_text("VALUE = 1", encoding="utf-8")
    repo_map = build_repo_map(tmp_path)
    assert {file.path for file in repo_map.files} == {"safe.py"}


def test_directory_tree_contains_nested_paths(tmp_path: Path) -> None:
    nested = tmp_path / "src" / "codepilot" / "app.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("VALUE = 1", encoding="utf-8")
    repo_map = build_repo_map(tmp_path)
    assert "src/" in repo_map.directory_tree
    assert "codepilot/" in repo_map.directory_tree
    assert "app.py" in repo_map.directory_tree


def test_render_includes_tree_and_summaries(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Example", encoding="utf-8")
    rendered = build_repo_map(tmp_path).render()
    assert "Directory tree:" in rendered
    assert "File summaries:" in rendered
    assert "README.md" in rendered


def test_missing_repository_raises_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_repo_map(tmp_path / "missing")


def test_file_path_instead_of_directory_raises_error(tmp_path: Path) -> None:
    file_path = tmp_path / "example.py"
    file_path.write_text("VALUE = 1", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        build_repo_map(file_path)


def test_file_order_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "zebra.py").write_text("VALUE = 1", encoding="utf-8")
    (tmp_path / "alpha.py").write_text("VALUE = 2", encoding="utf-8")
    repo_map = build_repo_map(tmp_path)
    assert [file.path for file in repo_map.files] == ["alpha.py", "zebra.py"]


def test_token_budget_is_respected(tmp_path: Path) -> None:
    for index in range(20):
        (tmp_path / f"module_{index:02}.py").write_text(
            f'"""Module {index}."""\ndef function_{index}():\n    return {index}\n',
            encoding="utf-8",
        )
    budget = 500
    repo_map = build_repo_map(tmp_path, token_budget=budget)
    assert estimate_token_count(repo_map.render()) <= budget


def test_rejects_invalid_token_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_repo_map(tmp_path, token_budget=0)


def test_cache_builds_and_reuses_repo_map(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(
        '"""Initial module."""\n',
        encoding="utf-8",
    )
    cache = RepoMapCache(tmp_path / "cache")
    first = cache.get_or_build(tmp_path)
    second = cache.get_or_build(tmp_path)
    assert first == second
    assert (tmp_path / "cache" / "repo_map.json").is_file()
    assert [item.path for item in second.files] == ["app.py"]


def test_cache_invalidates_after_file_changes(tmp_path: Path) -> None:
    source = tmp_path / "app.py"
    source.write_text('"""Initial module."""\n', encoding="utf-8")
    cache = RepoMapCache(tmp_path / "cache")
    first = cache.get_or_build(tmp_path)
    source.write_text(
        '"""Updated module."""\ndef new_function():\n    return True\n',
        encoding="utf-8",
    )
    second = cache.get_or_build(tmp_path)
    assert first != second
    assert "new_function" in second.files[0].summary


def test_cache_recovers_from_corrupt_json(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    (cache_directory / "repo_map.json").write_text(
        "{invalid JSON",
        encoding="utf-8",
    )
    repo_map = RepoMapCache(cache_directory).get_or_build(tmp_path)
    assert [item.path for item in repo_map.files] == ["app.py"]


def test_cache_rebuilds_when_budget_changes(tmp_path: Path) -> None:
    for index in range(10):
        (tmp_path / f"module_{index}.py").write_text(
            f"VALUE = {index}\n",
            encoding="utf-8",
        )
    cache = RepoMapCache(tmp_path / "cache")
    cache.get_or_build(tmp_path, token_budget=1000)
    smaller_map = cache.get_or_build(tmp_path, token_budget=300)
    assert estimate_token_count(smaller_map.render()) <= 300
    assert "cache/" not in smaller_map.directory_tree
    assert all(not item.path.startswith("cache/") for item in smaller_map.files)


def test_cache_outside_repository_does_not_exclude_repository_files(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    cache = RepoMapCache(tmp_path / "external-cache")
    repo_map = cache.get_or_build(repository)
    assert [item.path for item in repo_map.files] == ["app.py"]


def test_cache_directory_nested_in_repository_is_excluded(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    cache = RepoMapCache(tmp_path / "artifacts" / "cache")
    repo_map = cache.get_or_build(tmp_path)
    assert [item.path for item in repo_map.files] == ["app.py"]
    assert "artifacts/" not in repo_map.directory_tree


def test_cache_invalidates_after_git_tracked_file_changes(
    tmp_path: Path,
) -> None:
    """Repo Map is rebuilt after a tracked file changes in Git."""
    import subprocess

    def run_git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    run_git("init")
    run_git("config", "user.email", "codepilot-test@example.com")
    run_git("config", "user.name", "CodePilot Test")

    source = tmp_path / "app.py"
    source.write_text('"""Initial module."""\n', encoding="utf-8")

    run_git("add", "app.py")
    run_git("commit", "-m", "Initial repository")

    cache = RepoMapCache(tmp_path / "cache")
    first = cache.get_or_build(tmp_path)

    source.write_text(
        '"""Updated module."""\ndef new_function():\n    return True\n',
        encoding="utf-8",
    )

    second = cache.get_or_build(tmp_path)

    assert first != second
    assert "new_function" in second.files[0].summary


def test_cache_rebuilds_when_tracked_file_has_uncommitted_changes(
    tmp_path: Path,
) -> None:
    """Uncommitted Git changes invalidate a cached Repo Map."""
    import subprocess

    def run_git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    run_git("init")
    run_git("config", "user.email", "codepilot-test@example.com")
    run_git("config", "user.name", "CodePilot Test")

    source = tmp_path / "app.py"
    source.write_text('"""Initial module."""\n', encoding="utf-8")

    run_git("add", "app.py")
    run_git("commit", "-m", "Initial repository")

    cache = RepoMapCache(tmp_path / "cache")
    first = cache.get_or_build(tmp_path)

    source.write_text(
        '"""Updated module."""\ndef changed_function():\n    return True\n',
        encoding="utf-8",
    )

    second = cache.get_or_build(tmp_path)

    assert first != second
    assert "changed_function" in second.files[0].summary
