"""Repository map generation, token budgeting, and disk caching."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".codepilot",
}
LANGUAGE_BY_EXTENSION = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".sql": "SQL",
    ".html": "HTML",
    ".css": "CSS",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".md": "Markdown",
    ".sh": "Shell",
    ".ps1": "PowerShell",
}
MAX_SUMMARY_LENGTH = 160
DEFAULT_TOKEN_BUDGET = 4000
CACHE_FILENAME = "repo_map.json"


@dataclass(frozen=True, slots=True)
class FileSummary:
    """Compact metadata describing a repository file."""

    path: str
    language: str
    summary: str


@dataclass(frozen=True, slots=True)
class RepoMap:
    """Structured representation of a repository."""

    root: str
    directory_tree: str
    files: tuple[FileSummary, ...]

    def render(self) -> str:
        """Render the repository map as readable text."""
        sections = [
            f"Repository: {self.root}",
            "",
            "Directory tree:",
            self.directory_tree,
            "",
            "File summaries:",
        ]
        sections.extend(f"- {file.path} [{file.language}]: {file.summary}" for file in self.files)
        return "\n".join(sections)


def estimate_token_count(text: str) -> int:
    """Estimate tokens conservatively without a model-specific tokenizer."""
    if not text:
        return 0
    return max(1, (len(text) + 2) // 3)


def _is_sensitive_file(path: Path) -> bool:
    """Identify files that must not be inspected by the Repo Explorer."""
    name = path.name.lower()
    if name == ".env" or name.startswith(".env."):
        return True
    if name.endswith((".secret", ".pem", ".key")):
        return True
    return "credentials" in name


def _is_within(path: Path, directory: Path) -> bool:
    """Return True if path is directory itself or is contained beneath it."""
    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=False))
    except ValueError:
        return False
    return True


def _is_excluded(
    relative_path: Path,
    absolute_path: Path,
    excluded_paths: tuple[Path, ...] = (),
) -> bool:
    """Check standard excluded directories and caller-supplied absolute paths."""
    if any(part in EXCLUDED_DIRECTORIES for part in relative_path.parts):
        return True
    return any(_is_within(absolute_path, excluded) for excluded in excluded_paths)


def _language_for(path: Path) -> str:
    """Return a human-readable language name for a file."""
    return LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "Other")


def _summarize_file(path: Path, language: str) -> str:
    """Create a short summary without retaining full file contents."""
    if language == "Python":
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            descriptions: list[str] = []
            module_doc = ast.get_docstring(tree)
            if module_doc:
                descriptions.append(module_doc.splitlines()[0].strip())
            symbols = [
                node.name
                for node in tree.body
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                )
            ]
            if symbols:
                descriptions.append("Defines: " + ", ".join(symbols[:6]))
            summary = "; ".join(descriptions)
            return summary[:MAX_SUMMARY_LENGTH] or "Python source file."
        except (OSError, UnicodeDecodeError, SyntaxError):
            return "Python source file; summary unavailable."
    if path.name.lower().startswith("readme"):
        return "Project overview and documentation."
    if path.name.lower() in {"pyproject.toml", "requirements.txt"}:
        return "Project configuration and dependency definitions."
    return f"{language} file."


def _build_directory_tree(paths: list[str]) -> str:
    """Build a deterministic tree from relative file paths."""
    tree: dict[str, dict] = {}
    for relative_path in paths:
        current = tree
        for part in Path(relative_path).parts:
            current = current.setdefault(part, {})
    lines: list[str] = []

    def visit(node: dict[str, dict], prefix: str = "") -> None:
        entries = sorted(
            node.items(),
            key=lambda item: (not bool(item[1]), item[0].lower()),
        )
        for index, (name, children) in enumerate(entries):
            is_last = index == len(entries) - 1
            connector = "└── " if is_last else "├── "
            is_directory = bool(children)
            lines.append(f"{prefix}{connector}{name}{'/' if is_directory else ''}")
            if is_directory:
                extension = "    " if is_last else "│   "
                visit(children, prefix + extension)

    visit(tree)
    return "\n".join(lines)


def _iter_repository_files(
    root: Path,
    excluded_paths: tuple[Path, ...] = (),
):
    """Yield repository files deterministically, excluding sensitive/cache paths."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root)
        if _is_excluded(relative_path, path, excluded_paths):
            continue
        if _is_sensitive_file(path):
            continue
        yield path, relative_path


def _map_fingerprint(
    root: Path,
    excluded_paths: tuple[Path, ...] = (),
) -> str:
    """Fingerprint relevant repository file metadata."""
    digest = hashlib.sha256()
    for path, relative_path in _iter_repository_files(root, excluded_paths):
        try:
            stat = path.stat()
        except OSError:
            continue
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _git_revision(root: Path) -> str | None:
    """Return the current Git revision, or None outside a Git repository."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_changed_files(root: Path) -> set[str] | None:
    """Return changed tracked paths relative to HEAD, or None if unavailable."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "--name-only",
                "HEAD",
                "--",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return {Path(line).as_posix() for line in result.stdout.splitlines() if line.strip()}


def _fit_token_budget(repo_map: RepoMap, token_budget: int) -> RepoMap:
    """Trim file summaries until the rendered map fits its token budget."""
    if token_budget < 1:
        raise ValueError("Token budget must be at least 1.")
    files = list(repo_map.files)
    while files:
        candidate = RepoMap(
            root=repo_map.root,
            directory_tree=_build_directory_tree([file.path for file in files]),
            files=tuple(files),
        )
        if estimate_token_count(candidate.render()) <= token_budget:
            return candidate
        files.pop()
    minimal = RepoMap(root=repo_map.root, directory_tree="", files=())
    if estimate_token_count(minimal.render()) > token_budget:
        raise ValueError("Token budget is too small for the minimum Repo Map header.")
    return minimal


def build_repo_map(
    repository_root: str | Path,
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    excluded_paths: tuple[Path, ...] = (),
) -> RepoMap:
    """Scan a repository and build a token-budgeted Repo Map."""
    root = Path(repository_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")
    if token_budget < 1:
        raise ValueError("Token budget must be at least 1.")
    summaries = [
        FileSummary(
            path=relative_path.as_posix(),
            language=_language_for(path),
            summary=_summarize_file(path, _language_for(path)),
        )
        for path, relative_path in _iter_repository_files(root, excluded_paths)
    ]
    repo_map = RepoMap(
        root=root.name,
        directory_tree=_build_directory_tree([file.path for file in summaries]),
        files=tuple(summaries),
    )
    return _fit_token_budget(repo_map, token_budget)


class RepoMapCache:
    """Persist and reuse Repo Maps while repository metadata is unchanged."""

    def __init__(self, cache_directory: str | Path) -> None:
        self._cache_directory = Path(cache_directory).resolve(strict=False)
        self._cache_file = self._cache_directory / CACHE_FILENAME

    def get_or_build(
        self,
        repository_root: str | Path,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> RepoMap:
        """Load a valid cached map or generate and save a new one."""
        root = Path(repository_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Repository directory does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {root}")
        if token_budget < 1:
            raise ValueError("Token budget must be at least 1.")

        excluded_paths = (self._cache_directory,) if _is_within(self._cache_directory, root) else ()
        fingerprint = _map_fingerprint(root, excluded_paths)
        revision = _git_revision(root)
        changed_files = _git_changed_files(root)

        if self._cache_file.is_file():
            try:
                cached = json.loads(self._cache_file.read_text(encoding="utf-8"))
                if (
                    cached.get("root_path") == str(root)
                    and cached.get("fingerprint") == fingerprint
                    and cached.get("git_revision") == revision
                    and cached.get("token_budget") == token_budget
                    and changed_files == set()
                ):
                    cached_map = cached["map"]
                    return RepoMap(
                        root=cached_map["root"],
                        directory_tree=cached_map["directory_tree"],
                        files=tuple(FileSummary(**item) for item in cached_map["files"]),
                    )
            except (OSError, ValueError, KeyError, TypeError):
                pass

        repo_map = build_repo_map(
            root,
            token_budget=token_budget,
            excluded_paths=excluded_paths,
        )
        payload = {
            "root_path": str(root),
            "fingerprint": fingerprint,
            "git_revision": revision,
            "token_budget": token_budget,
            "map": {
                "root": repo_map.root,
                "directory_tree": repo_map.directory_tree,
                "files": [asdict(file) for file in repo_map.files],
            },
        }
        self._cache_directory.mkdir(parents=True, exist_ok=True)
        temporary_file = self._cache_file.with_suffix(".tmp")
        temporary_file.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        temporary_file.replace(self._cache_file)
        return repo_map
