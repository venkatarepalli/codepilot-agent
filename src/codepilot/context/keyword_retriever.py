"""Keyword-based retrieval of files relevant to a GitHub issue."""

from __future__ import annotations

import re
from dataclasses import dataclass

from codepilot.context.repo_map import FileSummary, RepoMap

DEFAULT_TOP_K = 10
# Do not include periods in tokens: otherwise "client.py" stays one token
# and fails to match the issue term "client".
TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_+#-]*")
STOP_WORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "issue",
    "please",
    "need",
    "make",
    "work",
    "works",
    "working",
}
SENSITIVE_NAME_PARTS = (".secret", ".pem", ".key")
EXCLUDED_DIRECTORY_NAMES = {
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


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A ranked file summary returned by keyword retrieval."""

    file: FileSummary
    score: float
    matched_keywords: tuple[str, ...]


def _tokens(text: str) -> set[str]:
    """Extract normalized, meaningful terms from text."""
    tokens = {token.lower() for token in TOKEN_PATTERN.findall(text)}
    return {
        token for token in tokens if token not in STOP_WORDS and (len(token) > 1 or token.isdigit())
    }


def _is_safe_path(relative_path: str) -> bool:
    """Reject sensitive filenames and paths inside generated directories."""
    parts = relative_path.replace("\\", "/").split("/")
    if any(part.lower() in EXCLUDED_DIRECTORY_NAMES for part in parts):
        return False

    for part in parts:
        lowered = part.lower()
        if lowered == ".env" or lowered.startswith(".env."):
            return False
        if any(lowered.endswith(suffix) for suffix in SENSITIVE_NAME_PARTS):
            return False
        if "credentials" in lowered:
            return False
    return True


def retrieve_relevant_files(
    repo_map: RepoMap,
    issue_title: str,
    issue_description: str = "",
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievalResult]:
    """Rank repository files against an issue using keyword overlap.

    Paths receive a stronger weight than summaries because filenames and
    directory names often identify a component directly. Ties are resolved
    deterministically by relative path. Files with no matching terms are not
    returned. This is lexical retrieval, not semantic similarity.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_terms = _tokens(f"{issue_title or ''} {issue_description or ''}")
    if not query_terms:
        return []

    results: list[RetrievalResult] = []
    for file_summary in repo_map.files:
        if not _is_safe_path(file_summary.path):
            continue

        path_terms = _tokens(file_summary.path.replace("/", " "))
        summary_terms = _tokens(file_summary.summary)
        path_matches = query_terms & path_terms
        summary_matches = query_terms & summary_terms
        matched = path_matches | summary_matches

        if not matched:
            continue

        score = (2.0 * len(path_matches)) + len(summary_matches)
        results.append(
            RetrievalResult(
                file=file_summary,
                score=score,
                matched_keywords=tuple(sorted(matched)),
            )
        )

    results.sort(key=lambda result: (-result.score, result.file.path.lower()))
    return results[:top_k]
