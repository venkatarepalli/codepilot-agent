"""Tests for CodePilot keyword-based file retrieval."""

import pytest

from codepilot.context.keyword_retriever import (
    RetrievalResult,
    retrieve_relevant_files,
)
from codepilot.context.repo_map import FileSummary, RepoMap


def make_repo_map(*files: FileSummary) -> RepoMap:
    """Build a small RepoMap for retrieval tests."""
    return RepoMap(
        root="sample-repository",
        directory_tree="",
        files=tuple(files),
    )


def test_retrieves_file_from_issue_title_and_description() -> None:
    repo_map = make_repo_map(
        FileSummary(
            path="src/codepilot/github/client.py",
            language="Python",
            summary="GitHub API client and issue operations.",
        ),
        FileSummary(
            path="src/codepilot/context/repo_map.py",
            language="Python",
            summary="Repository map generation and file summaries.",
        ),
    )

    results = retrieve_relevant_files(
        repo_map,
        "GitHub client issue retrieval",
        "The GitHub API client fails when loading issues.",
    )

    assert results
    assert results[0].file.path == "src/codepilot/github/client.py"
    assert "github" in results[0].matched_keywords
    assert results[0].score > 0


def test_path_matches_receive_higher_weight_than_summary_only_matches() -> None:
    repo_map = make_repo_map(
        FileSummary(
            path="src/payment/client.py",
            language="Python",
            summary="Handles HTTP requests.",
        ),
        FileSummary(
            path="src/general/helpers.py",
            language="Python",
            summary="Contains payment client utilities.",
        ),
    )

    results = retrieve_relevant_files(repo_map, "payment client")

    assert results[0].file.path == "src/payment/client.py"
    assert results[0].score > results[1].score


def test_results_are_sorted_deterministically_for_equal_scores() -> None:
    repo_map = make_repo_map(
        FileSummary("src/zebra.py", "Python", "Handles cache invalidation."),
        FileSummary("src/alpha.py", "Python", "Handles cache invalidation."),
    )

    results = retrieve_relevant_files(repo_map, "cache invalidation")

    assert [result.file.path for result in results] == [
        "src/alpha.py",
        "src/zebra.py",
    ]


def test_respects_top_k() -> None:
    repo_map = make_repo_map(
        FileSummary("src/cache_one.py", "Python", "Cache implementation."),
        FileSummary("src/cache_two.py", "Python", "Cache invalidation."),
        FileSummary("src/cache_three.py", "Python", "Cache storage."),
    )

    results = retrieve_relevant_files(repo_map, "cache", top_k=2)

    assert len(results) == 2


def test_returns_no_results_for_empty_or_stopword_only_query() -> None:
    repo_map = make_repo_map(
        FileSummary("src/app.py", "Python", "Application entry point."),
    )

    assert retrieve_relevant_files(repo_map, "") == []
    assert retrieve_relevant_files(repo_map, "the and please") == []


def test_returns_no_results_when_no_terms_match() -> None:
    repo_map = make_repo_map(
        FileSummary("src/app.py", "Python", "Application entry point."),
    )

    assert retrieve_relevant_files(repo_map, "database migration") == []


def test_rejects_invalid_top_k() -> None:
    repo_map = make_repo_map(
        FileSummary("src/app.py", "Python", "Application entry point."),
    )

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retrieve_relevant_files(repo_map, "application", top_k=0)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        ".env",
        ".env.production",
        "config/database_credentials.json",
        "certs/server.pem",
        "keys/private.key",
        "src/.codepilot/cache.py",
        "src/.venv/package.py",
    ],
)
def test_skips_sensitive_and_excluded_paths(unsafe_path: str) -> None:
    repo_map = make_repo_map(
        FileSummary(unsafe_path, "Python", "GitHub issue client."),
        FileSummary(
            "src/github/client.py",
            "Python",
            "GitHub issue client.",
        ),
    )

    results = retrieve_relevant_files(repo_map, "GitHub issue client")

    assert all(result.file.path != unsafe_path for result in results)
    assert any(result.file.path == "src/github/client.py" for result in results)


def test_returns_retrieval_result_type() -> None:
    repo_map = make_repo_map(
        FileSummary("src/github/client.py", "Python", "GitHub API client."),
    )

    results = retrieve_relevant_files(repo_map, "GitHub client")

    assert isinstance(results[0], RetrievalResult)
    assert isinstance(results[0].matched_keywords, tuple)
