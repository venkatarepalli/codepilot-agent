"""Tests for context assembly."""

from dataclasses import dataclass

import pytest

from codepilot.context.context_assembler import ContextAssembler, _safe_path
from codepilot.context.repo_map import FileSummary, RepoMap


@dataclass
class FakeResult:
    file_summary: FileSummary
    score: float = 1.0
    distance: float | None = None


class FakeSemanticRetriever:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error

    def retrieve(self, repo_map, issue_title, issue_description="", *, top_k=10):
        if self.error:
            raise self.error
        return self.results[:top_k]


def make_repo_map():
    files = (
        FileSummary(
            path="src/client.py",
            language="Python",
            summary="GitHub client polls issues and retrieves repository data.",
        ),
        FileSummary(
            path="src/models.py",
            language="Python",
            summary="Issue models define title, labels, and issue number.",
        ),
        FileSummary(
            path="src/secrets.key",
            language="Other",
            summary="should never appear",
        ),
        FileSummary(
            path=".env",
            language="Other",
            summary="token=do-not-show",
        ),
    )
    return RepoMap(
        root="fake-repository",
        directory_tree="src/\n  client.py\n  models.py",
        files=files,
    )


def test_assembles_repo_map_and_keyword_retrieval():
    package = ContextAssembler(token_budget=1000).assemble(make_repo_map(), "GitHub issue polling")
    assert "Repository: fake-repository" in package.repo_map_text
    assert any(item.path == "src/client.py" for item in package.files)
    assert all(_safe_path(item.path) for item in package.files)
    assert package.estimated_tokens <= package.token_budget


def test_semantic_results_are_merged_and_deduplicated():
    semantic = FakeSemanticRetriever(
        [
            FakeResult(
                FileSummary("src/client.py", "Python", "semantic client summary"),
                0.2,
                0.1,
            ),
            FakeResult(
                FileSummary("src/new.py", "Python", "A new relevant module."),
                0.8,
                0.2,
            ),
        ]
    )
    package = ContextAssembler(token_budget=1000, semantic_retriever=semantic).assemble(
        make_repo_map(), "GitHub issue polling"
    )
    paths = [item.path for item in package.files]
    assert len(paths) == len(set(paths))
    client = next(item for item in package.files if item.path == "src/client.py")
    assert "keyword" in client.source
    assert "semantic" in client.source
    assert "src/new.py" in paths


def test_semantic_failure_falls_back_to_keyword_results():
    package = ContextAssembler(
        token_budget=1000,
        semantic_retriever=FakeSemanticRetriever(error=RuntimeError("Chroma unavailable")),
    ).assemble(make_repo_map(), "GitHub issue polling")
    assert package.files
    assert any(item.source.startswith("keyword") for item in package.files)

def test_keyword_strategy_does_not_call_semantic_retriever():
    semantic = FakeSemanticRetriever(
        [FakeResult(FileSummary("src/semantic.py", "Python", "Semantic result."))]
    )
    assembler = ContextAssembler(
        token_budget=1000,
        semantic_retriever=semantic,
        retrieval_strategy="keyword",
    )

    package = assembler.assemble(make_repo_map(), "GitHub issue polling")

    assert any(item.source == "keyword" for item in package.files)
    assert not any(item.path == "src/semantic.py" for item in package.files)


def test_semantic_strategy_uses_semantic_results_only():
    semantic = FakeSemanticRetriever(
        [FakeResult(FileSummary("src/semantic.py", "Python", "Semantic result."))]
    )
    assembler = ContextAssembler(
        token_budget=1000,
        semantic_retriever=semantic,
        retrieval_strategy="semantic",
    )

    package = assembler.assemble(make_repo_map(), "unrelated query")

    assert any(item.path == "src/semantic.py" for item in package.files)
    assert all("keyword" not in item.source for item in package.files)


def test_hybrid_strategy_merges_keyword_and_semantic_results():
    semantic = FakeSemanticRetriever(
        [FakeResult(FileSummary("src/semantic.py", "Python", "Semantic result."))]
    )
    assembler = ContextAssembler(
        token_budget=1000,
        semantic_retriever=semantic,
        retrieval_strategy="hybrid",
    )

    package = assembler.assemble(make_repo_map(), "GitHub issue polling")

    assert any("keyword" in item.source for item in package.files)
    assert any(item.path == "src/semantic.py" for item in package.files)


def test_rejects_unknown_retrieval_strategy():
    with pytest.raises(ValueError, match="retrieval_strategy"):
        ContextAssembler(retrieval_strategy="unknown")

def test_context_budget_truncates_additional_summaries():
    package = ContextAssembler(token_budget=30).assemble(make_repo_map(), "GitHub issue polling")
    assert package.truncated
    assert package.estimated_tokens <= 30


def test_rejects_invalid_budget():
    with pytest.raises(ValueError, match="token_budget"):
        ContextAssembler(token_budget=0)


def test_rejects_unsafe_retrieval_paths():
    assert not _safe_path("../outside.py")
    assert not _safe_path(".env")
    assert not _safe_path("config/credentials.json")
    assert _safe_path("src/client.py")
