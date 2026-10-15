"""Tests for CodePilot semantic file retrieval."""

from pathlib import Path
from typing import Any

import pytest

from codepilot.context.repo_map import FileSummary, RepoMap
from codepilot.context.semantic_retriever import (
    SemanticFileRetriever,
    SemanticRetrievalResult,
)


class FakeCollection:
    """Minimal fake implementing the Chroma collection API used here."""

    def __init__(self) -> None:
        self.documents: dict[str, str] = {}
        self.metadatas: dict[str, dict[str, str]] = {}
        self.query_response: dict[str, Any] = {
            "ids": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []

    def upsert(
        self, *, ids: list[str], documents: list[str], metadatas: list[dict[str, str]]
    ) -> None:
        self.upsert_calls.append({"ids": ids, "documents": documents, "metadatas": metadatas})
        for item_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            self.documents[item_id] = document
            self.metadatas[item_id] = metadata

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.query_calls.append(kwargs)
        return self.query_response


def make_repo_map(*files: FileSummary) -> RepoMap:
    return RepoMap(root="sample-repository", directory_tree="", files=tuple(files))


def test_indexes_file_summaries_with_path_and_metadata() -> None:
    collection = FakeCollection()
    retriever = SemanticFileRetriever(collection)
    repo_map = make_repo_map(
        FileSummary("src/github/client.py", "Python", "GitHub API client."),
        FileSummary("src/context/repo_map.py", "Python", "Repository map."),
    )

    count = retriever.index_repo_map(repo_map)

    assert count == 2
    assert set(collection.documents) == {
        "src/github/client.py",
        "src/context/repo_map.py",
    }
    assert "Path: src/github/client.py" in collection.documents["src/github/client.py"]
    assert collection.metadatas["src/github/client.py"]["language"] == "Python"


def test_skips_sensitive_and_excluded_paths_when_indexing() -> None:
    collection = FakeCollection()
    retriever = SemanticFileRetriever(collection)
    repo_map = make_repo_map(
        FileSummary(".env", "Other", "secret settings"),
        FileSummary("certs/server.pem", "Other", "certificate"),
        FileSummary("src/.venv/package.py", "Python", "third-party package"),
        FileSummary("src/github/client.py", "Python", "GitHub API client"),
    )

    count = retriever.index_repo_map(repo_map)

    assert count == 1
    assert list(collection.documents) == ["src/github/client.py"]


def test_returns_semantic_results_from_collection_response() -> None:
    collection = FakeCollection()
    collection.query_response = {
        "ids": [["src/github/client.py", "src/auth/service.py"]],
        "metadatas": [
            [
                {
                    "path": "src/github/client.py",
                    "language": "Python",
                    "summary": "GitHub API client.",
                },
                {
                    "path": "src/auth/service.py",
                    "language": "Python",
                    "summary": "Authentication service.",
                },
            ]
        ],
        "distances": [[0.08, 0.21]],
    }
    retriever = SemanticFileRetriever(collection, top_k=10)

    results = retriever.retrieve("Why does GitHub authentication fail?")

    assert [result.file.path for result in results] == [
        "src/github/client.py",
        "src/auth/service.py",
    ]
    assert results[0].distance == pytest.approx(0.08)
    assert all(isinstance(item, SemanticRetrievalResult) for item in results)


def test_passes_query_and_top_k_to_collection() -> None:
    collection = FakeCollection()
    collection.query_response = {
        "ids": [["src/app.py"]],
        "metadatas": [[{"path": "src/app.py", "language": "Python", "summary": "App."}]],
        "distances": [[0.1]],
    }
    retriever = SemanticFileRetriever(collection, top_k=5)

    retriever.retrieve("application fails", top_k=2)

    assert collection.query_calls[0]["query_texts"] == ["application fails"]
    assert collection.query_calls[0]["n_results"] == 2


def test_empty_query_returns_no_results_without_querying_collection() -> None:
    collection = FakeCollection()
    retriever = SemanticFileRetriever(collection)

    assert retriever.retrieve("   ") == []
    assert collection.query_calls == []


def test_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        SemanticFileRetriever(FakeCollection(), top_k=0)


def test_rejects_invalid_per_call_top_k() -> None:
    retriever = SemanticFileRetriever(FakeCollection())

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.retrieve("test", top_k=0)


def test_skips_unsafe_paths_in_query_results() -> None:
    collection = FakeCollection()
    collection.query_response = {
        "ids": [[".env", "src/safe.py"]],
        "metadatas": [
            [
                {"path": ".env", "language": "Other", "summary": "Secrets"},
                {"path": "src/safe.py", "language": "Python", "summary": "Safe file"},
            ]
        ],
        "distances": [[0.01, 0.1]],
    }
    results = SemanticFileRetriever(collection).retrieve("configuration")

    assert [item.file.path for item in results] == ["src/safe.py"]


def test_handles_empty_collection_response() -> None:
    retriever = SemanticFileRetriever(FakeCollection())

    assert retriever.retrieve("no matches") == []


def test_from_persistent_directory_reports_missing_chromadb_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If ChromaDB is installed, this test is not applicable to the missing
    # dependency path. The main retrieval tests use an injected fake collection.
    import builtins

    original_import = builtins.__import__

    def import_without_chromadb(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "chromadb":
            raise ImportError("blocked by test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_chromadb)

    with pytest.raises(ImportError, match="requires ChromaDB"):
        SemanticFileRetriever.from_persistent_directory(
            tmp_path,
            embedding_function=object(),
        )
