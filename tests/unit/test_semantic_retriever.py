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
        self.metadatas: dict[str, dict[str, Any]] = {}
        self.query_response: dict[str, Any] = {
            "ids": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []

    def upsert(
        self, *, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]
    ) -> None:
        self.upsert_calls.append({"ids": ids, "documents": documents, "metadatas": metadatas})
        for item_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            self.documents[item_id] = document
            self.metadatas[item_id] = metadata

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.query_calls.append(kwargs)
        return self.query_response

    def get(self, **kwargs: Any) -> dict[str, Any]:
        """Return stored records, optionally filtered by metadata."""
        where = kwargs.get("where", {})
        matching_ids = [
            item_id
            for item_id, metadata in self.metadatas.items()
            if all(metadata.get(key) == value for key, value in where.items())
        ]

        return {
            "ids": matching_ids,
            "metadatas": [self.metadatas[item_id] for item_id in matching_ids],
        }

    def delete(self, *, ids: list[str]) -> None:
        """Delete selected records from the fake collection."""
        for item_id in ids:
            self.documents.pop(item_id, None)
            self.metadatas.pop(item_id, None)


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


def test_index_source_files_splits_large_file_into_chunks(tmp_path: Path) -> None:
    """Source indexing stores large files as multiple searchable chunks."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    source_file = source_dir / "app.py"
    source_file.write_text("x" * 1400, encoding="utf-8")

    repo_map = make_repo_map(
        FileSummary(
            path="src/app.py",
            language="Python",
            summary="Application module",
        )
    )
    retriever = SemanticFileRetriever.__new__(SemanticFileRetriever)
    retriever._collection = FakeCollection()

    indexed_count = retriever.index_source_files(tmp_path, repo_map)

    assert indexed_count == 2

    collection = retriever._collection
    source_chunks = [
        (item_id, metadata)
        for item_id, metadata in collection.metadatas.items()
        if metadata.get("document_type") == "source_chunk"
    ]

    assert len(source_chunks) == 2
    assert {metadata["path"] for _, metadata in source_chunks} == {"src/app.py"}
    assert {metadata["chunk_index"] for _, metadata in source_chunks} == {0, 1}
    assert len({item_id for item_id, _ in source_chunks}) == 2


def test_index_source_files_removes_stale_chunks_only_for_current_repository(
    tmp_path: Path,
) -> None:
    """Stale source chunks are removed without deleting other repository data."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    source_file = source_dir / "current.py"
    source_file.write_text("print('current')", encoding="utf-8")

    other_repository = str(tmp_path / "other-repository")
    collection = FakeCollection()

    collection.documents.update(
        {
            "stale-current-repo": "obsolete source chunk",
            "summary-record": "file summary",
            "other-repo-chunk": "another repository's source chunk",
        }
    )
    collection.metadatas.update(
        {
            "stale-current-repo": {
                "path": "src/obsolete.py",
                "repository": str(tmp_path),
                "document_type": "source_chunk",
            },
            "summary-record": {
                "path": "src/obsolete.py",
                "repository": str(tmp_path),
                "document_type": "file_summary",
            },
            "other-repo-chunk": {
                "path": "src/obsolete.py",
                "repository": other_repository,
                "document_type": "source_chunk",
            },
        }
    )

    repo_map = make_repo_map(
        FileSummary(
            path="src/current.py",
            language="Python",
            summary="Current module",
        )
    )
    retriever = SemanticFileRetriever.__new__(SemanticFileRetriever)
    retriever._collection = collection

    retriever.index_source_files(tmp_path, repo_map)

    assert "stale-current-repo" not in collection.documents
    assert "summary-record" in collection.documents
    assert "other-repo-chunk" in collection.documents
    assert any(
        metadata.get("path") == "src/current.py" and metadata.get("document_type") == "source_chunk"
        for metadata in collection.metadatas.values()
    )


def test_index_source_files_preserves_chunks_for_unreadable_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing chunks survive when a source file cannot be read."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()

    current_file = source_dir / "current.py"
    current_file.write_text("print('current')", encoding="utf-8")

    unreadable_file = source_dir / "unreadable.py"
    unreadable_file.write_text("print('previous content')", encoding="utf-8")

    collection = FakeCollection()
    existing_chunk_id = "existing-unreadable-file-chunk"
    collection.documents[existing_chunk_id] = "Previously indexed source content"
    collection.metadatas[existing_chunk_id] = {
        "path": "src/unreadable.py",
        "repository": str(tmp_path.resolve()),
        "document_type": "source_chunk",
        "chunk_index": 0,
    }

    repo_map = make_repo_map(
        FileSummary(
            path="src/current.py",
            language="Python",
            summary="Current module",
        ),
        FileSummary(
            path="src/unreadable.py",
            language="Python",
            summary="Unreadable module",
        ),
    )

    original_read_text = Path.read_text

    def read_text_with_failure(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        if path.resolve() == unreadable_file.resolve():
            raise OSError("Simulated file read failure")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text_with_failure)

    retriever = SemanticFileRetriever.__new__(SemanticFileRetriever)
    retriever._collection = collection

    retriever.index_source_files(tmp_path, repo_map)

    # The existing chunk must survive the read failure.
    assert existing_chunk_id in collection.documents

    # The readable file must still be indexed.
    assert any(
        metadata.get("path") == "src/current.py" and metadata.get("document_type") == "source_chunk"
        for metadata in collection.metadatas.values()
    )


def test_retrieve_deduplicates_files_returned_by_multiple_chunks() -> None:
    """Multiple matching chunks from one file produce one file result."""
    collection = FakeCollection()
    collection.query_response = {
        "ids": [
            [
                "app-chunk-1",
                "app-chunk-2",
                "utils-chunk-1",
            ]
        ],
        "metadatas": [
            [
                {
                    "path": "src/app.py",
                    "language": "Python",
                    "summary": "Application module",
                },
                {
                    "path": "src/app.py",
                    "language": "Python",
                    "summary": "Application module",
                },
                {
                    "path": "src/utils.py",
                    "language": "Python",
                    "summary": "Utility module",
                },
            ]
        ],
        "distances": [[0.1, 0.2, 0.3]],
    }

    retriever = SemanticFileRetriever.__new__(SemanticFileRetriever)
    retriever._collection = collection
    retriever._top_k = 10

    results = retriever.retrieve("calculate total")

    result_paths = [result.file.path for result in results]

    assert result_paths == ["src/app.py", "src/utils.py"]


def test_index_source_files_includes_files_missing_from_repo_map(
    tmp_path: Path,
) -> None:
    """Source indexing includes safe files omitted from the Repo Map."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()

    mapped_file = source_dir / "mapped.py"
    mapped_file.write_text("print('mapped')", encoding="utf-8")

    unmapped_file = source_dir / "unmapped.py"
    unmapped_file.write_text(
        "def important_function():\n    return 42\n",
        encoding="utf-8",
    )

    repo_map = make_repo_map(
        FileSummary(
            path="src/mapped.py",
            language="Python",
            summary="Mapped module",
        )
    )

    retriever = SemanticFileRetriever.__new__(SemanticFileRetriever)
    retriever._collection = FakeCollection()

    indexed_count = retriever.index_source_files(tmp_path, repo_map)

    assert indexed_count >= 2

    indexed_paths = {
        metadata["path"]
        for metadata in retriever._collection.metadatas.values()
        if metadata.get("document_type") == "source_chunk"
    }

    assert "src/mapped.py" in indexed_paths
    assert "src/unmapped.py" in indexed_paths
