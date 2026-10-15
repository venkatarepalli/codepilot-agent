"""Semantic retrieval of repository files using ChromaDB embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codepilot.context.repo_map import FileSummary, RepoMap

DEFAULT_TOP_K = 10
DEFAULT_COLLECTION_NAME = "codepilot_repo_files"
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
class SemanticRetrievalResult:
    """A file summary returned by semantic similarity search."""

    file: FileSummary
    distance: float


def _is_safe_path(relative_path: str) -> bool:
    """Reject sensitive filenames and files under generated directories."""
    parts = relative_path.replace("\\", "/").split("/")
    if any(part.lower() in EXCLUDED_DIRECTORY_NAMES for part in parts):
        return False

    for part in parts:
        lowered = part.lower()
        if lowered == ".env" or lowered.startswith(".env."):
            return False
        if lowered.endswith((".secret", ".pem", ".key")):
            return False
        if "credentials" in lowered:
            return False
    return True


def _embedding_document(file_summary: FileSummary) -> str:
    """Build the text embedded for one file without reading its source."""
    return (
        f"Path: {file_summary.path}\n"
        f"Language: {file_summary.language}\n"
        f"Summary: {file_summary.summary}"
    )


class SemanticFileRetriever:
    """Index Repo Map summaries in a persistent ChromaDB collection.

    ChromaDB and its embedding function are injected by the caller so this
    module remains testable without downloading a model or requiring network
    access. The collection must implement the standard Chroma collection API.
    """

    def __init__(
        self,
        collection: Any,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        self._collection = collection
        self._top_k = top_k

    @classmethod
    def from_persistent_directory(
        cls,
        persist_directory: str | Path,
        embedding_function: Any,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
    ) -> SemanticFileRetriever:
        """Create a persistent ChromaDB-backed retriever."""
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "Semantic retrieval requires ChromaDB. Install it with `pip install chromadb`."
            ) from exc

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        client = chromadb.PersistentClient(path=str(persist_directory))
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        retriever = cls(collection, top_k=top_k)
        # Keep the client alive for the lifetime of the retriever.
        retriever._client = client
        return retriever

    def index_repo_map(self, repo_map: RepoMap) -> int:
        """Upsert safe file summaries into the vector index.

        Returns the number of safe file summaries submitted for indexing.
        Existing entries for the same relative path are updated.
        """
        safe_files = [
            file_summary for file_summary in repo_map.files if _is_safe_path(file_summary.path)
        ]
        if not safe_files:
            return 0

        ids = [file_summary.path for file_summary in safe_files]
        documents = [_embedding_document(file_summary) for file_summary in safe_files]
        metadatas = [
            {
                "path": file_summary.path,
                "language": file_summary.language,
                "summary": file_summary.summary,
                "repository": repo_map.root,
            }
            for file_summary in safe_files
        ]
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(safe_files)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[SemanticRetrievalResult]:
        """Return nearest indexed file summaries for a natural-language query."""
        limit = self._top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be at least 1.")
        if not query or not query.strip():
            return []

        response = self._collection.query(
            query_texts=[query.strip()],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        ids_by_query = response.get("ids") or [[]]
        metadata_by_query = response.get("metadatas") or [[]]
        distances_by_query = response.get("distances") or [[]]

        results: list[SemanticRetrievalResult] = []
        ids = ids_by_query[0] if ids_by_query else []
        metadatas = metadata_by_query[0] if metadata_by_query else []
        distances = distances_by_query[0] if distances_by_query else []

        for index, item_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else None
            if not metadata:
                continue
            path = str(metadata.get("path", item_id))
            if not _is_safe_path(path):
                continue
            try:
                file_summary = FileSummary(
                    path=path,
                    language=str(metadata.get("language", "Other")),
                    summary=str(metadata.get("summary", "")),
                )
                distance = float(distances[index])
            except (IndexError, TypeError, ValueError):
                continue
            results.append(SemanticRetrievalResult(file=file_summary, distance=distance))
        return results
