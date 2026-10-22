"""Semantic retrieval of repository files using ChromaDB embeddings."""

from __future__ import annotations

import hashlib
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


DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".sql",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
    ".ps1",
}


def _chunk_text(
    content: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into bounded, overlapping character chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    text = content.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


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

    def index_source_files(
        self,
        repository_root: str | Path,
        repo_map: RepoMap,
    ) -> int:
        """Index safe source-file chunks in the persistent vector collection."""
        root = Path(repository_root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {root}")

        # Discover source files independently of the token-budgeted Repo Map.
        # The Repo Map supplies summaries; it does not limit source indexing.
        map_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and _is_safe_path(path.relative_to(root).as_posix())
            and path.suffix.lower() in SOURCE_EXTENSIONS
        }

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        unreadable_paths: set[str] = set()

        for relative_path in sorted(map_paths):
            source_path = (root / relative_path).resolve()

            # Prevent paths in the map from escaping the repository.
            if not source_path.is_relative_to(root):
                continue

            if source_path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue

            try:
                content = source_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                unreadable_paths.add(relative_path)
                continue

            file_summary = next(
                (item for item in repo_map.files if item.path == relative_path),
                None,
            )
            language = file_summary.language if file_summary else "Other"
            summary = file_summary.summary if file_summary else ""

            for chunk_index, chunk in enumerate(_chunk_text(content)):
                chunk_id = hashlib.sha256(
                    f"{root}:{relative_path}:{chunk_index}".encode()
                ).hexdigest()

                ids.append(chunk_id)

                documents.append(
                    f"Path: {relative_path}\n"
                    f"Language: {language}\n"
                    f"Summary: {summary}\n"
                    f"Chunk {chunk_index + 1}:\n{chunk}"
                )

                metadatas.append(
                    {
                        "path": relative_path,
                        "language": language,
                        "summary": summary,
                        "chunk_index": chunk_index,
                        "repository": str(root),
                        "document_type": "source_chunk",
                    }
                )

        # Remove obsolete source chunks without touching Repo Map summaries.
        existing = self._collection.get(
            where={"document_type": "source_chunk"},
            include=["metadatas"],
        )

        existing_ids = existing.get("ids", [])
        existing_metadatas = existing.get("metadatas", [])

        desired_ids = set(ids)
        stale_ids = []

        for existing_id, metadata in zip(
            existing_ids,
            existing_metadatas,
            strict=False,
        ):
            if not metadata or metadata.get("repository") != str(root):
                continue

            if existing_id not in desired_ids and metadata.get("path") not in unreadable_paths:
                stale_ids.append(existing_id)

        if stale_ids:
            self._collection.delete(ids=stale_ids)

        if not ids:
            return 0

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)

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
        seen_paths: set[str] = set()
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

            if path in seen_paths:
                continue

            seen_paths.add(path)
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
