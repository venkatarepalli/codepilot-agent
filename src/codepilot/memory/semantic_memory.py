"""Persistent semantic memory for lessons learned from coding tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_COLLECTION_NAME = "codepilot_task_lessons"
DEFAULT_TOP_K = 3


@dataclass(frozen=True, slots=True)
class TaskLesson:
    """A reusable lesson learned from a coding task."""

    lesson_id: str
    repository: str
    category: str
    problem: str
    resolution: str
    outcome: str


@dataclass(frozen=True, slots=True)
class LessonRetrievalResult:
    """A lesson returned by semantic similarity search."""

    lesson: TaskLesson
    distance: float


def _lesson_document(lesson: TaskLesson) -> str:
    """Build the text used for semantic similarity search."""
    return (
        f"Repository: {lesson.repository}\n"
        f"Category: {lesson.category}\n"
        f"Problem: {lesson.problem}\n"
        f"Resolution: {lesson.resolution}\n"
        f"Outcome: {lesson.outcome}"
    )


class SemanticMemory:
    """Store and retrieve lessons using a ChromaDB-compatible collection.

    The collection is injected into the constructor, allowing unit tests
    to run without installing ChromaDB or downloading an embedding model.
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
    ) -> SemanticMemory:
        """Create semantic memory backed by persistent ChromaDB storage."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "Semantic memory requires ChromaDB. "
                "Install it with `python -m pip install chromadb`."
            ) from exc

        client = chromadb.PersistentClient(path=str(persist_directory))
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        memory = cls(collection, top_k=top_k)

        # Keep the client alive for the lifetime of this object.
        memory._client = client

        return memory

    def store_lesson(self, lesson: TaskLesson) -> None:
        """Insert or update a lesson in persistent semantic storage."""
        if not lesson.lesson_id.strip():
            raise ValueError("lesson_id must not be empty.")

        if not lesson.repository.strip():
            raise ValueError("repository must not be empty.")

        if not lesson.category.strip():
            raise ValueError("category must not be empty.")

        if not lesson.problem.strip():
            raise ValueError("problem must not be empty.")

        if not lesson.resolution.strip():
            raise ValueError("resolution must not be empty.")

        if not lesson.outcome.strip():
            raise ValueError("outcome must not be empty.")

        self._collection.upsert(
            ids=[lesson.lesson_id],
            documents=[_lesson_document(lesson)],
            metadatas=[
                {
                    "lesson_id": lesson.lesson_id,
                    "repository": lesson.repository,
                    "category": lesson.category,
                    "problem": lesson.problem,
                    "resolution": lesson.resolution,
                    "outcome": lesson.outcome,
                }
            ],
        )

    def retrieve_lessons(
        self,
        query: str,
        *,
        repository: str | None = None,
        category: str | None = None,
        top_k: int | None = None,
    ) -> list[LessonRetrievalResult]:
        """Retrieve lessons relevant to a query.

        Optional repository and category filters restrict the search
        to matching lesson metadata.
        """
        limit = self._top_k if top_k is None else top_k

        if limit < 1:
            raise ValueError("top_k must be at least 1.")

        if not query or not query.strip():
            return []

        filters: dict[str, Any] = {}

        if repository is not None:
            filters["repository"] = repository

        if category is not None:
            filters["category"] = category

        where: dict[str, Any] | None = None

        if len(filters) == 1:
            where = filters
        elif len(filters) > 1:
            where = {"$and": [{key: value} for key, value in filters.items()]}

        query_arguments: dict[str, Any] = {
            "query_texts": [query.strip()],
            "n_results": limit,
            "include": ["metadatas", "distances"],
        }

        if where is not None:
            query_arguments["where"] = where

        response = self._collection.query(**query_arguments)

        ids_by_query = response.get("ids") or [[]]
        metadata_by_query = response.get("metadatas") or [[]]
        distances_by_query = response.get("distances") or [[]]

        ids = ids_by_query[0] if ids_by_query else []
        metadatas = metadata_by_query[0] if metadata_by_query else []
        distances = distances_by_query[0] if distances_by_query else []

        results: list[LessonRetrievalResult] = []

        for index, lesson_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else None

            if not metadata:
                continue

            try:
                lesson = TaskLesson(
                    lesson_id=str(metadata.get("lesson_id", lesson_id)),
                    repository=str(metadata["repository"]),
                    category=str(metadata["category"]),
                    problem=str(metadata["problem"]),
                    resolution=str(metadata["resolution"]),
                    outcome=str(metadata["outcome"]),
                )
                distance = float(distances[index])
            except (KeyError, IndexError, TypeError, ValueError):
                continue

            results.append(
                LessonRetrievalResult(
                    lesson=lesson,
                    distance=distance,
                )
            )

        return results
