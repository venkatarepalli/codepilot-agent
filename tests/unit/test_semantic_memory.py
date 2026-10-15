"""Tests for semantic memory."""

from unittest.mock import patch

import pytest

from codepilot.memory.semantic_memory import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_TOP_K,
    SemanticMemory,
    TaskLesson,
)


class FakeCollection:
    """Minimal ChromaDB-compatible collection for unit tests."""

    def __init__(self):
        self.records = {}
        self.last_query = None

    def upsert(self, *, ids, documents, metadatas):
        for lesson_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            self.records[lesson_id] = {
                "document": document,
                "metadata": metadata,
            }

    def query(
        self,
        *,
        query_texts,
        n_results,
        include,
        where=None,
    ):
        self.last_query = {
            "query_texts": query_texts,
            "n_results": n_results,
            "include": include,
            "where": where,
        }

        matching_records = []

        for lesson_id, record in self.records.items():
            metadata = record["metadata"]

            if not self._matches(metadata, where):
                continue

            matching_records.append((lesson_id, record))

        matching_records = matching_records[:n_results]

        return {
            "ids": [[lesson_id for lesson_id, _ in matching_records]],
            "metadatas": [[record["metadata"] for _, record in matching_records]],
            "distances": [[0.1] * len(matching_records)],
        }

    @staticmethod
    def _matches(metadata, where):
        if where is None:
            return True

        if "$and" in where:
            return all(FakeCollection._matches(metadata, condition) for condition in where["$and"])

        return all(metadata.get(key) == value for key, value in where.items())


@pytest.fixture
def collection():
    return FakeCollection()


@pytest.fixture
def memory(collection):
    return SemanticMemory(collection)


@pytest.fixture
def lesson():
    return TaskLesson(
        lesson_id="owner-repo-issue-10",
        repository="owner/repo",
        category="bug_fix",
        problem="Tests fail when a configuration value is missing.",
        resolution="Validate the configuration before execution.",
        outcome="All tests passed.",
    )


def test_default_configuration():
    assert DEFAULT_COLLECTION_NAME == "codepilot_task_lessons"
    assert DEFAULT_TOP_K == 3


def test_store_lesson(memory, collection, lesson):
    memory.store_lesson(lesson)

    assert "owner-repo-issue-10" in collection.records

    stored = collection.records["owner-repo-issue-10"]

    assert stored["metadata"]["repository"] == "owner/repo"
    assert stored["metadata"]["category"] == "bug_fix"
    assert "Validate the configuration" in stored["document"]


def test_store_lesson_updates_existing_id(memory, collection, lesson):
    memory.store_lesson(lesson)

    updated = TaskLesson(
        lesson_id=lesson.lesson_id,
        repository=lesson.repository,
        category=lesson.category,
        problem=lesson.problem,
        resolution="Add a default configuration value.",
        outcome="All tests passed.",
    )

    memory.store_lesson(updated)

    assert len(collection.records) == 1
    assert (
        collection.records[lesson.lesson_id]["metadata"]["resolution"]
        == "Add a default configuration value."
    )


def test_retrieve_lessons(memory, lesson):
    memory.store_lesson(lesson)

    results = memory.retrieve_lessons("missing configuration")

    assert len(results) == 1
    assert results[0].lesson == lesson
    assert results[0].distance == 0.1


def test_retrieve_lessons_filters_by_repository(memory, lesson):
    memory.store_lesson(lesson)

    results = memory.retrieve_lessons(
        "configuration",
        repository="another/repo",
    )

    assert results == []


def test_retrieve_lessons_filters_by_category(memory, lesson):
    memory.store_lesson(lesson)

    results = memory.retrieve_lessons(
        "configuration",
        category="documentation",
    )

    assert results == []


def test_retrieve_lessons_applies_both_filters(memory, collection, lesson):
    memory.store_lesson(lesson)

    results = memory.retrieve_lessons(
        "configuration",
        repository="owner/repo",
        category="bug_fix",
    )

    assert len(results) == 1
    assert collection.last_query["where"] == {
        "$and": [
            {"repository": "owner/repo"},
            {"category": "bug_fix"},
        ]
    }


def test_retrieve_lessons_uses_default_top_k(memory, collection):
    memory.retrieve_lessons("configuration")

    assert collection.last_query["n_results"] == 3


def test_retrieve_lessons_accepts_custom_top_k(memory, collection):
    memory.retrieve_lessons("configuration", top_k=5)

    assert collection.last_query["n_results"] == 5


def test_retrieve_lessons_returns_empty_for_blank_query(memory, collection):
    assert memory.retrieve_lessons("   ") == []
    assert collection.last_query is None


@pytest.mark.parametrize(
    "field",
    [
        "lesson_id",
        "repository",
        "category",
        "problem",
        "resolution",
        "outcome",
    ],
)
def test_store_lesson_rejects_empty_required_fields(memory, lesson, field):
    values = {
        "lesson_id": lesson.lesson_id,
        "repository": lesson.repository,
        "category": lesson.category,
        "problem": lesson.problem,
        "resolution": lesson.resolution,
        "outcome": lesson.outcome,
    }
    values[field] = "   "

    invalid_lesson = TaskLesson(**values)

    with pytest.raises(ValueError, match=f"{field} must not be empty"):
        memory.store_lesson(invalid_lesson)


def test_constructor_rejects_invalid_top_k(collection):
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        SemanticMemory(collection, top_k=0)


def test_retrieve_rejects_invalid_top_k(memory):
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        memory.retrieve_lessons("configuration", top_k=0)


def test_missing_chromadb_has_helpful_error():
    with patch.dict("sys.modules", {"chromadb": None}):
        with pytest.raises(ImportError, match="requires ChromaDB"):
            SemanticMemory.from_persistent_directory(
                ".codepilot/chroma",
                embedding_function=object(),
            )
