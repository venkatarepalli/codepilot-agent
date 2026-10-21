"""Assemble bounded, prioritized context for CodePilot coding tasks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from codepilot.context.keyword_retriever import RetrievalResult, retrieve_relevant_files
from codepilot.context.repo_map import RepoMap


@dataclass(frozen=True)
class ContextFile:
    """A selected repository file summary and the reason it was selected."""

    path: str
    summary: str
    source: str
    score: float = 0.0
    distance: float | None = None


@dataclass(frozen=True)
class ContextPackage:
    """The bounded context supplied to a coding agent."""

    repo_map_text: str
    files: tuple[ContextFile, ...]
    estimated_tokens: int
    token_budget: int
    truncated: bool

    def to_prompt_text(self) -> str:
        """Render the package as plain text suitable for an agent prompt."""
        sections: list[str] = []
        if self.repo_map_text:
            sections.append("## Repository map\n" + self.repo_map_text)
        if self.files:
            rendered = ["## Relevant file summaries"]
            for item in self.files:
                rendered.append(f"### {item.path} (source: {item.source})\n{item.summary}")
            sections.append("\n\n".join(rendered))
        return "\n\n".join(sections)


def _estimate_tokens(text: str) -> int:
    """Conservative lightweight estimate; avoids a tokenizer dependency."""
    if not text:
        return 0
    # Four characters per token is a common rough estimate; round upward.
    return (len(text) + 3) // 4


def _safe_path(path: str) -> bool:
    """Reject absolute/traversal paths and sensitive-looking file names."""
    normalized = path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        return False
    lowered = pure.name.lower()
    if lowered == ".env" or lowered.startswith(".env."):
        return False
    if lowered.endswith((".secret", ".pem", ".key")):
        return False
    if "credentials" in lowered:
        return False
    return True


def _summary_from_result(result: Any) -> tuple[str, str, float, float | None]:
    """Support keyword and semantic retriever result shapes."""
    if isinstance(result, RetrievalResult):
        # Keyword RetrievalResult uses `.file`; keep the explicit type path
        # so its score and matched keywords remain available.
        return (
            result.file.path,
            result.file.summary,
            float(result.score),
            None,
        )

    # Keyword retrieval calls the field `file`; semantic retrieval calls it
    # `file_summary`. Support both interfaces.
    file_summary = getattr(result, "file_summary", None)
    if file_summary is None:
        file_summary = getattr(result, "file", None)
    if file_summary is not None:
        return (
            str(file_summary.path),
            str(file_summary.summary),
            float(getattr(result, "score", 0.0)),
            (float(result.distance) if getattr(result, "distance", None) is not None else None),
        )

    # Also accept a FileSummary directly, which simplifies integrations and tests.
    if hasattr(result, "path") and hasattr(result, "summary"):
        return str(result.path), str(result.summary), 0.0, None

    raise TypeError(f"Unsupported retrieval result: {type(result).__name__}")


class ContextAssembler:
    """Merge Repo Map and retrieval results into a token-bounded context package.

    `semantic_retriever` is intentionally optional: keyword retrieval remains
    available when ChromaDB or its embedding model is unavailable.
    """

    def __init__(
        self,
        *,
        token_budget: int = 4000,
        semantic_retriever: Any | None = None,
        keyword_top_k: int = 10,
        semantic_top_k: int = 10,
        retrieval_strategy: str = "hybrid",
    ) -> None:
        if token_budget < 1:
            raise ValueError("token_budget must be at least 1")
        if keyword_top_k < 1 or semantic_top_k < 1:
            raise ValueError("retrieval top_k values must be at least 1")
        if retrieval_strategy not in {"keyword", "semantic", "hybrid"}:
            raise ValueError(
                "retrieval_strategy must be 'keyword', 'semantic', or 'hybrid'"
            )

        self.token_budget = token_budget
        self.semantic_retriever = semantic_retriever
        self.keyword_top_k = keyword_top_k
        self.semantic_top_k = semantic_top_k
        self.retrieval_strategy = retrieval_strategy


    def assemble(
        self,
        repo_map: RepoMap,
        issue_title: str,
        issue_description: str = "",
    ) -> ContextPackage:
        """Build context using repo map, keyword matches, and optional semantics."""
        repo_map_text = self._render_repo_map(repo_map)
        repo_map_cost = _estimate_tokens(repo_map_text)
        selected: dict[str, ContextFile] = {}
        truncated = repo_map_cost > self.token_budget

        if self.retrieval_strategy in {"keyword", "hybrid"}:
            keyword_results = retrieve_relevant_files(
                repo_map,
                issue_title,
                issue_description,
                top_k=self.keyword_top_k,
            )
            for result in keyword_results:
                path, summary, score, distance = _summary_from_result(result)
                if _safe_path(path):
                    self._merge(
                        selected,
                        ContextFile(path, summary, "keyword", score, distance),
                    )

        if self.retrieval_strategy in {"semantic", "hybrid"}:
            semantic_results: Iterable[Any] = ()

            if self.semantic_retriever is not None:
                try:
                    semantic_results = self.semantic_retriever.retrieve(
                        repo_map,
                        issue_title,
                        issue_description,
                        top_k=self.semantic_top_k,
                    )
                except Exception:
                    # Preserve the existing resilience when semantic search fails.
                    semantic_results = ()

            for result in semantic_results:
                try:
                    path, summary, score, distance = _summary_from_result(result)
                except (TypeError, AttributeError, ValueError):
                    continue

                if _safe_path(path):
                    self._merge(
                        selected,
                        ContextFile(path, summary, "semantic", score, distance),
                    )

        # Keyword score and semantic distance are not directly comparable. Keep
        # keyword-first order, then semantic-only files in their retrieval order.
        ordered = sorted(
            selected.values(),
            key=lambda item: (
                0 if "keyword" in item.source else 1,
                -item.score if "keyword" in item.source else 0,
            ),
        )

        included: list[ContextFile] = []
        used = repo_map_cost
        for item in ordered:
            rendered = f"### {item.path}\n{item.summary}"
            cost = _estimate_tokens(rendered)
            if cost <= max(0, self.token_budget - used):
                included.append(item)
                used += cost
            else:
                truncated = True

        # If the map alone exceeds the budget, preserve a truncated map rather
        # than returning no context. Token estimate may be approximate.
        if repo_map_cost > self.token_budget:
            max_chars = self.token_budget * 4
            repo_map_text = repo_map_text[:max_chars]
            used = _estimate_tokens(repo_map_text)
            included = []
        return ContextPackage(
            repo_map_text=repo_map_text,
            files=tuple(included),
            estimated_tokens=min(self.token_budget, used),
            token_budget=self.token_budget,
            truncated=truncated,
        )

    @staticmethod
    def _render_repo_map(repo_map: RepoMap) -> str:
        """Render the RepoMap while supporting its documented data structure."""
        for attribute in ("render", "to_text", "format"):
            method = getattr(repo_map, attribute, None)
            if callable(method):
                value = method()
                if isinstance(value, str):
                    return value
        for attribute in ("text", "content", "rendered_text"):
            value = getattr(repo_map, attribute, None)
            if isinstance(value, str):
                return value

        # Current RepoMap stores the directory tree and summaries as attributes.
        tree = getattr(repo_map, "directory_tree", "")
        summaries = getattr(repo_map, "file_summaries", None)
        if summaries is None:
            summaries = getattr(repo_map, "files", ())
        pieces: list[str] = []
        if tree:
            pieces.append(str(tree))
        if isinstance(summaries, dict):
            for path, summary in summaries.items():
                summary_text = getattr(summary, "summary", summary)
                pieces.append(f"{path}: {summary_text}")
        else:
            for summary in summaries:
                if hasattr(summary, "path") and hasattr(summary, "summary"):
                    pieces.append(f"{summary.path}: {summary.summary}")
        return "\n".join(pieces)

    @staticmethod
    def _merge(target: dict[str, ContextFile], candidate: ContextFile) -> None:
        """Deduplicate paths while retaining both retrieval sources when possible."""
        existing = target.get(candidate.path)
        if existing is None:
            target[candidate.path] = candidate
            return
        sources = set(existing.source.split("+"))
        sources.add(candidate.source)
        # Prefer keyword score for ordering, but retain semantic distance if present.
        target[candidate.path] = ContextFile(
            path=existing.path,
            summary=existing.summary or candidate.summary,
            source="+".join(sorted(sources, key=lambda s: (s != "keyword", s))),
            score=max(existing.score, candidate.score),
            distance=(existing.distance if existing.distance is not None else candidate.distance),
        )
