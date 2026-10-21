"""Application bootstrap for CodePilot."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.config.model_factory import create_chat_model
from codepilot.context.context_assembler import ContextAssembler
from codepilot.context.repo_map import RepoMapCache
from codepilot.context.semantic_retriever import SemanticFileRetriever
from codepilot.github.merge_monitor import PullRequestMergeMonitor
from codepilot.github.pr_workflow import PullRequestWorkflow
from codepilot.memory.episodic_memory import EpisodicMemory
from codepilot.memory.semantic_memory import SemanticMemory
from codepilot.sandbox.coder_workspace import CoderWorkspace
from codepilot.sandbox.executor import SandboxExecutor
from codepilot.sandbox.policy import SandboxPolicy
from codepilot.sandbox.workspace import WorkspaceManager


def create_orchestrator(
    *,
    tools: list[Any] | None = None,
    workspace_root: Path | None = None,
    test_command: list[str] | None = None,
    repository: str | None = None,
    pr_workflow: PullRequestWorkflow | None = None,
    approval_callback: Callable[[str, Mapping[str, Any]], bool] | None = None,
) -> CodePilotOrchestrator:
    """Create and configure the CodePilot orchestrator and its dependencies."""
    model = create_chat_model()

    resolved_workspace_root = (
        workspace_root if workspace_root is not None else Path.cwd() / ".codepilot" / "workspaces"
    )
    resolved_workspace_root = resolved_workspace_root.resolve()
    resolved_workspace_root.mkdir(parents=True, exist_ok=True)

    manager = WorkspaceManager(resolved_workspace_root)
    executor = SandboxExecutor(
        SandboxPolicy(resolved_workspace_root),
        timeout_seconds=120.0,
        approval_callback=approval_callback,
    )
    coder_workspace = CoderWorkspace(manager, executor)

    resolved_test_command = (
        test_command if test_command is not None else [sys.executable, "-m", "pytest", "-q"]
    )

    # Episodic memory: task history for the current application session.
    episodic_memory = EpisodicMemory()

    # Semantic memory: persistent lessons learned across application runs.
    memory_root = Path.cwd() / ".codepilot" / "memory"
    memory_root.mkdir(parents=True, exist_ok=True)

    semantic_memory = SemanticMemory.from_persistent_directory(
        persist_directory=memory_root / "semantic",
        embedding_function=DefaultEmbeddingFunction(),
    )

    # Context engineering: Repo Map, keyword retrieval, and semantic retrieval.
    repo_map_token_budget = 4000
    repo_map_top_k = 10

    repo_map_cache = RepoMapCache(cache_directory=memory_root / "repo_map_cache")

    semantic_retriever = SemanticFileRetriever.from_persistent_directory(
        persist_directory=memory_root / "repo_files",
        embedding_function=DefaultEmbeddingFunction(),
        top_k=repo_map_top_k,
    )

    context_assembler = ContextAssembler(
        token_budget=repo_map_token_budget,
        semantic_retriever=semantic_retriever,
        keyword_top_k=repo_map_top_k,
        semantic_top_k=repo_map_top_k,
    )

    def repo_map_provider(repository_root: Path):
        repo_map = repo_map_cache.get_or_build(
            repository_root,
            token_budget=repo_map_token_budget,
        )
        semantic_retriever.index_repo_map(repo_map)
        semantic_retriever.index_source_files(repository_root, repo_map)
        return repo_map

    orchestrator = CodePilotOrchestrator(
        model=model,
        tools=tools,
        coder_workspace=coder_workspace,
        test_command=resolved_test_command,
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        context_assembler=context_assembler,
        repo_map_provider=repo_map_provider,
        repository=repository,
        pr_workflow=pr_workflow,
        approval_callback=approval_callback,
    )

    # Monitor published PRs only when a workflow is configured.
    if pr_workflow is not None:
        orchestrator.attach_merge_monitor(PullRequestMergeMonitor(orchestrator=orchestrator))

    return orchestrator
