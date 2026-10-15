"""Application bootstrap for CodePilot."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.config.model_factory import create_chat_model
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
    pr_workflow: PullRequestWorkflow | None = None,
) -> CodePilotOrchestrator:
    """Create and configure the CodePilot orchestrator and its dependencies."""
    model = create_chat_model()

    resolved_workspace_root = (
        workspace_root
        if workspace_root is not None
        else Path.cwd() / ".codepilot" / "workspaces"
    )
    resolved_workspace_root = resolved_workspace_root.resolve()
    resolved_workspace_root.mkdir(parents=True, exist_ok=True)

    manager = WorkspaceManager(resolved_workspace_root)
    executor = SandboxExecutor(
        SandboxPolicy(resolved_workspace_root),
        timeout_seconds=120.0,
    )
    coder_workspace = CoderWorkspace(manager, executor)

    resolved_test_command = (
        test_command
        if test_command is not None
        else [sys.executable, "-m", "pytest", "-q"]
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

    orchestrator = CodePilotOrchestrator(
        model=model,
        tools=tools,
        coder_workspace=coder_workspace,
        test_command=resolved_test_command,
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        pr_workflow=pr_workflow,
    )

    # Monitor published PRs only when a workflow is configured.
    if pr_workflow is not None:
        orchestrator.attach_merge_monitor(
            PullRequestMergeMonitor(orchestrator=orchestrator)
        )

    return orchestrator