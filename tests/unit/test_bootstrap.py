"""Tests for application bootstrap wiring."""

import sys
from unittest.mock import MagicMock

from codepilot.config import bootstrap


def test_create_orchestrator_uses_configured_model(monkeypatch):
    model = object()
    expected_orchestrator = object()
    orchestrator_constructor = MagicMock(return_value=expected_orchestrator)
    monkeypatch.setattr(bootstrap, "create_chat_model", lambda: model)
    monkeypatch.setattr(bootstrap, "CodePilotOrchestrator", orchestrator_constructor)

    result = bootstrap.create_orchestrator()

    assert result is expected_orchestrator
    call_kwargs = orchestrator_constructor.call_args.kwargs

    assert call_kwargs["model"] is model
    assert call_kwargs["tools"] is None
    assert call_kwargs["coder_workspace"] is not None
    assert call_kwargs["test_command"] == [
        sys.executable,
        "-m",
        "pytest",
        "-q",
    ]


def test_create_orchestrator_wires_repo_context_components(monkeypatch, tmp_path):
    """The orchestrator receives Repo Map and context assembly components."""
    model = object()
    orchestrator_constructor = MagicMock(return_value=object())

    monkeypatch.setattr(bootstrap, "create_chat_model", lambda: model)
    monkeypatch.setattr(
        bootstrap,
        "CodePilotOrchestrator",
        orchestrator_constructor,
    )

    bootstrap.create_orchestrator(workspace_root=tmp_path / "workspace")

    call_kwargs = orchestrator_constructor.call_args.kwargs

    assert call_kwargs["context_assembler"] is not None
    assert callable(call_kwargs["repo_map_provider"])


def test_create_orchestrator_passes_tools(monkeypatch):
    model = object()
    tools = [object()]
    orchestrator_constructor = MagicMock(return_value=object())
    monkeypatch.setattr(bootstrap, "create_chat_model", lambda: model)
    monkeypatch.setattr(bootstrap, "CodePilotOrchestrator", orchestrator_constructor)

    bootstrap.create_orchestrator(tools=tools)

    call_kwargs = orchestrator_constructor.call_args.kwargs

    assert call_kwargs["model"] is model
    assert call_kwargs["tools"] is tools
    assert call_kwargs["coder_workspace"] is not None
    assert call_kwargs["test_command"] == [
        sys.executable,
        "-m",
        "pytest",
        "-q",
    ]


def test_create_orchestrator_passes_approval_callback_to_sandbox_executor(monkeypatch, tmp_path):
    """The interactive HITL callback is wired into guarded command execution."""
    approval_callback = MagicMock(return_value=True)
    executor_factory = MagicMock(return_value=object())
    monkeypatch.setattr(bootstrap, "create_chat_model", lambda: object())
    monkeypatch.setattr(bootstrap, "SandboxExecutor", executor_factory)
    monkeypatch.setattr(bootstrap, "WorkspaceManager", lambda _root: object())
    monkeypatch.setattr(bootstrap, "CoderWorkspace", lambda *_args: object())
    monkeypatch.setattr(bootstrap, "EpisodicMemory", lambda: object())
    semantic_memory_factory = MagicMock()
    semantic_memory_factory.from_persistent_directory.return_value = object()
    monkeypatch.setattr(bootstrap, "SemanticMemory", semantic_memory_factory)
    monkeypatch.setattr(
        bootstrap,
        "SemanticFileRetriever",
        MagicMock(),
    )
    monkeypatch.setattr(bootstrap, "CodePilotOrchestrator", MagicMock(return_value=object()))

    bootstrap.create_orchestrator(
        workspace_root=tmp_path / "workspace",
        approval_callback=approval_callback,
    )

    assert executor_factory.call_args.kwargs["approval_callback"] is approval_callback
