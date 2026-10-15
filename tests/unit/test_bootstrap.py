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
