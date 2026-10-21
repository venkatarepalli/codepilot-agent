"""Component 2: auto-summarization must be enabled on the orchestrator.

DeepAgents 0.7 has no ``summarization=`` kwarg; it installs
SummarizationMiddleware itself. These tests prove that against the real
library rather than trusting the docstring.
"""

import deepagents.graph as deepagents_graph
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from codepilot.agents.orchestrator import (
    ORCHESTRATOR_BACKEND_CONFIG,
    ORCHESTRATOR_SUBAGENTS,
    CodePilotOrchestrator,
)


def _model() -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))


def test_summarization_is_configured_on() -> None:
    assert ORCHESTRATOR_BACKEND_CONFIG["summarization"] is True
    assert CodePilotOrchestrator(model=_model()).summarization is True


def test_real_deep_agent_installs_summarization_for_root_and_every_subagent(
    monkeypatch,
) -> None:
    created_for = []
    original = deepagents_graph.create_summarization_middleware

    def spy(model, backend, **kwargs):
        created_for.append(model)
        return original(model, backend, **kwargs)

    monkeypatch.setattr(deepagents_graph, "create_summarization_middleware", spy)
    model = _model()

    CodePilotOrchestrator(model=model)

    # root + general-purpose subagent + each custom subagent
    assert len(created_for) == 2 + len(ORCHESTRATOR_SUBAGENTS)
    assert all(m is model for m in created_for)
