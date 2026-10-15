"""Tests for DeepAgents adapter retry and quota error handling."""

from unittest.mock import Mock, patch

import pytest

from codepilot.agents.deepagents_coder_adapter import DeepAgentsCoderAdapter


@pytest.fixture
def adapter():
    agent = Mock()
    return DeepAgentsCoderAdapter(agent, Mock()), agent


def test_openai_credit_exhaustion_stops_without_retry(adapter):
    coder_adapter, agent = adapter
    agent.invoke.side_effect = Exception(
        "OpenAIRateLimitError: 429 insufficient_quota "
        "credit_balance_exhausted: You have no credits remaining."
    )

    with pytest.raises(RuntimeError, match="credit balance"):
        coder_adapter._invoke("Test prompt")

    agent.invoke.assert_called_once()


def test_openai_insufficient_quota_stops_without_retry(adapter):
    coder_adapter, agent = adapter
    agent.invoke.side_effect = Exception("OpenAIRateLimitError: 429 insufficient_quota")

    with pytest.raises(RuntimeError, match="quota"):
        coder_adapter._invoke("Test prompt")

    agent.invoke.assert_called_once()


def test_google_quota_exhaustion_stops_without_retry(adapter):
    coder_adapter, agent = adapter
    agent.invoke.side_effect = Exception(
        "Quota exceeded for metric: generate_content_free_tier_requests"
    )

    with pytest.raises(RuntimeError, match="quota"):
        coder_adapter._invoke("Test prompt")

    agent.invoke.assert_called_once()


def test_transient_error_retries_then_succeeds(adapter):
    coder_adapter, agent = adapter
    agent.invoke.side_effect = [
        Exception("503 Service Unavailable"),
        {"messages": [{"role": "assistant", "content": "Plan ready."}]},
    ]

    with patch("codepilot.agents.deepagents_coder_adapter.time.sleep") as sleep_mock:
        with patch(
            "codepilot.agents.deepagents_coder_adapter.random.uniform",
            return_value=0,
        ):
            result = coder_adapter._invoke("Test prompt")

    assert result == "Plan ready."
    assert agent.invoke.call_count == 2
    sleep_mock.assert_called_once_with(2.0)
