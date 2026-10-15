"""Tests for provider model construction without network calls."""

from unittest.mock import MagicMock

import pytest

from codepilot.config.model_factory import create_chat_model


def test_google_provider_constructs_chat_model(monkeypatch):
    settings = MagicMock()
    settings.llm_provider = "google"
    settings.llm_model = "gemini-2.5-flash"
    settings.google_api_key = "test-key"
    settings.validate_llm_configuration.return_value = None

    expected = object()
    constructor = MagicMock(return_value=expected)
    monkeypatch.setattr(
        "codepilot.config.model_factory.ChatGoogleGenerativeAI",
        constructor,
    )

    result = create_chat_model(settings)

    assert result is expected
    constructor.assert_called_once_with(
        model="gemini-2.5-flash",
        google_api_key="test-key",
        timeout=60,
        max_retries=0,
    )
    settings.validate_llm_configuration.assert_called_once_with()


def test_missing_model_name_is_rejected(monkeypatch):
    settings = MagicMock()
    settings.llm_provider = "google"
    settings.llm_model = ""
    settings.google_api_key = "test-key"
    settings.validate_llm_configuration.return_value = None

    monkeypatch.setattr(
        "codepilot.config.model_factory.load_dotenv",
        lambda **kwargs: None,
    )

    with pytest.raises(ValueError, match="LLM_MODEL must be set"):
        create_chat_model(settings)


def test_missing_google_key_is_rejected(monkeypatch):
    settings = MagicMock()
    settings.llm_provider = "google"
    settings.llm_model = "gemini-2.5-flash"
    settings.google_api_key = ""
    settings.validate_llm_configuration.return_value = None

    monkeypatch.setattr(
        "codepilot.config.model_factory.load_dotenv",
        lambda **kwargs: None,
    )

    with pytest.raises(ValueError, match="GOOGLE_API_KEY is required"):
        create_chat_model(settings)


def test_unsupported_provider_is_rejected(monkeypatch):
    settings = MagicMock()
    settings.llm_provider = "anthropic"
    settings.llm_model = "some-model"
    settings.google_api_key = "test-key"
    settings.validate_llm_configuration.return_value = None

    monkeypatch.setattr(
        "codepilot.config.model_factory.load_dotenv",
        lambda **kwargs: None,
    )

    with pytest.raises(ValueError, match="not supported by create_chat_model"):
        create_chat_model(settings)


def test_openai_provider_constructs_chat_model(monkeypatch):
    settings = MagicMock()
    settings.llm_provider = "openai"
    settings.llm_model = "gpt-4.1-mini"
    settings.openai_api_key = "test-openai-key"
    settings.validate_llm_configuration.return_value = None

    expected = object()
    constructor = MagicMock(return_value=expected)

    monkeypatch.setattr(
        "codepilot.config.model_factory.ChatOpenAI",
        constructor,
    )
    monkeypatch.setattr(
        "codepilot.config.model_factory.load_dotenv",
        lambda **kwargs: None,
    )

    result = create_chat_model(settings)

    assert result is expected
    constructor.assert_called_once_with(
        model="gpt-4.1-mini",
        api_key="test-openai-key",
        timeout=60,
        max_retries=0,
    )
    settings.validate_llm_configuration.assert_called_once_with()


def test_missing_openai_key_is_rejected(monkeypatch):
    settings = MagicMock()
    settings.llm_provider = "openai"
    settings.llm_model = "gpt-4.1-mini"
    settings.openai_api_key = ""
    settings.validate_llm_configuration.return_value = None

    monkeypatch.setattr(
        "codepilot.config.model_factory.load_dotenv",
        lambda **kwargs: None,
    )

    with pytest.raises(
        ValueError,
        match="OPENAI_API_KEY is required",
    ):
        create_chat_model(settings)
