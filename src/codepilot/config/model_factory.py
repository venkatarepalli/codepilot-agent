"""Model construction for configured LLM providers."""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from codepilot.config.settings import Settings


def create_chat_model(settings: Settings | None = None) -> Any:
    """Create a chat model for the configured LLM provider.

    Args:
        settings: Optional application settings. When omitted, settings
            are loaded from the environment.

    Returns:
        A configured LangChain chat model.

    Raises:
        ValueError: If the provider, model name, or required API key
            is missing or invalid.
    """
    load_dotenv(override=False)

    configuration = settings or Settings.from_env()
    configuration.validate_llm_configuration()

    provider = (configuration.llm_provider or "").strip().lower()
    model_name = (configuration.llm_model or "").strip()

    if not model_name:
        raise ValueError("LLM_MODEL must be set to a supported model name.")

    if provider == "google":
        api_key = getattr(configuration, "google_api_key", None)

        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER is 'google'.")

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            timeout=60,
            max_retries=0,
        )

    if provider == "openai":
        api_key = getattr(configuration, "openai_api_key", None)

        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'.")

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            timeout=60,
            max_retries=0,
        )
    raise ValueError(
        f"LLM provider {provider!r} is not supported by create_chat_model. "
        "Supported providers are 'google' and 'openai'."
    )
