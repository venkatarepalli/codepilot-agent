"""Tests for CodePilot configuration loading and validation."""

import pytest

from codepilot.config.settings import ConfigurationError, Settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Ensure tests do not depend on the developer's environment."""
    names = [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GITHUB_TOKEN",
        "GITHUB_REPOSITORY_OWNER",
        "GITHUB_REPOSITORY_NAME",
        "GITHUB_DEFAULT_BRANCH",
        "CODEPILOT_POLL_INTERVAL_MINUTES",
        "CODEPILOT_COMPLEXITY_THRESHOLD",
        "CODEPILOT_REPO_MAP_TOKEN_BUDGET",
        "CODEPILOT_REPO_MAP_TOP_K",
        "CODEPILOT_MAX_TEST_RETRIES",
        "CODEPILOT_STATE_DIR",
        "CODEPILOT_CHROMA_DIR",
    ]

    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_defaults_load_without_credentials():
    settings = Settings.from_env()

    assert settings.poll_interval_minutes == 5
    assert settings.complexity_threshold == 7
    assert settings.repo_map_token_budget == 4000
    assert settings.repo_map_top_k == 10
    assert settings.max_test_retries == 3
    assert settings.github_default_branch == "main"
    assert settings.github_token is None
    assert settings.llm_provider == ""


def test_provider_is_normalized_and_api_key_is_loaded(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", " OpenAI ")
    monkeypatch.setenv("LLM_MODEL", "example-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = Settings.from_env()

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "example-model"
    assert settings.openai_api_key == "test-key"


def test_missing_llm_configuration_is_reported():
    settings = Settings.from_env()

    with pytest.raises(ConfigurationError, match="LLM_PROVIDER"):
        settings.validate_llm_configuration()


def test_missing_provider_api_key_is_reported(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "example-model")

    settings = Settings.from_env()

    with pytest.raises(ConfigurationError, match="API key"):
        settings.validate_llm_configuration()


def test_missing_github_configuration_is_reported():
    settings = Settings.from_env()

    with pytest.raises(ConfigurationError, match="GITHUB_TOKEN"):
        settings.validate_github_configuration()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CODEPILOT_POLL_INTERVAL_MINUTES", "0"),
        ("CODEPILOT_REPO_MAP_TOKEN_BUDGET", "-10"),
        ("CODEPILOT_REPO_MAP_TOP_K", "abc"),
    ],
)
def test_invalid_positive_integer_settings_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError):
        Settings.from_env()


def test_zero_test_retries_is_allowed(monkeypatch):
    monkeypatch.setenv("CODEPILOT_MAX_TEST_RETRIES", "0")

    settings = Settings.from_env()

    assert settings.max_test_retries == 0


def test_complexity_threshold_above_ten_is_rejected(monkeypatch):
    monkeypatch.setenv("CODEPILOT_COMPLEXITY_THRESHOLD", "11")

    with pytest.raises(ConfigurationError, match="between 1 and 10"):
        Settings.from_env()


def test_github_configuration_passes_when_all_values_are_present(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "example-owner")
    monkeypatch.setenv("GITHUB_REPOSITORY_NAME", "example-repo")

    settings = Settings.from_env()
    settings.validate_github_configuration()
