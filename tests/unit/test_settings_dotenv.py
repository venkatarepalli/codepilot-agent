"""Tests for explicit .env loading by CodePilot settings."""

from codepilot.config.settings import Settings


def test_from_env_loads_values_from_explicit_dotenv_path(tmp_path, monkeypatch):
    """Settings load .env values only when a path is explicitly provided."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=openai",
                "LLM_MODEL=gpt-4.1-mini",
                "OPENAI_API_KEY=dotenv-test-key",
                "GITHUB_TOKEN=dotenv-test-token",
                "GITHUB_REPOSITORY_OWNER=example-owner",
                "GITHUB_REPOSITORY_NAME=example-repo",
                "GITHUB_DEFAULT_BRANCH=main",
            ]
        ),
        encoding="utf-8",
    )
    for name in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GITHUB_TOKEN",
        "GITHUB_REPOSITORY_OWNER",
        "GITHUB_REPOSITORY_NAME",
        "GITHUB_DEFAULT_BRANCH",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env(dotenv_path=env_file)

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4.1-mini"
    assert settings.openai_api_key == "dotenv-test-key"
    assert settings.github_token == "dotenv-test-token"
    assert settings.github_repository_owner == "example-owner"
    assert settings.github_repository_name == "example-repo"


def test_process_environment_takes_precedence_over_dotenv(tmp_path, monkeypatch):
    """Existing process variables take precedence over .env values."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=openai\nLLM_MODEL=gpt-4.1-mini\nOPENAI_API_KEY=dotenv-key\n"
        "GITHUB_REPOSITORY_NAME=from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_REPOSITORY_NAME", "from-process")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "process-test-key")

    settings = Settings.from_env(dotenv_path=env_file)

    assert settings.github_repository_name == "from-process"
    assert settings.openai_api_key == "process-test-key"
