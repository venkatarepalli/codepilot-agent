"""Environment-driven configuration for CodePilot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when CodePilot configuration is invalid."""


def _positive_int(name: str, default: int) -> int:
    """Read a positive integer from the environment."""
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer; received {raw_value!r}.") from exc

    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero.")

    return value


def _non_negative_int(name: str, default: int) -> int:
    """Read a non-negative integer from the environment."""
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer; received {raw_value!r}.") from exc

    if value < 0:
        raise ConfigurationError(f"{name} must not be negative.")

    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings for CodePilot."""

    llm_provider: str
    llm_model: str
    openai_api_key: str | None
    anthropic_api_key: str | None
    google_api_key: str | None

    github_token: str | None
    github_repository_owner: str | None
    github_repository_name: str | None
    github_default_branch: str

    poll_interval_minutes: int
    complexity_threshold: int
    repo_map_token_budget: int
    repo_map_top_k: int
    max_test_retries: int

    state_dir: Path
    chroma_dir: Path

    @classmethod
    def from_env(cls, *, dotenv_path: Path | None = None) -> Settings:
        """Load settings from the process environment and an optional .env file.

        Loading is opt-in so unit tests and library callers are not affected by a
        developer's local .env file. Existing process variables take precedence.
        """
        if dotenv_path is not None:
            load_dotenv(dotenv_path=dotenv_path, override=False)
        settings = cls(
            llm_provider=os.getenv("LLM_PROVIDER", "").strip().lower(),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            google_api_key=os.getenv("GOOGLE_API_KEY") or None,
            github_token=os.getenv("GITHUB_TOKEN") or None,
            github_repository_owner=os.getenv("GITHUB_REPOSITORY_OWNER") or None,
            github_repository_name=os.getenv("GITHUB_REPOSITORY_NAME") or None,
            github_default_branch=os.getenv("GITHUB_DEFAULT_BRANCH", "main").strip(),
            poll_interval_minutes=_positive_int("CODEPILOT_POLL_INTERVAL_MINUTES", 5),
            complexity_threshold=_positive_int("CODEPILOT_COMPLEXITY_THRESHOLD", 7),
            repo_map_token_budget=_positive_int("CODEPILOT_REPO_MAP_TOKEN_BUDGET", 4000),
            repo_map_top_k=_positive_int("CODEPILOT_REPO_MAP_TOP_K", 10),
            max_test_retries=_non_negative_int("CODEPILOT_MAX_TEST_RETRIES", 3),
            state_dir=Path(os.getenv("CODEPILOT_STATE_DIR", ".codepilot")),
            chroma_dir=Path(os.getenv("CODEPILOT_CHROMA_DIR", ".codepilot/chroma")),
        )
        settings.validate_runtime_values()
        return settings

    def validate_runtime_values(self) -> None:
        """Validate settings that must be usable regardless of integration state."""
        if not self.github_default_branch:
            raise ConfigurationError("GITHUB_DEFAULT_BRANCH must not be empty.")

        if self.complexity_threshold > 10:
            raise ConfigurationError("CODEPILOT_COMPLEXITY_THRESHOLD must be between 1 and 10.")

    def validate_llm_configuration(self) -> None:
        """Validate LLM settings when an LLM-backed operation is requested."""
        if not self.llm_provider:
            raise ConfigurationError("LLM_PROVIDER is required for LLM operations.")

        if not self.llm_model:
            raise ConfigurationError("LLM_MODEL is required for LLM operations.")

        provider_keys = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
        }

        if self.llm_provider not in provider_keys:
            supported = ", ".join(provider_keys)
            raise ConfigurationError(
                f"Unsupported LLM_PROVIDER {self.llm_provider!r}. Supported providers: {supported}."
            )

        if not provider_keys[self.llm_provider]:
            raise ConfigurationError(f"The API key for provider {self.llm_provider!r} is missing.")

    def validate_github_configuration(self) -> None:
        """Validate GitHub settings when GitHub operations are requested."""
        required = {
            "GITHUB_TOKEN": self.github_token,
            "GITHUB_REPOSITORY_OWNER": self.github_repository_owner,
            "GITHUB_REPOSITORY_NAME": self.github_repository_name,
        }

        missing = [name for name, value in required.items() if not value]

        if missing:
            raise ConfigurationError("Missing GitHub configuration: " + ", ".join(missing) + ".")
