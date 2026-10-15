"""Tests for PAT-based GitHub client initialization."""

from unittest.mock import MagicMock

import pytest

from codepilot.config.settings import ConfigurationError, Settings
from codepilot.github import client as github_client_module
from codepilot.github.client import GitHubClient


def make_settings(*, token: str | None = "test-token") -> Settings:
    """Create settings for a GitHub client test."""
    return Settings(
        llm_provider="openai",
        llm_model="test-model",
        openai_api_key=None,
        anthropic_api_key=None,
        google_api_key=None,
        github_token=token,
        github_repository_owner="example-owner",
        github_repository_name="example-repo",
        github_default_branch="main",
        poll_interval_minutes=5,
        complexity_threshold=7,
        repo_map_token_budget=4000,
        repo_map_top_k=10,
        max_test_retries=3,
        state_dir=".codepilot",
        chroma_dir=".codepilot/chroma",
    )


def test_initializes_github_client_with_pat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the configured PAT through PyGithub without GitHub App credentials."""
    github = MagicMock()
    repository = MagicMock()
    repository.default_branch = "main"
    github.get_repo.return_value = repository
    github_constructor = MagicMock(return_value=github)
    auth_token = object()

    monkeypatch.setattr(github_client_module, "Github", github_constructor)
    monkeypatch.setattr(github_client_module.Auth, "Token", lambda token: auth_token)

    toolkit = MagicMock()
    client = GitHubClient(make_settings(), toolkit=toolkit)

    github_constructor.assert_called_once_with(auth=auth_token)
    github.get_repo.assert_called_once_with("example-owner/example-repo")
    assert client.repository is repository
    assert client.toolkit is toolkit
    assert client._api_wrapper.github is github
    assert client._api_wrapper.github_repository == "example-owner/example-repo"


def test_requires_pat_when_constructing_real_client() -> None:
    """A real client must not proceed without a PAT."""
    with pytest.raises(ConfigurationError, match="GITHUB_TOKEN"):
        GitHubClient(make_settings(token=None))
