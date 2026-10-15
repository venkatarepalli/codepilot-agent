# Configuration

## Overview

CodePilot uses environment variables for runtime configuration.

- `.env.example` documents the available configuration variables.
- `.env` is intended for local development and must never contain committed credentials.
- The `Settings` class in `src/codepilot/config/settings.py` loads and validates configuration.
- Loading settings does not require GitHub credentials or an LLM API key. Credentials are validated when the corresponding integration is requested.

## Local setup

1. Copy `.env.example` to `.env`.
2. Set the values required for the integrations you intend to use.
3. Keep `.env` out of version control.
4. Start CodePilot using the documented application entry point.

**Current implementation note:** `Settings.from_env()` reads the process environment using Python's `os.getenv()`. It does not automatically parse `.env` files. Values from `.env` must be loaded into the process environment by a separate mechanism before they can be read.

## Environment variables

### LLM provider

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | Empty | LLM provider identifier. Supported identifiers are `openai`, `anthropic`, and `google`. |
| `LLM_MODEL` | Empty | Model identifier for the selected provider. |
| `OPENAI_API_KEY` | Unset | API key for OpenAI. |
| `ANTHROPIC_API_KEY` | Unset | API key for Anthropic. |
| `GOOGLE_API_KEY` | Unset | API key for Google. |

LLM settings are validated when `validate_llm_configuration()` is called.

The selected provider and model must be configured, and the corresponding provider API key must be present. Other providers' API keys are not required.

### GitHub integration

| Variable | Default | Description |
|---|---|---|
| `GITHUB_TOKEN` | Unset | Token used for authenticated GitHub operations. |
| `GITHUB_REPOSITORY_OWNER` | Unset | GitHub repository owner or organization. |
| `GITHUB_REPOSITORY_NAME` | Unset | Repository name. |
| `GITHUB_DEFAULT_BRANCH` | `main` | Default branch name. |

GitHub settings are validated when `validate_github_configuration()` is called. The token, repository owner, and repository name must be present.

### Runtime settings

| Variable | Default | Validation |
|---|---:|---|
| `CODEPILOT_POLL_INTERVAL_MINUTES` | `5` | Positive integer |
| `CODEPILOT_COMPLEXITY_THRESHOLD` | `7` | Integer from 1 to 10 |
| `CODEPILOT_REPO_MAP_TOKEN_BUDGET` | `4000` | Positive integer |
| `CODEPILOT_REPO_MAP_TOP_K` | `10` | Positive integer |
| `CODEPILOT_MAX_TEST_RETRIES` | `3` | Non-negative integer |

Invalid numeric values raise `ConfigurationError` during settings loading.

### Storage settings

| Variable | Default | Description |
|---|---|---|
| `CODEPILOT_STATE_DIR` | `.codepilot` | Directory intended for CodePilot state. |
| `CODEPILOT_CHROMA_DIR` | `.codepilot/chroma` | Directory intended for ChromaDB storage. |

Paths are represented as `pathlib.Path` objects. Loading settings does not create these directories.

## Validation behaviour

The configuration implementation currently provides:

- Default values for runtime and storage settings.
- Case normalization for `LLM_PROVIDER`.
- Validation of numeric ranges.
- Separate validation methods for LLM and GitHub integrations.
- Clear errors for missing required integration settings.
- No requirement for real credentials during configuration unit tests.

Configuration loading and validation do not make network requests.

## Security

- Never commit `.env` or real credentials.
- Never hardcode API keys or GitHub tokens in source code or tests.
- Use fake credentials in automated tests.
- Do not print secrets in application logs or error messages.
- Give GitHub tokens only the permissions needed for the intended operations.

## Current limitations

- `.env` parsing is not implemented by `Settings.from_env()`.
- The LLM provider identifiers are configuration-level checks; they do not establish that the corresponding provider integration has been implemented or tested.
- GitHub configuration validation checks that required values are present. It does not authenticate with GitHub or verify repository access.
- Storage directories are not created automatically.
- The complexity threshold is currently validated against a 1–10 range; its operational meaning depends on the complexity-scoring implementation.

These limitations will be updated as the relevant features are implemented and tested.