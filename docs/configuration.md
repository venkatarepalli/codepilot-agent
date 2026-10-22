# Configuration

All options, with defaults, are listed in the README's [Configuration reference](../README.md#configuration-reference) and in [`.env.example`](../.env.example).

Notes:

- Providers: only `openai` and `google` are supported by the model factory. `ANTHROPIC_API_KEY` is accepted by settings validation but has no model implementation.
- `CODEPILOT_MAX_TEST_RETRIES`, `CODEPILOT_STATE_DIR` and `CODEPILOT_CHROMA_DIR` are parsed by `Settings` but **not used**: the retry limit is fixed at 3 (`CoderAgent.MAX_ATTEMPTS`) and local state lives in `.codepilot/` under the working directory.
- Process environment variables take precedence over `.env`.
