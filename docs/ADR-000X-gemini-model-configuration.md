# ADR: Gemini model configuration

- **Status:** Implemented. Live demo runs used `LLM_PROVIDER=openai` (`gpt-4.1-mini`); Gemini remains a supported alternative
- **Decision:** Construct the configured Google chat model in a dedicated model factory and inject it into the existing CodePilot Orchestrator through an application bootstrap function.
- **Configuration:** `LLM_PROVIDER=google`, `LLM_MODEL=<model name>`, and `GOOGLE_API_KEY=<secret>` in the local `.env`.
- **Secret handling:** `.env` must remain untracked; `.env.example` must contain no real credentials.
- **Testing:** Unit tests mock model construction and Orchestrator construction; they do not make network requests. `scripts/smoke_test_gemini.py` performs one real API request through DeepAgents.
- **Scope:** The current model factory supports Google only. Other providers should be added deliberately with provider-specific tests.
- **Limitations:** Successful unit tests do not prove the key is valid, the model is available, or a live API request succeeds. The smoke test verifies request/response connectivity but does not validate coding-task execution.
