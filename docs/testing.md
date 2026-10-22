# Testing

```powershell
python -m pytest -q            # about 615 tests; offline, no API keys or network
python -m ruff check src tests
python scripts/smoke_test_coder.py   # optional live-model check (costs API usage)
```

## What is tested

- **Unit tests** (`tests/unit/`): every component, named after the requirement it covers (e.g. `test_merge_conflict_detection.py`, `test_session_summary.py`, `test_pr_conventions.py`, `test_nemo_guardrails.py`).
- **Real-framework tests**, with no mocks of the framework itself:
  - real DeepAgents graphs driven by a scripted fake chat model: auto-summarization installed on every agent, live tool-call streaming from root and subagents, `read_file` on seeded VFS files
  - the real LangChain `GitHubToolkit`: only read-only tools bound to the LLM
  - the real NeMo Guardrails engine
- **Integration tests** (`tests/test_orchestrator_live_integration.py`, `test_runner*.py`): the delegation contract, recovery attempts, runner ↔ orchestrator ↔ PR workflow wiring.
- **Isolation:** `tests/conftest.py` redirects episodic session summaries to a temporary directory, so tests never touch real project state.
- **Live end-to-end runs** against `venkatarepalli/codepilot-smoke-test` are recorded in `docs/assets/` (see the README's Demo section).

On Windows, one symlink test is skipped unless Developer Mode is enabled.
