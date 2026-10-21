# CodePilot Requirement Matrix

This matrix maps assignment themes to the current source tree and available tests. A component implementation or unit test does **not** by itself prove complete assignment-level end-to-end compliance. Live GitHub/LLM workflows require separate verification in a disposable repository.

| ID | Requirement | Implementation reference | Test reference | Current evidence status |
|---|---|---|---|---|
| CP-001 | DeepAgents Orchestrator | `src/codepilot/agents/orchestrator.py` | `tests/unit/test_orchestrator.py`, `tests/unit/test_orchestrator_invoke_config.py` | Component implemented; integration coverage exists |
| CP-002 | GitHub issue polling | `src/codepilot/github/polling.py`, `src/codepilot/runner.py` | `tests/unit/test_github_polling.py`, `tests/unit/test_runner.py` | Component implemented; live polling not verified in this matrix |
| CP-003 | Task classification | `src/codepilot/agents/classifier.py`, `src/codepilot/agents/complexity.py` | `tests/unit/test_issue_classifier.py`, `tests/unit/test_complexity.py` | Component implemented and unit-tested |
| CP-004 | Repo Map | `src/codepilot/context/repo_map.py` | `tests/unit/test_repo_map.py` | Component implemented and unit-tested |
| CP-005 | Keyword retrieval | `src/codepilot/context/keyword_retriever.py` | `tests/unit/test_keyword_retriever.py` | Component implemented and unit-tested |
| CP-006 | ChromaDB / semantic retrieval | `src/codepilot/context/semantic_retriever.py`, `src/codepilot/memory/semantic_memory.py` | `tests/unit/test_semantic_retriever.py`, `tests/unit/test_semantic_memory.py` | Component implemented and unit-tested; runtime persistence needs environment verification |
| CP-007 | Sandboxed Coder | `src/codepilot/agents/coder.py`, `src/codepilot/sandbox/coder_workspace.py`, `src/codepilot/sandbox/executor.py` | `tests/unit/test_coder_agent.py`, `tests/unit/test_coder_workspace.py`, `tests/unit/test_sandbox_executor.py` | Component implemented and unit-tested |
| CP-008 | Test Agent and retry loop | `src/codepilot/agents/test_agent.py`, `src/codepilot/agents/coder.py` | `tests/unit/test_test_agent.py`, `tests/unit/test_coder_agent_test_integration.py` | Component implemented; retry-after-two-failures approval added in this review copy, pending execution in a dependency-complete environment |
| CP-009 | Skills | `src/codepilot/skills/skill_registry.py` | `tests/unit/test_skill_registry.py`, `tests/unit/test_orchestrator_skills.py` | Component implemented and unit-tested |
| CP-010 | Working, episodic, and semantic memory | `src/codepilot/memory/working_memory.py`, `src/codepilot/memory/episodic_memory.py`, `src/codepilot/memory/semantic_memory.py` | `tests/unit/test_working_memory.py`, `tests/unit/test_episodic_memory.py`, `tests/unit/test_semantic_memory.py` | Components implemented and unit-tested |
| CP-011 | Guardrails | `src/codepilot/sandbox/policy.py`, `src/codepilot/sandbox/executor.py` | `tests/unit/test_sandbox_policy.py`, `tests/unit/test_sandbox_executor.py` | Component implemented and unit-tested |
| CP-012 | Human-in-the-loop approval gates | `src/codepilot/github/pr_workflow.py`, `src/codepilot/agents/coder.py`, `src/codepilot/runner.py` | `tests/unit/test_pr_workflow.py`, `tests/unit/test_coder_agent_test_integration.py` | PR approval gates covered by tests; retry gate added in this review copy, pending full-suite verification |
| CP-013 | Pull-request workflow | `src/codepilot/github/pr_workflow.py`, `src/codepilot/agents/orchestrator.py` | `tests/unit/test_pr_workflow.py`, `tests/unit/test_orchestrator_pr_integration.py` | Component/integration tests exist; live PR publication not verified here |
| CP-014 | Four-panel TUI | `src/codepilot/tui/app.py`, `src/codepilot/runner.py`, `src/codepilot/github/polling.py` | `tests/unit/test_tui_app.py`, `tests/unit/test_runner.py`, `tests/unit/test_github_polling.py` | Real-model direct-task execution through the TUI was verified against the disposable `codepilot-smoke-test` repository on 2026-10-16; four-panel layout, multiline input, log forwarding, and full issue-to-PR approval flow are under active refinement/verification. |

## Verification commands

Run from the repository root in the project's supported Python environment:

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The live model smoke test is separate and may incur provider usage:

```powershell
python scripts/smoke_test_coder.py
```

Update statuses only after the corresponding tests or live checks have actually run. Do not mark a requirement fully complete solely because a source file or test filename exists.
