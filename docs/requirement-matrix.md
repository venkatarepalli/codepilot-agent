# CodePilot Requirement Matrix

The status reflects **both** the unit/integration tests (offline) and the live runs against `venkatarepalli/codepilot-smoke-test`, recorded in `docs/assets/`.

| ID | Requirement | Implementation | Tests | Live evidence |
|---|---|---|---|---|
| CP-001 | Root `create_deep_agent` Orchestrator, `write_todos` before delegation, `task` subagents | `agents/orchestrator.py` | `test_orchestrator*.py`, `tests/test_orchestrator_live_integration.py`, `test_agent_streaming.py` | Streamed `write_todos` → `task → repo_explorer` → `task → implementation_planner` |
| CP-002 | Polling every N min via the GitHubToolkit wrapper; `ai-assignable` or unassigned below threshold; exclude in-progress and open-PR issues | `github/polling.py`, `github/issue_selector.py`, `github/client.py` | `test_github_polling.py`, `test_issue_selector.py`, `test_open_pr_duplicate_guard.py` | "Skipping issue #3 because an open CodePilot pull request already exists" |
| CP-003 | Classification into 5 categories → Skill | `agents/classifier.py`, `skills/skill_registry.py` | `test_issue_classifier.py`, `test_orchestrator_skills.py`, `test_runner_skill_display.py` | `Skill: documentation_skill` / `feature_addition_skill` in the TUI |
| CP-004 | Repo Map (tree, language, symbols, summary; token budget; git-diff cache) | `context/repo_map.py` | `test_repo_map.py`, `test_repo_map_config_wiring.py` | `/repo-map.md` read by repo_explorer |
| CP-005 | Keyword + embedding (ChromaDB) retrieval; Orchestrator chooses per task | `context/*`, `choose_retrieval_strategy` | `test_keyword_retriever.py`, `test_semantic_retriever.py`, `test_context_assembler.py`, `test_retrieval_strategy_choice.py` | "Retrieval strategy for issue #8 (documentation): keyword"; "(feature_addition): hybrid" |
| CP-006 | Files on demand via `read_file`; paths only in delegation; Repo Map in VFS | `_initial_virtual_files` | `test_vfs_seeding.py`, `test_coder_context_relevance.py` | `[repo_explorer] tool call: read_file(/src/calculator.py)` returns real content |
| CP-007 | Auto-summarization | DeepAgents `SummarizationMiddleware` (D-3) | `test_orchestrator_summarization.py` | n/a |
| CP-008 | Sandboxed Coder (copy, permissions, block-lists) | `sandbox/*`, `agents/coder.py` (S-1) | `test_sandbox_*.py`, `test_coder_workspace.py`, `test_coder_agent.py` | "Tests: PASSED" in the workspace before publishing |
| CP-009 | Retry ≤3, HITL after 2 failed test runs, diff preview | `agents/coder.py` | `test_coder_agent_test_integration.py` | `demo-run-1.log`: "2 failed test runs; requesting human approval" |
| CP-010 | Working, episodic (session summary, last 3, skip recent failures) and semantic memory | `memory/*`, `github/merge_monitor.py` | `test_working_memory.py`, `test_episodic_memory*.py`, `test_session_summary.py`, `test_semantic_memory.py`, `test_merge_monitor*.py`, `test_memory_visibility.py` | "Episodic memory: loaded 1 previous session summary(ies)"; "Memory for issue …: 1 recent task(s)" |
| CP-011 | Guardrails: NeMo input rail, toolkit scoping, sandbox policy | `guardrails/`, `github/client.py`, `sandbox/policy.py` | `test_nemo_guardrails.py`, `test_github_toolkit_wiring.py`, `test_guardrail_*.py` | Issue #6 BLOCKED [prompt_injection]; direct task BLOCKED [destructive_shell] |
| CP-012 | HITL gates: PR to main/master, more than 5 files, `git push`, retry after 2 failures | `pr_workflow.py`, `policy.py`, `coder.py` | `test_pr_workflow.py`, `test_sandbox_policy.py`, `test_coder_agent_test_integration.py` | `publish_branch` + `create_pull_request_against_protected_branch` approvals |
| CP-013 | PR conventions: branch, `fix(#n)` commit, `[CodePilot]` title, body, labels, reviewer | `pr_workflow.py`, `github/client.py` | `test_pr_conventions.py`, `test_pr_workflow.py`, `test_github_client_reviewer_regression.py` | PR #7, PR #9 |
| CP-014 | Merge conflict → FAILED + TUI notice, never auto-resolved | `MergeConflictError` | `test_merge_conflict_detection.py` | No false conflict on PR #9 |
| CP-015 | TUI: 4 panels, live issues, streaming logs, approve/reject/inspect, `[i]` direct task | `tui/app.py`, `runner.py` | `test_tui_app.py`, `test_runner.py`, `test_agent_streaming.py` | Screenshots, `direct-task-success.log` |

## Verification commands

```powershell
python -m pytest -q
python -m ruff check src tests
python scripts/smoke_test_coder.py   # live model; costs API usage
```
