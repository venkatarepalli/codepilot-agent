# CodePilot Architecture

The full diagram is in the [README](../README.md#architecture).

## Components and their roles

| Agent / component | Type | Tools / capabilities | Source |
|---|---|---|---|
| **Orchestrator** | Root `create_deep_agent()` with a planning system prompt; `TodoListMiddleware`; `StateBackend`; auto-summarization | `write_todos`, `task`, VFS tools (`ls`, `read_file`, `grep`, `glob`), read-only GitHubToolkit tools | `agents/orchestrator.py` |
| **repo_explorer** | DeepAgents subagent (spawned via `task`) | VFS tools over `/repo-map.md` and the seeded relevant files | `ORCHESTRATOR_SUBAGENTS` |
| **implementation_planner** | DeepAgents subagent (spawned via `task`) | VFS tools; follows the selected Skill's workflow and forbidden actions | `ORCHESTRATOR_SUBAGENTS` |
| **Coder** | Separate deep agent, **no tools** | Returns a plan, then a JSON change set | `agents/deepagents_coder_adapter.py`, `agents/coder.py` |
| **Test Agent** | Deterministic component | Runs the configured test command in the sandbox and parses the result | `agents/test_agent.py` |
| **PR Agent** | Deterministic component | Merge-conflict check, HITL gates, branch/commit/PR/labels/reviewer | `github/pr_workflow.py` |
| **Guardrails** | NeMo Guardrails input rail + sandbox policy | Screens task text; blocks commands and paths | `guardrails/`, `sandbox/policy.py` |
| **Runtime** | Threads for polling and processing; bridge to the TUI | Approvals, session tracking, lifecycle | `runner.py` |
| **TUI** | Textual app | 4 panels, approve/reject/inspect, `[i]` `[g]` `[s]` `[l]` `[q]` | `tui/app.py` |

## Run sequence for one issue

1. **Poll:** `GitHubPollingService` → `select_eligible_issues` (label / unassigned + complexity, not in progress, no open CodePilot PR).
2. **Episodic skip:** skip the issue if it FAILED in one of the last 3 sessions.
3. **Guardrails:** the NeMo input rail screens the title and body. If blocked, `TRIAGED → FAILED` and no LLM call is made.
4. **Clone** a shallow copy (the token goes in an environment header, not the URL).
5. **Prepare the task:** classify → Skill → retrieval strategy → Repo Map + top-K files → semantic lessons (top-3) + recent episodic tasks → working memory.
6. **Root agent (streamed):** `write_todos` → `task(repo_explorer)` → `task(implementation_planner)` → `Handoff:`. This is validated, with up to 2 recovery attempts.
7. **Coder loop:** plan → change set → apply in the sandbox → Test Agent → retry ≤3 (HITL after 2 failed test runs) → `working/proposed_diff.txt`.
8. **PR Agent:** conflict check → HITL (`publish_branch`; PR to `main`/`master`; more than 5 files) → commit, branch, PR, labels, reviewer.
9. **Memory:** record the task in the session tracker and LangGraph episodic store; the merge monitor records a semantic lesson when the PR is merged.
10. **Exit:** the session summary is appended to `.codepilot/memory/sessions.jsonl`.

The state machine is `TRIAGED → EXPLORING → IMPLEMENTING → TESTING → PR_OPENED → DONE | FAILED` (`graph/lifecycle.py`).
