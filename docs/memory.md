# Memory

CodePilot has three memory tiers.

| Tier | Scope | Storage | Contents | Source |
|---|---|---|---|---|
| **Working** | One task | In memory (`WorkingMemory`) | Issue metadata, Repo Map, relevant files, current diff, test results, retry count, retrieved memory | `memory/working_memory.py` |
| **Episodic** | Session, plus summaries that persist across sessions | LangGraph `InMemoryStore` during the session; `.codepilot/memory/sessions.jsonl` across sessions | Per task: issue ID, task type, source (issue / direct task), files modified, outcome, duration, PR URL | `memory/episodic_memory.py`, `memory/session_summary.py` |
| **Semantic** | Across sessions | ChromaDB (`.codepilot/memory/semantic`) | "Lesson learned" per merged PR: problem, resolution, outcome, keyed by repository + category | `memory/semantic_memory.py`, `github/merge_monitor.py` |

## Lifecycle

- **Startup:** the last 3 session summaries for the repository are loaded. Issues that FAILED in them are skipped, with a `[Memory]` log line. BLOCKED issues are re-checked by the guardrail rail rather than skipped.
- **Before each task:** the top-3 similar lessons (semantic) and recent tasks (episodic) are injected into the task context as *untrusted reference material*. The log shows `Memory for issue #N: X past lesson(s) …, Y recent task(s) …`.
- **After each task:** the outcome is recorded in the session tracker and in the LangGraph episodic store. Working memory is cleared at DONE/FAILED.
- **After a PR is merged:** if CodePilot is still running, the merge monitor records a semantic lesson (`Semantic memory: recorded lesson from merged PR #N`).
- **Exit:** one structured summary is appended (`Episodic memory: session summary written`).

## Reset

Delete `.codepilot/memory/sessions.jsonl` to reset episodic memory, or `.codepilot/memory/semantic/` to reset semantic memory. `CODEPILOT_MEMORY_DIR` relocates the session summaries.

## Known limitation

Semantic lessons are recorded only for PRs merged while CodePilot is running, because the merge monitor tracks PRs published in the current session.
