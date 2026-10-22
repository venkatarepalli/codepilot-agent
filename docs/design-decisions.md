# Design Decisions

| ID | Decision | Reason |
|---|---|---|
| DD-001 | Documentation maintained alongside the code | Keeps setup and rationale accurate |
| DD-002 | Tests added with each feature; every requirement has a named test file | Continuous requirement verification |
| DD-003 | Explicit substitution log | See [`tool-substitutions.md`](tool-substitutions.md) (S-1, S-2, D-1 … D-4) |
| DD-004 | Engineering-issue log | See [`engineering-issues/`](engineering-issues/) |
| DD-005 | **Five Skills for five categories** (resolved) | The assignment lists 5 classification categories but 4 required Skills. A 5th, `config_change_skill`, was added so every category maps to a Skill |
| DD-006 | **LLMs propose, code executes** | No LLM holds a shell, a file-write tool or a GitHub write tool, so guardrails and HITL are enforced in code (S-2) |
| DD-007 | **NeMo input rail without an LLM** | The rail runs a deterministic Colang flow and Python action: no API cost, identical results in tests and the demo, and fail-closed with a Python fallback |
| DD-008 | **Read-only GitHubToolkit for the Orchestrator** | Every toolkit tool is classified by `mode` (read-only / PR-write / blocked); a test fails if langchain adds an unclassified tool |
| DD-009 | **Orchestrator chooses the retrieval strategy per task** | `keyword` (fast, summary-based) for documentation, config and dependency tasks; `hybrid` (keyword + ChromaDB embeddings over content chunks) for bug fixes and features |
| DD-010 | **Relevant files seeded into the DeepAgents VFS** | `/repo-map.md` plus up to 12 retrieved files (secrets excluded), so subagents use `read_file` on demand and delegation prompts carry only paths |
| DD-011 | **Live streaming of agent tool calls** | The root agent runs via LangGraph `.stream(stream_mode="values", subgraphs=True)`; every root and subagent tool call appears in Agent Logs. The final state is identical to `invoke()` (tested) |
| DD-012 | **Merge-conflict detection before any write** | Changed files on the base branch are compared with the cloned checkout (CRLF-normalised) before approvals or remote writes; any difference fails the task. It is never auto-resolved |
| DD-013 | **Persistent episodic session summaries** | `sessions.jsonl` (git-ignored) is appended at exit; the last 3 are read at startup and recently FAILED issues are skipped. BLOCKED issues are re-checked because the guardrail check is cheap and deterministic |
| DD-014 | **PR conventions enforced at the single publication point** | `PullRequestWorkflow.publish` adds the `[CodePilot]` prefix exactly once and rebuilds any free-form commit message into the `fix(#n): …` format, so no caller can bypass them |
| DD-015 | **HITL retry gate counts failed test runs, not attempts** | An attempt that produced no change set never ran tests and must not trigger the gate |
