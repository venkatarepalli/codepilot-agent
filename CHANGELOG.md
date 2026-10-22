# Changelog

## 0.1.0 — Capstone submission (2026-10-22)

- Root DeepAgents Orchestrator with enforced `write_todos` → `task(repo_explorer)` → `task(implementation_planner)` → Handoff; live streaming of root and subagent tool calls into the TUI.
- GitHubToolkit integration with role scoping (read-only tools for the LLM); polling with label/complexity filters and a duplicate-PR guard.
- Context engineering: Repo Map with token budget and git-diff cache; keyword + ChromaDB retrieval chosen per task; relevant files seeded into the DeepAgents VFS; relevance-first Coder context; auto-summarization verified.
- Sandboxed Coder loop with max 3 retries, HITL after 2 failed test runs, and diff preview.
- Five structured Skills, one per issue category, shown in the TUI.
- Memory: working, episodic (persistent session summaries; recently failed issues skipped) and semantic (lessons from merged PRs).
- Guardrails: NeMo Guardrails input rail, sandbox policy, four HITL gates, merge-conflict detection.
- PR conventions enforced at publication: branch, `[CodePilot]` title, `fix(#n)` commit format, labels, reviewer.
- Documentation: README, `.env.example`, design decisions, substitutions (S-1, S-2, D-1 … D-4), requirement matrix, demo script.
