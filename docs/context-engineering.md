# Context Engineering

| Rule / requirement | Implementation |
|---|---|
| Repo Map: tree + path, language, exported symbols, 1-line description | `context/repo_map.py` (Python symbols via `ast`; other languages get language + path) |
| Token budget, default 4000, configurable | `CODEPILOT_REPO_MAP_TOKEN_BUDGET`; `_fit_token_budget` trims summaries until the map fits |
| Disk cache invalidated with git diff | `RepoMapCache`: git revision + `git diff --name-only HEAD` + file fingerprint |
| Repo Map in the DeepAgents VFS | `/repo-map.md`, plus up to 12 relevant files (40k chars, secrets excluded), seeded into the `StateBackend` and kept on recovery attempts |
| Top-K relevant files, default 10 | `CODEPILOT_REPO_MAP_TOP_K`; `ContextAssembler` |
| Keyword strategy (fast, file summaries) | `context/keyword_retriever.py` |
| Embedding strategy (ChromaDB, content chunks) | `context/semantic_retriever.py` (chunked, persistent ChromaDB) |
| Orchestrator chooses the strategy | `choose_retrieval_strategy`: `keyword` for documentation/config/dependency, `hybrid` for bug fixes and features; logged per task |
| Subagents load files on demand | Repo Explorer and Planner use `ls`/`read_file`/`grep` on the VFS. The Coder exception is D-2 |
| No raw file content in delegation prompts | Delegation carries the Repo Map, paths and summaries only (`ContextPackage.to_prompt_text`) |
| Auto-summarization | `SummarizationMiddleware` on the root agent and every subagent (D-3; proven by test) |
| Coder context | Relevance-first selection (retrieved files before others), capped at 12 files / 40k chars |
