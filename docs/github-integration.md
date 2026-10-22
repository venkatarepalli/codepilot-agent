# GitHub Integration

- **Client:** `github/client.py` builds the LangChain `GitHubToolkit` around a PAT-authenticated `GitHubAPIWrapper`.
- **LLM tools:** read-only toolkit tools only (`Get_Issues`, `Get_Issue`, `Read_File`, `Search_code`, …). Write tools are never bound (DD-008).
- **Polling:** `github/polling.py` uses the toolkit wrapper's repository (D-1). It excludes in-progress issues and issues with an open `[CodePilot]` PR on a `codepilot/issue-N-*` branch.
- **Publication:** `github/pr_workflow.py` runs a merge-conflict check, then the HITL gates, then blob → tree → commit (`fix(#n)` format) → branch `codepilot/issue-{n}-{slug}` → PR `[CodePilot] {title}` → labels `codepilot-generated`, `needs-review` → review request for the reporter (skipped if the reporter is the PR author).
- **Merge monitor:** `github/merge_monitor.py` checks PRs published this session, and records a semantic lesson once one is merged.
- **Token:** a fine-grained PAT (Contents, Pull requests, Issues: read and write; Metadata: read). It is passed to `git clone` through an environment header, never in the URL or the command line.
