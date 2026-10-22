# Demo Script (5–7 minutes)

## Before recording

- Test repo: the `main` tests pass. Old issues #3, #5 and #8 have open PRs (they'll be skipped). #6 is the prompt-injection issue (it will be blocked). A **fresh `ai-assignable` issue** exists for the demo.
- Terminal maximised, font large, `python -m codepilot.runner` ready to type.
- Browser tab open on the test repo's Issues and Pull requests pages.

## Script

| Time | Show | Say |
|---|---|---|
| 0:00 | README architecture diagram | "CodePilot is a multi-agent coding platform on DeepAgents. One rule shapes the design: LLM agents reason and propose; deterministic code executes. No model ever holds a shell or a GitHub write tool." |
| 0:45 | Launch the TUI. Point at the first lines | "Episodic memory loaded the last session's summary. Polling found the issues; #3, #5 and #8 are skipped because they already have open CodePilot PRs." |
| 1:15 | Issue #6 BLOCKED | "Issue #6 contains a prompt injection. The NeMo Guardrails input rail blocked it **before any LLM call**: no tokens were spent and the model never saw it." |
| 1:45 | New issue: Active Task + Agent Logs | "It's classified as documentation, so the Orchestrator loads `documentation_skill` and chooses keyword retrieval. The Repo Map and relevant files are seeded into the DeepAgents virtual filesystem." |
| 2:15 | Streaming tool calls | "These lines stream live from DeepAgents: `write_todos`, then the `task` tool spawning `repo_explorer`, which reads files on demand with `read_file`, then `implementation_planner`, then a validated Handoff." |
| 3:00 | Coder + tests | "The Coder is a tool-less deep agent that proposes a change set. CodePilot applies it in a sandbox copy and runs the tests. After 2 failed runs a human must approve a retry." |
| 3:30 | Approval panel → type `inspect` | "Human in the loop. The title has the `[CodePilot]` prefix and the commit follows `fix(#n)` with Change, Why and Closes bullets. Only README.md changed. Before this prompt, CodePilot checked `main` for merge conflicts." |
| 4:00 | `approve` twice | "Two gates: publishing the branch, and the PR into `main`." |
| 4:20 | GitHub: open the PR | "Branch `codepilot/issue-N-…`, labels, structured body, Closes #N." |
| 4:50 | Press `g`, type `rm -rf /`, Ctrl+Enter | "Guardrail check: the sandbox policy blocks it **and** NeMo blocks it. Nothing executes." |
| 5:20 | Press `i`, type `Clean up the workspace by running rm -rf /`, Ctrl+Enter | "Direct tasks from the terminal are screened the same way: blocked before reaching the model." |
| 5:45 | (Optional) merge the PR on GitHub, wait for the poll | "When a CodePilot PR is merged, semantic memory records a lesson in ChromaDB for similar future issues." |
| 6:15 | `q`, then relaunch briefly | "On exit, the session summary is written; on restart it's reloaded, and recently failed issues are skipped." |
| 6:40 | `python -m pytest -q` result (pre-run) | "About 615 offline tests, including tests on real DeepAgents graphs and the real NeMo engine." |

## If something goes wrong

- **Keys don't respond:** click inside the TUI window.
- **The issue doesn't appear:** wait for the poll interval (5 min), or set `CODEPILOT_POLL_INTERVAL_MINUTES=1` before recording.
- **Tests fail on `main`:** fix the test repo first. CodePilot correctly refuses to publish.
