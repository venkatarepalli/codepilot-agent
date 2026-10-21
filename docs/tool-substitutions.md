# Tool Substitutions

Each substitution of an assignment-specified tool is recorded below. Nothing is substituted silently.

## S-1 — Component 3: DeepAgents local sandbox backend → CodePilot sandbox workspace + SandboxPolicy

| Field | Detail |
|---|---|
| **1. Assignment requirement** | Component 3: the Coder uses DeepAgents' *local sandbox backend* so `execute` runs in an isolated directory holding a copy of the relevant repo subset, with declarative `Permission(path="/sandbox/", access="read_write")` / `Permission(path="/", access="read_only")`. |
| **2. Original tool** | DeepAgents sandbox/filesystem backend with `permissions=[...]` on the Coder deep agent, the LLM calling `execute` and `edit_file` directly. |
| **3. Replacement** | The Coder deep agent (`create_deep_agent(..., name="codepilot_coder", tools=[])`) runs on the invocation-scoped `StateBackend`, which is a virtual filesystem seeded only with filtered workspace files. It *proposes* a JSON change set and never touches disk or a shell. CodePilot then applies the change set and runs tests itself, through: <br>• `WorkspaceManager` / `CoderWorkspace` (`src/codepilot/sandbox/`): a per-issue copy of the cloned repo under a dedicated workspace root, outside the live checkout (`.git`, `.venv`, `node_modules`, caches excluded). <br>• `SandboxPolicy` + `SandboxExecutor`: one declarative policy object that blocks `rm -rf`, `curl`, `wget`, `pip install`/`python -m pip install`, shell control operators, any path outside the sandbox root, and writes to `.env`, `*.secret`, `*.pem`, `*.key`, `*credentials*`; `git push` is routed to the HITL approval panel. |
| **4. Reason** | (a) The live DeepAgents 0.7 API for local shell execution (`LocalShellBackend`) runs commands on the host with **no** command filtering. Its permission model covers file tools, not `execute`, so it cannot express the required command block-list on its own. (b) Having the LLM call `execute` directly means every guardrail would depend on the model's cooperation. With the substitution, the model has no shell at all, and every command goes through one auditable chokepoint. (c) It runs the same on Windows (the development machine) and Linux. |
| **5. Alternatives considered** | `LocalShellBackend` plus `FilesystemPermission` (no command block-list; host execution). Daytona/Modal cloud sandboxes (listed as a bonus; needs paid accounts and network setup, and adds demo risk before the deadline). Docker (not available on all reviewer machines). |
| **6. Trade-offs** | The Coder LLM cannot iterate interactively with `execute`; it gets test feedback once per attempt (up to 3 attempts, HITL gate after 2 failed test runs). Isolation is directory-plus-policy, not OS-level: there is no container or seccomp. |
| **7. Assignment impact** | Every Component 3 guarantee still holds: isolated copy, not the live repo; no writes outside the sandbox; the required command and file block-lists; diff preview at `working/proposed_diff.txt`; max 3 retries; human approval for blocked or approval-required operations. Only the mechanism differs. |
| **8. Verification evidence** | `tests/unit/test_sandbox_policy.py`, `test_sandbox_executor.py`, `test_sandbox_workspace.py`, `test_coder_workspace.py`, `test_coder_agent.py`, `test_coder_agent_test_integration.py`, `test_adapter_task_context_security.py`. Live: the TUI `[g]` guardrail check and the demo video. |

## Not a substitution — auto-summarization (Component 2)

The assignment says "`summarization=True` in the backend config". `create_deep_agent()` in DeepAgents 0.7 has **no** `summarization` parameter (passing it raises `TypeError`). Instead, the library always installs `SummarizationMiddleware` on the root agent and on every subagent. CodePilot records the setting as `ORCHESTRATOR_BACKEND_CONFIG["summarization"] = True`, checks at startup that the installed library still provides it, and proves it in `tests/unit/test_orchestrator_summarization.py`, which spies on the real library and confirms summarization middleware is created for the root agent, the general-purpose subagent and both custom subagents.

## Deferred — not implemented by the submission deadline

- Session-end summary hook for episodic memory (Component 5). Tasks are recorded in episodic memory during the session, but the end-of-session summary write and the "read last 3 summaries at startup" step are not implemented.
- Storing the Repo Map in the DeepAgents virtual filesystem via an explicit `write_file` tool call (Component 2). The Repo Map is cached to disk (git-diff invalidation) and seeded into the agent's `StateBackend` as `/repo-map.md` at invocation time (`CodePilotOrchestrator._initial_virtual_files`), so subagents read it with `read_file`; it is not written through a `write_file` tool call.
