# Tool Substitutions and Deviations

Every place where CodePilot departs from the letter of the assignment is recorded here. Nothing is substituted silently.

---

## S-1: Component 3 sandbox (DeepAgents local sandbox backend → workspace copy + SandboxPolicy)

| Field | Detail |
|---|---|
| **Requirement** | The Coder uses DeepAgents' local sandbox backend, so `execute` runs in an isolated directory holding a copy of the relevant repo subset, with declarative `Permission(path="/sandbox/", access="read_write")` and `Permission(path="/", access="read_only")`. |
| **Original** | DeepAgents sandbox/filesystem backend with `permissions=[...]`; the LLM calls `execute` and `edit_file` directly. |
| **Replacement** | The Coder deep agent has **no tools** and *proposes* a JSON change set. CodePilot applies it through `CoderWorkspace`, a per-issue copy of the cloned repo outside the live checkout (`.git`, `.venv`, `node_modules` and caches excluded), and runs tests through `SandboxExecutor`. One declarative `SandboxPolicy` blocks:<br>• `rm -rf`, `curl`, `wget`, `pip install` / `python -m pip install`<br>• shell control operators<br>• any path outside the sandbox root<br>• writes to `.env`, `*.secret`, `*.pem`, `*.key`, `*credentials*`<br>`git push` is routed to the HITL panel. |
| **Reason** | (a) DeepAgents 0.7 `LocalShellBackend` documents itself as "unrestricted local shell command execution… directly on the host system without any sandboxing, process isolation, or security restrictions". (b) `FilesystemPermission` covers only `read`/`write` file operations, not `execute`, so the required command block-list cannot be expressed with it. (c) Removing the shell from the LLM entirely means every guardrail is enforced in code, at one auditable chokepoint. |
| **Alternatives considered** | `LocalShellBackend` + `FilesystemPermission` (no command filtering; runs on the host). Daytona/Modal cloud sandboxes (a bonus item; needs paid accounts and network setup). Docker (not available on every reviewer machine). |
| **Trade-offs** | The Coder gets test feedback once per attempt rather than iterating interactively with `execute`. Isolation is directory-plus-policy, not OS-level. The **whole** repository is copied rather than a subset, because a partial copy breaks imports in the test suite. |
| **Impact** | Every Component 3 guarantee holds: isolated copy, no writes outside the sandbox, the required command and file block-lists, max 3 retries, the HITL gate after 2 failed test runs, and the diff preview at `working/proposed_diff.txt`. |
| **Note on "Coder requests a blocked operation"** | By design the Coder has no shell, so it cannot request one. The only executed commands are the configured test command and the `[g]` check-only preview, which shows both guardrail layers and never executes anything. |
| **Evidence** | `test_sandbox_policy.py`, `test_sandbox_executor.py`, `test_sandbox_workspace.py`, `test_coder_workspace.py`, `test_coder_agent*.py`, `test_adapter_task_context_security.py`; live logs in `docs/assets/`. |

---

## S-2: Agent architecture (Coder, Test and PR agents are not `task`-spawned LLM subagents)

| Field | Detail |
|---|---|
| **Requirement** | The Orchestrator spawns Repo Explorer, Coder, Test and PR agents as subagents via the `task` tool. The Coder uses `write_file`/`edit_file`/`execute`, the Test Agent uses `write_file`/`execute`, and the PR Agent uses GitHub Toolkit `create_branch`/`create_pull_request`/`create_commit`. |
| **Implementation** | • `repo_explorer` and `implementation_planner` **are** real DeepAgents subagents, spawned via `task`. The run is rejected unless `write_todos` comes first and both are called (`_validate_and_format_root_handoff`).<br>• **Coder** is a separate deep agent (`codepilot_coder`) with **no tools**. It plans and returns a JSON change set; `CoderAgent` applies and tests it.<br>• **Test Agent** (`TestAgent`) is a deterministic component called by `CoderAgent`. It runs the suite and parses results.<br>• **PR Agent** (`PullRequestWorkflow`) is a deterministic component. It creates the branch, commit and PR through PyGithub's Git Data API, behind the HITL gates. |
| **Reason** | **No LLM ever holds a shell, a file-write tool, or a GitHub write tool.** All guardrails, path checks, conventions and approval gates are therefore enforced in code and cannot be bypassed by a prompt-injected model. The LangChain GitHubToolkit also has no `create_commit` tool, and its `Create File` / `Update File` tools commit one file at a time on the shared active branch, so they can't create one structured multi-file commit. |
| **Trade-offs** | The Test Agent doesn't write tests itself; the Coder's change set may include tests, as the `bug_fix_skill` instructs. Execution and publication are not LLM-driven. |
| **Evidence** | `test_orchestrator_live_integration.py`, `test_github_toolkit_wiring.py` (write tools never bound to the LLM), `test_agent_streaming.py`, `test_pr_workflow.py`, `test_pr_conventions.py`. |

---

## D-1: Component 1 polling through the GitHubToolkit wrapper, not its `Get Issues` tool

The toolkit's `Get Issues` tool returns plain text containing only `title`, `number` and `opened_by`: no labels, assignee or body. Filtering on `ai-assignable` and "unassigned" is impossible with it. So polling uses the toolkit's own `GitHubAPIWrapper` repository instance (`GitHubClient.repository` *is* `api_wrapper.github_repo_instance`), making the same `get_issues(state="open")` call but keeping the structured fields. The Orchestrator LLM is still bound to the toolkit's read-only tools, including `Get_Issues` and `Get_Issue`.

## D-2: Component 2 Coder context includes file contents

Repo Explorer and Implementation Planner follow the "load on demand" rule: they read `/repo-map.md` and the relevant files from the DeepAgents virtual filesystem with `read_file`. The Coder prompt does include file contents, because the Coder must return **complete replacement file contents** (a whole-file JSON change set) and has to see the exact current text to do that safely. The context is chosen **relevance-first** from Repo Map retrieval and capped (12 files / 40k characters, secrets excluded). The same files are also seeded into its virtual filesystem.

## D-3: Component 2 "summarization=True"

`create_deep_agent()` in DeepAgents 0.7 has **no** `summarization` parameter; passing one raises `TypeError`. The library always installs `SummarizationMiddleware` on the root agent and on every subagent. CodePilot records the setting as `ORCHESTRATOR_BACKEND_CONFIG["summarization"] = True`, checks at startup that the installed library provides it, and proves it in `test_orchestrator_summarization.py`, which spies on the real library: summarization is created for the root agent, the general-purpose subagent and both custom subagents.

## D-4: Component 4 Skills registry instead of `deepagents.skills`

The five Skills are structured Python objects (`SkillDefinition`: `name`, `instructions`, `workflow_steps`, `example_prompts`, `forbidden_actions`), selected by issue category and injected into the Orchestrator's task context. They are not DeepAgents SKILL.md files. The behaviour is the same, and they are type-checked and unit-tested.
