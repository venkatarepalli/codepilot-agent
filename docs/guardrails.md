# Guardrails

CodePilot layers its guardrails. Every layer fails closed.

| Layer | Where | What it stops | When |
|---|---|---|---|
| **1. NeMo Guardrails input rail** | `src/codepilot/guardrails/` (`config/config.yml`, `config/rails.co`) | Prompt injection, secret exfiltration, destructive shell requests (`rm -rf`, `curl … \| sh`, `mkfs`, `dd`), edits to protected files, attempts to bypass approval, force-push or push straight to main | Before **any** issue title/body or direct TUI task reaches the Orchestrator LLM |
| **2. GitHubToolkit role scoping** | `src/codepilot/github/client.py` | The Orchestrator LLM gets only read-only toolkit tools (`Get Issues`, `Get Issue`, `Read File`, `Search code`, …). `Delete File`, `Create/Update File`, `Create Pull Request`, `Set active branch` are never bound to an LLM. | At agent construction |
| **3. Sandbox execution policy** | `src/codepilot/sandbox/policy.py`, `executor.py` | `rm -rf`, `curl`, `wget`, `pip install`, shell operators, paths outside the sandbox, writes to `.env`, `*.secret`, `*.pem`, `*.key`, `*credentials*` | Every command and file write |
| **4. Human-in-the-loop gates** | `pr_workflow.py`, `coder.py`, `policy.py` → TUI approval panel | PR to `main`/`master`; commit touching more than 5 files; `git push`; retry after 2 **failed test runs** | Before the operation runs |

## NeMo Guardrails input rail

- The Colang 1.0 flow `codepilot input policy` calls the Python action `codepilot_input_policy_check`. If the verdict is blocked, the bot answers `CODEPILOT_GUARDRAIL_BLOCKED` and the flow stops.
- Only input rails run (`options={"rails": ["input"]}`), so **no LLM call, API key or network is needed**. Results are deterministic and the check takes milliseconds after a ~1 s first load.
- If `nemoguardrails` fails to import or raises, the same rules run directly in Python (`engine="python-fallback"`). It never fails open.
- A plain mention of `pip install` is deliberately **not** blocked at this layer, because `dependency_update` issues legitimately discuss it. Layer 3 blocks the actual command.

What happens when the rail blocks:

- **GitHub issue**: state becomes `TRIAGED → FAILED`, the issue panel shows `BLOCKED`, and the Agent Logs show `[Guardrails] … BLOCKED [rule]: reason`. No clone, no LLM call, and the claim is released.
- **Direct task (`[i]`)**: rejected with `GuardrailBlockedError` before cloning.
- **`[g]` guardrail check**: shows the sandbox-policy verdict *and* the NeMo verdict for the typed command. It never executes anything.

## Tests

`tests/unit/test_nemo_guardrails.py` (real NeMo engine, benign and malicious cases, worker-thread use under a running event loop, fallback, runner wiring), `test_github_toolkit_wiring.py`, `test_sandbox_policy.py`, `test_sandbox_executor.py`, `test_guardrail_check.py`, `test_guardrail_tui.py`, `test_pr_workflow.py`, `test_coder_agent_test_integration.py`.
