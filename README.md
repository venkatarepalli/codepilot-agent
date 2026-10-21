# CodePilot

> A multi-agent AI coding platform that autonomously analyzes GitHub issues, explores repositories, implements and tests changes in a sandbox, and opens pull requests with human approval at defined safety gates.

**Capstone:** Assignment 01 — Multi-Agent Coding Platform (CodePilot)
**Repository:** `codepilot-agent`
**Status:** Capstone implementation in progress; see the requirement matrix for verified component coverage.

---

## Table of Contents

- [Overview](#overview)
- [Core Capabilities](#core-capabilities)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [First-Time Setup](#first-time-setup)
- [Subsequent Runs](#subsequent-runs)
- [Configuration](#configuration)
- [Running CodePilot](#running-codepilot)
- [Testing](#testing)
- [Troubleshooting & Recovery](#troubleshooting--recovery)
- [Security & Guardrails](#security--guardrails)
- [Memory](#memory)
- [Skills](#skills)
- [Context Engineering](#context-engineering)
- [Design Decisions](#design-decisions)
- [Tool Substitutions](#tool-substitutions)
- [Engineering Issues & Lessons Learned](#engineering-issues--lessons-learned)
- [Known Limitations](#known-limitations)
- [Demo](#demo)
- [Assignment Requirement Mapping](#assignment-requirement-mapping)
- [Future Enhancements](#future-enhancements)

---

## Overview

CodePilot is a terminal-based multi-agent coding platform designed to work like a skilled junior software engineer. It can poll GitHub for suitable issues, classify and plan work, explore the repository, implement changes in a sandbox, run tests, and prepare a pull request while enforcing safety controls and human approval gates.

> **Documentation status:** `python -m codepilot.runner` launches the Textual interface; add `--headless` to use the console runner. See the requirement matrix for component-level evidence and known unverified requirements.

## Core Capabilities

- GitHub issue polling and triage
- Direct coding tasks through the Coder Agent workflow
- DeepAgents-based multi-agent orchestration
- Repository mapping and context engineering
- Keyword and embedding-based relevant-file retrieval
- Sandboxed code execution
- Reusable Skills for coding workflows
- Working, episodic, and semantic memory
- Command, path, and sensitive-file guardrails
- Human-in-the-loop approval gates
- Automated branch, commit, and pull-request workflow

## Architecture

```text
Textual TUI <---- thread-safe runtime updates / approval bridge
    |                                  ^
    | direct task / skip / shutdown    | issue snapshots / status / logs
    v                                  |
CodePilotRuntime ---------------- GitHubPollingService
    |                                  |
    v                                  v
CodePilotOrchestrator <---------- eligible GitHub issues
    |-- DeepAgents root agent
    |-- context / Repo Map components
    |-- CoderAgent -> DeepAgentsCoderAdapter -> CoderWorkspace -> SandboxExecutor
    |                                             |                  |
    |                                             v                  v
    |                                         TestAgent         policy checks
    |-- PullRequestWorkflow -> TUI/console approval callback -> GitHub API
    |-- working / episodic / semantic memory components
```

The TUI and runtime communicate through callbacks. Polling and coding work run away from the Textual event loop; approval requests are marshalled back to the UI thread. See [`docs/architecture.md`](docs/architecture.md).

## Technology Stack

| Layer | Technology | Status |
|---|---|---|
| Agent framework | DeepAgents | Used by orchestrator and coder adapter |
| Model integration | LangChain Google GenAI / OpenAI | Factory implemented; verify selected provider credentials/model |
| GitHub integration | LangChain `GitHubToolkit` (read-only tools bound to the Orchestrator) + PyGithub for HITL-gated publication | Implemented; see `tests/unit/test_github_toolkit_wiring.py` |
| Sandbox | CodePilot workspace copy + `SandboxPolicy` (justified substitution S-1 for the DeepAgents local sandbox backend) | Implemented; see [`docs/tool-substitutions.md`](docs/tool-substitutions.md) |
| Episodic memory | Project-managed task history | Implemented component; see tests |
| Semantic memory | ChromaDB | Implemented component; requires runtime initialization |
| TUI | Textual | Default runner entry point; runtime integration and approval bridge implemented, interactive end-to-end verification pending |
| Guardrails | NeMo Guardrails input rail + custom Python sandbox policy + HITL approval gates | Implemented; see [`docs/guardrails.md`](docs/guardrails.md) |

Dependency ranges are declared in `pyproject.toml`. Python 3.11, 3.12, and 3.13 are supported by the declared Python range; Python 3.14 is not supported.

## Project Structure

```text
codepilot-agent/
├── src/codepilot/
│   ├── agents/          # Orchestrator and specialized agents
│   ├── config/          # Configuration and environment handling
│   ├── graph/           # Workflow/state-machine definitions
│   ├── github/          # GitHub integration
│   ├── memory/          # Working, episodic, semantic memory
│   ├── repo/            # Repo Map and retrieval
│   ├── skills/          # Reusable agent Skills
│   ├── sandbox/         # Sandboxed execution
│   ├── guardrails/      # Safety policies and HITL gates
│   ├── tui/             # Textual interface
│   ├── tools/           # Agent tools
│   └── utils/           # Shared utilities
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── adr/
│   ├── engineering-issues/
│   ├── runbooks/
│   └── assets/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Prerequisites

- Windows PowerShell, macOS, or Linux shell
- Git
- Python **3.11, 3.12, or 3.13** (`pyproject.toml` declares `>=3.11,<3.14`)
- A supported LLM API key: Google (`LLM_PROVIDER=google`) or OpenAI (`LLM_PROVIDER=openai`)
- A GitHub personal access token and target repository for issue polling / pull-request operations
- Network access to install dependencies and reach the configured LLM/GitHub APIs

Use a repository where you have permission to read issues and create branches/pull requests. Start with a disposable test repository rather than a production repository.

## First-Time Setup

Run these commands from the directory where you want the checkout. Replace the repository URL with your actual GitHub repository URL.

### 1. Clone the repository

```powershell
git clone https://github.com/OWNER/codepilot-agent.git
Set-Location codepilot-agent
```

If you already have a local checkout, open PowerShell in its root instead.

### 2. Create and activate a virtual environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If Python 3.13 is not installed, use `py -3.12` or `py -3.11`. If PowerShell blocks activation, use the environment's interpreter directly (for example, `.\.venv\Scripts\python.exe -m pip install -e .`) or follow your organization's PowerShell execution policy; do not change system-wide policy solely for this project.

### 3. Install the project and development tools

```powershell
python -m pip install -e .
python -m pip install "pytest>=8,<9" "pytest-cov>=5,<7" "ruff>=0.8,<1"
```

### 4. Create the environment file

```powershell
Copy-Item .env.example .env
notepad .env
```

For Google, set `LLM_PROVIDER=google`, `LLM_MODEL` to a model available to your account, and `GOOGLE_API_KEY`. For OpenAI, set `LLM_PROVIDER=openai`, `LLM_MODEL` to a model available to your account, and `OPENAI_API_KEY`.

For GitHub issue polling, set `GITHUB_TOKEN`, `GITHUB_REPOSITORY_OWNER`, `GITHUB_REPOSITORY_NAME`, and `GITHUB_DEFAULT_BRANCH` (usually `main` or `master`). The token needs permissions appropriate to reading issues and creating branches/pull requests in the selected repository. Use least privilege. Never paste credentials into source files or commit `.env`.

### 5. Run tests and lint checks before using GitHub integration

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

These are local checks and do not require a live GitHub polling run. Fix environment/dependency errors separately from actual test failures.

### 6. Start CodePilot

After configuring the LLM and GitHub values, run:

```powershell
python -m codepilot.runner
```

The runner opens the four-panel Textual interface and starts GitHub polling in the background. It may clone the configured repository, invoke the LLM, run tests in a workspace, and ask for approval in the Human Approval panel before protected operations. Use a disposable test repository first. Press `q` to exit.

To use the previous console-only loop instead, run:

```powershell
python -m codepilot.runner --headless
```

### Local coding smoke test (uses the configured LLM)

This is a live-model smoke test, not an offline unit test. Configure the LLM provider and API key first:

```powershell
python scripts/smoke_test_coder.py
```

It may incur provider usage/costs. Passing this test does not prove that live GitHub issue polling and pull-request publication work end to end.

## Subsequent Runs

From the project root, activate the existing environment and start the runner:

```powershell
.\.venv\Scripts\Activate.ps1
python -m codepilot.runner
```

After changing dependencies or creating a fresh environment, reinstall the project and test tools using the commands above. Keep `.env` and `.codepilot/` local. `.codepilot/` contains runtime workspace/memory state; deleting it may remove persisted state and workspaces. Inspect and back up needed work before resetting it.

## Configuration

Runtime settings are read from environment variables; the runner loads `.env` from the current working directory. See `.env.example` for the template. Main configuration categories are:

- LLM provider/model
- GitHub repository
- issue polling interval
- issue complexity threshold
- Repo Map token budget
- Repo Map retrieval top-K
- maximum test retries
- CodePilot local state directory
- ChromaDB storage location

See [`docs/configuration.md`](docs/configuration.md).

## Running CodePilot

### GitHub issue mode

Run `python -m codepilot.runner` from the project root after configuring `.env`. The Issues panel displays the open issue snapshot, and eligible issues are queued for processing. The Active Task and Agent Logs panels receive runtime status updates. Press `l` to open the full-screen, scrollable log viewer; press `c` to copy all logs or `s` to save a transcript under `.codepilot/logs/`. Use `Esc` to return to the dashboard. Protected operations wait for explicit `approve` or `reject` input in the Human Approval panel. `inspect` logs the pending operation's details without approving it.

### Direct task mode

Press `i` to open the multi-line task editor. Paste or type the complete task, use `Enter` for new lines, and press `Ctrl+Enter` to submit it. CodePilot clones the configured repository and runs the existing Coder Agent/test workflow in its isolated task workspace. The proposed diff is retained for inspection. Direct tasks do **not** automatically open a pull request because they are not associated with a real GitHub issue; issue-linked PR publication continues through the GitHub issue workflow.

### Skip and stop

Press `s` to skip the oldest queued issue for the current session. It does not forcibly cancel an issue that is already running. Press `q` to request shutdown. Use `python -m codepilot.runner --headless` when an interactive TUI is not desired.

### Stopping and restarting

Press `q` to stop the TUI, or `Ctrl+C` to stop the headless loop. Runtime workspace and memory state is stored under `.codepilot/` by default (the runner's working clone is configured separately). Inspect existing work before deleting local state.

## Testing

Testing is implemented alongside each feature rather than added at the end of the project.

```text
Unit tests         -> fast, isolated component checks
Integration tests  -> external/service/component interactions
E2E tests          -> complete issue-to-PR workflow
```

Run the test suite from the project root with `python -m pytest -q`. For focused tests, use `python -m pytest tests/unit/test_coder_agent_test_integration.py -q`. Test markers are declared in `pyproject.toml`; not every test requires live external services. Run lint and format checks with `python -m ruff check .` and `python -m ruff format --check .`.

See [`docs/testing.md`](docs/testing.md).

## Troubleshooting & Recovery

Common recovery checks:

- Confirm Python with `python --version` and verify the virtual environment is active.
- Confirm dependencies with `python -m pip install -e .` and reinstall the test tools if needed.
- Confirm `.env` is in the project root and contains the selected provider key and GitHub repository settings. Never print or share token values while debugging.
- If approval is denied or unavailable, CodePilot should stop the protected operation; inspect the workspace and test output before rerunning.
- Preserve `.codepilot/` until you have inspected retained workspace changes.

See [`docs/troubleshooting.md`](docs/troubleshooting.md) and [`docs/engineering-issues/`](docs/engineering-issues/).

## Security & Guardrails

CodePilot must prevent unsafe operations in the sandbox and require human approval for defined high-risk actions.

Guardrails are layered, and every layer fails closed:

1. **NeMo Guardrails input rail** (`src/codepilot/guardrails/`): screens every issue and direct task *before* the LLM sees it. Blocks prompt injection, secret exfiltration, destructive shell requests, protected-file edits and approval bypass. It runs without an LLM, so it is deterministic.
2. **GitHubToolkit scoping**: the Orchestrator LLM only gets read-only toolkit tools; `Delete File`, `Create Pull Request` and the other write tools are never bound to it.
3. **Sandbox policy** (`src/codepilot/sandbox/policy.py`): blocks `rm -rf`, `curl`, `wget`, `pip install`, paths outside the sandbox, and `.env`/`*.pem`/`*.key`/`*.secret`/`*credentials*` writes.
4. **HITL gates** in the TUI approval panel: PR to `main`/`master`, commit touching more than 5 files, `git push`, retry after 2 failed test runs.

PR conventions are enforced in `PullRequestWorkflow.publish`: branch `codepilot/issue-{n}-{slug}`, title `[CodePilot] {issue title}`, and the commit format `fix(#n): …` / `- Change` / `- Why` / `- Closes #n`.

See [`docs/guardrails.md`](docs/guardrails.md).

## Memory

CodePilot implements three memory tiers required by the assignment:

1. Working memory
2. Episodic memory
3. Semantic memory

See [`docs/memory.md`](docs/memory.md).

## Skills

Reusable skill definitions and selection logic are in `src/codepilot/skills/skill_registry.py`; see `tests/unit/test_skill_registry.py`.

See [`docs/skills.md`](docs/skills.md).

## Context Engineering

Repo Map and context assembly components are in `src/codepilot/context/` and `src/codepilot/repo/`; see `tests/unit/test_repo_map.py`, `tests/unit/test_context_assembler.py`, and retriever tests. See [`docs/context-engineering.md`](docs/context-engineering.md) for design notes.

## Design Decisions

Significant architectural and implementation decisions will be recorded as they are made.

See [`docs/design-decisions.md`](docs/design-decisions.md) and [`docs/adr/`](docs/adr/).

## Tool Substitutions

**S-1 (Component 3):** the DeepAgents local sandbox backend is replaced by a per-issue workspace copy plus a declarative `SandboxPolicy`. `LocalShellBackend` executes on the host with no command filtering, and DeepAgents permissions cover file tools only, so neither can express the required command block-list. Auto-summarization is *not* a substitution: DeepAgents 0.7 enables it on every agent, and `tests/unit/test_orchestrator_summarization.py` verifies that.

Each substitution is documented with:

- assignment requirement
- original technology
- replacement
- reason
- alternatives considered
- trade-offs
- impact on assignment compliance
- verification

See [`docs/tool-substitutions.md`](docs/tool-substitutions.md).

## Engineering Issues & Lessons Learned

Significant implementation problems and their resolutions will be recorded throughout development rather than reconstructed at the end.

See [`docs/engineering-issues/`](docs/engineering-issues/).

## Known Limitations

- The TUI logs runtime status and result events, but the underlying agent does not yet stream every model thought or tool call into the Logs panel.
- Direct tasks produce a sandboxed proposed diff but do not automatically publish a pull request; PR publication currently requires a real GitHub issue.
- A real-model direct coding task has been verified through the integrated TUI against a disposable GitHub repository; the full live GitHub issue-to-pull-request workflow remains unverified.
- Agent Logs forwards CodePilot and selected dependency log records while the TUI is active. Streaming every internal model thought/tool call remains outside the current implementation.
- The model factory implements Google and OpenAI providers; each provider/model combination must be tested with its own credentials before claiming runtime compatibility.

## Demo

The final submission will include a 5–7 minute demo covering the required CodePilot flow, including issue polling, an end-to-end task, HITL approval, and a guardrail block.

**Demo:** _To be recorded after the system is stable._

## Assignment Requirement Mapping

The requirement matrix in [`docs/requirement-matrix.md`](docs/requirement-matrix.md) records source paths and test references. Component-level implementation does not by itself prove full end-to-end assignment compliance.

See [`docs/requirement-matrix.md`](docs/requirement-matrix.md).

## Future Enhancements

Bonus challenges will be considered only after all mandatory requirements are implemented and tested.

Potential enhancements include:

- self-healing test setup
- issue complexity scoring
- LangSmith tracing
- cloud sandbox
- ACP integration

## License

This project is a learning capstone implementation.
