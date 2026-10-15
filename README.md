# CodePilot

> A multi-agent AI coding platform that autonomously analyzes GitHub issues, explores repositories, implements and tests changes in a sandbox, and opens pull requests with human approval at defined safety gates.

**Capstone:** Assignment 01 — Multi-Agent Coding Platform (CodePilot)
**Repository:** `codepilot-agent`
**Status:** 🚧 In active development

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

> **Documentation status:** This README is maintained throughout development. Setup, operational procedures, design decisions, issues, and testing guidance are added when the corresponding functionality is implemented.

## Core Capabilities

- GitHub issue polling and triage
- Direct coding tasks through a Textual TUI
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

> Architecture diagram will be finalized after the core agent workflow is implemented.

```text
Textual TUI
     |
     v
Orchestrator (DeepAgents root agent)
     |
     +--> Repo Explorer
     |
     +--> Coder --> Test Agent
     |
     +--> PR Agent
```

See [`docs/architecture.md`](docs/architecture.md) for the detailed architecture once implemented.

## Technology Stack

| Layer | Technology | Status |
|---|---|---|
| Agent framework | DeepAgents | Planned |
| Agent runtime | LangGraph via DeepAgents | Planned |
| GitHub integration | LangChain GitHub Toolkit | Planned |
| Sandbox | DeepAgents local sandbox backend | Planned |
| Episodic memory | LangGraph Memory Store | Planned |
| Semantic memory | ChromaDB | Planned |
| TUI | Textual | Planned |
| Guardrails | Custom Python + assignment-required controls | Planned |

> Dependency versions will be pinned only after the environment and compatibility matrix are verified. We will not invent versions prematurely.

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

> This section will contain verified prerequisites only after the Phase 0 environment check.

Expected prerequisites include:

- Python version compatible with the selected dependency set
- Git
- GitHub account/token with the permissions required by the assignment
- Access to a supported LLM provider
- A GitHub repository for testing CodePilot

## First-Time Setup

This section is intentionally written as an operational runbook and will be completed as setup is implemented.

### 1. Clone the repository

```bash
git clone <repository-url>
cd codepilot-agent
```

### 2. Create the virtual environment

_To be completed after the supported Python/dependency strategy is verified._

### 3. Install dependencies

_To be completed after dependency selection and version pinning._

### 4. Configure environment variables

Copy `.env.example` to `.env` and populate only the required values.

```text
.env.example  ->  .env
```

**Never commit `.env`.**

### 5. Validate the environment

_To be added with the project diagnostic command._

### 6. Initialize CodePilot local state

_To be added when the state/memory initialization workflow is implemented._

### 7. Run the startup validation

_To be added when the startup validation command is implemented._

## Subsequent Runs

If CodePilot has already been set up successfully, do **not** repeat first-time initialization unless the relevant component has been reset.

The final documented procedure will explicitly distinguish:

- normal second/subsequent run
- dependency changes
- configuration changes
- repository changes requiring Repo Map invalidation
- interrupted/failed task recovery
- ChromaDB/state recovery
- complete clean reset

## Configuration

Configuration will be centralized and documented rather than scattered across source files.

The initial configuration categories are:

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

_To be documented after issue polling is implemented._

### Direct task mode

_To be documented after the TUI direct-task workflow is implemented._

### Stopping and restarting

_To be documented with the actual runtime/state behavior._

## Testing

Testing is implemented alongside each feature rather than added at the end of the project.

```text
Unit tests         -> fast, isolated component checks
Integration tests  -> external/service/component interactions
E2E tests          -> complete issue-to-PR workflow
```

The final README will document the exact commands, prerequisites, environment requirements, test markers, expected duration, and cleanup/recovery behavior for each test category.

See [`docs/testing.md`](docs/testing.md).

## Troubleshooting & Recovery

Known setup and runtime issues will be documented as they are encountered, including:

- symptoms
- root cause
- investigation
- resolution
- verification
- prevention/workaround

See [`docs/troubleshooting.md`](docs/troubleshooting.md) and [`docs/engineering-issues/`](docs/engineering-issues/).

## Security & Guardrails

CodePilot must prevent unsafe operations in the sandbox and require human approval for defined high-risk actions.

The implemented guardrails and their tests will be documented in [`docs/guardrails.md`](docs/guardrails.md).

## Memory

CodePilot implements three memory tiers required by the assignment:

1. Working memory
2. Episodic memory
3. Semantic memory

See [`docs/memory.md`](docs/memory.md).

## Skills

Reusable Skills will be documented with their purpose, workflow, inputs, forbidden actions, selection rules, and tests.

See [`docs/skills.md`](docs/skills.md).

## Context Engineering

The Repo Map, token budget, retrieval strategies, file-on-demand loading, virtual filesystem, and summarization behavior will be documented in [`docs/context-engineering.md`](docs/context-engineering.md).

## Design Decisions

Significant architectural and implementation decisions will be recorded as they are made.

See [`docs/design-decisions.md`](docs/design-decisions.md) and [`docs/adr/`](docs/adr/).

## Tool Substitutions

If any assignment-specified tool or technology is replaced, the substitution will be explicitly documented with:

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

This section will contain verified limitations only. It will be updated as implementation progresses.

## Demo

The final submission will include a 5–7 minute demo covering the required CodePilot flow, including issue polling, an end-to-end task, HITL approval, and a guardrail block.

**Demo:** _To be recorded after the system is stable._

## Assignment Requirement Mapping

A living requirement-to-implementation-to-test-to-demo matrix will be maintained during development.

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
