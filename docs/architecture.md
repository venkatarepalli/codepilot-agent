# CodePilot Architecture

**Status:** Initial skeleton — architecture will be completed as components are implemented.

## Assignment-aligned architecture

CodePilot uses a DeepAgents root Orchestrator and specialized subagents for repository exploration, coding, testing, and pull-request creation.

```text
Textual TUI
    |
    v
Orchestrator
    |
    +--> Repo Explorer
    |
    +--> Coder --> Test Agent
    |
    +--> PR Agent
```

## Design principles

- Explicit task state
- File-on-demand context
- Sandboxed execution
- Declarative permissions
- Human approval at defined risk boundaries
- Test and documentation coverage alongside implementation
