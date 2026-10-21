# CodePilot Architecture

**Status:** Runtime-to-TUI integration implemented in source; full interactive verification remains pending.

## Assignment-aligned architecture

CodePilot uses a DeepAgents root Orchestrator and specialized subagents for repository exploration, coding, testing, and pull-request creation.

```text
Textual TUI
    ^       |
    |       v
thread-safe runtime callbacks
    ^       |
    |       v
CodePilotRuntime <---- GitHubPollingService
    |                         |
    v                         v
Orchestrator <---------- eligible issues
    |
    +--> Repo Explorer
    |
    +--> Coder --> Test Agent
    |
    +--> PR Agent
```

The Textual app is the default entry point. Polling and coding tasks run on
background threads so synchronous model calls do not block the UI. The runtime
publishes issue snapshots, task status, and execution events to the UI. Human
approval requests made by the runtime are bridged to the Textual event loop and
fail closed when approval cannot be obtained.

Direct tasks use the configured repository and the existing sandboxed Coder
Agent/test workflow. They retain a proposed diff for inspection but do not open
a PR because there is no linked GitHub issue. Issue-linked work uses the
existing PR workflow and its approval gates.

## Design principles

- Explicit task state
- File-on-demand context
- Sandboxed execution
- Declarative permissions
- Human approval at defined risk boundaries
- UI updates dispatched on the Textual event-loop thread
- Polling and coding work kept off the UI event loop
- Test and documentation coverage alongside implementation
