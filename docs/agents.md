# Agents

See the component table in [`architecture.md`](architecture.md) and design decision S-2 in [`tool-substitutions.md`](tool-substitutions.md).

- **Orchestrator** (root deep agent): classifies, selects the Skill, plans with `write_todos`, delegates with `task`, validates the Handoff.
- **repo_explorer**, **implementation_planner**: DeepAgents subagents working on the virtual filesystem.
- **Coder**: a tool-less deep agent returning a JSON change set; applied and tested by `CoderAgent`.
- **Test Agent**, **PR Agent**: deterministic components (no LLM), so execution and publication are always policy-checked and human-gated.
