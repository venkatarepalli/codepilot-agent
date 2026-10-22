# Skills

Skills are structured `SkillDefinition` objects (`src/codepilot/skills/skill_registry.py`), each with `name`, `instructions`, `workflow_steps`, `example_prompts` and `forbidden_actions`. The Orchestrator classifies each issue and loads the matching Skill into the task context, so the Implementation Planner and Coder follow its workflow and forbidden actions. The selected Skill is shown in the TUI's Active Task panel.

| Category | Skill | Workflow (summary) |
|---|---|---|
| `bug_fix` | `bug_fix_skill` | identify expected vs actual → inspect source/tests → reproduce with a test → minimal root-cause fix → regression tests → run tests → review diff |
| `feature_addition` | `feature_addition_skill` | acceptance criteria → inspect existing patterns → smallest design → implement → tests (normal/boundary/error) → run → review against criteria |
| `dependency_update` | `dependency_update_skill` | identify the dependency/version → inspect declarations and lock files → check compatibility → minimal update → update lock/config → run tests → review for unrelated changes |
| `documentation` | `documentation_skill` | audience → read existing docs and code → decide changes → update consistently → check links/commands/examples → run checks → review accuracy |
| `config_change` | `config_change_skill` | criteria → inspect settings/validation/defaults → security implications → minimal change → update examples/docs → test valid/invalid/default → review for secret exposure |

Each Skill lists 4 forbidden actions. Examples: "Do not hide or suppress a defect merely to make a test pass" (bug fix), and "Do not commit secrets, API keys, or credentials" (config change).

`config_change_skill` is a 5th Skill beyond the 4 the assignment requires, so every classification category has a Skill (DD-005). The Skills are a Python registry rather than DeepAgents SKILL.md files (D-4 in `tool-substitutions.md`).

Tests: `tests/unit/test_skill_registry.py`, `tests/unit/test_orchestrator_skills.py`, `tests/unit/test_runner_skill_display.py`.
