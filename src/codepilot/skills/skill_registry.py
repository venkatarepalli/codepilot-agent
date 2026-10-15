"""Registry of structured skills used by CodePilot agents."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from codepilot.agents.classifier import IssueCategory


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    """Define the instructions and constraints for a CodePilot skill."""

    name: str
    instructions: str
    workflow_steps: tuple[str, ...]
    example_prompts: tuple[str, ...]
    forbidden_actions: tuple[str, ...]


BUG_FIX_SKILL = SkillDefinition(
    name="bug_fix_skill",
    instructions=(
        "Investigate the reported defect, establish its root cause, "
        "implement the smallest appropriate correction, and add regression tests. "
        "Preserve existing behavior outside the scope of the fix."
    ),
    workflow_steps=(
        "Read the issue and identify the expected and actual behavior.",
        "Inspect relevant source files, tests, and repository conventions.",
        "Reproduce the defect or establish a test that demonstrates it.",
        "Identify the root cause and implement a minimal correction.",
        "Add or update regression tests.",
        "Run relevant tests and inspect the results.",
        "Review the proposed diff for correctness and unintended changes.",
    ),
    example_prompts=(
        "Fix the incorrect total returned by the order calculation.",
        "Investigate and resolve the exception raised for an empty input.",
    ),
    forbidden_actions=(
        "Do not hide or suppress a defect merely to make a test pass.",
        "Do not remove a failing regression test without justification.",
        "Do not make unrelated changes outside the scope of the issue.",
        "Do not claim that a defect is fixed without supporting verification.",
    ),
)


FEATURE_ADDITION_SKILL = SkillDefinition(
    name="feature_addition_skill",
    instructions=(
        "Implement the requested functionality consistently with existing "
        "architecture and coding conventions. Prefer focused, maintainable "
        "changes and cover normal, boundary, and error cases with tests."
    ),
    workflow_steps=(
        "Extract the requested behavior and acceptance criteria from the issue.",
        "Inspect relevant source files, interfaces, tests, and conventions.",
        "Identify the smallest suitable implementation approach.",
        "Implement the feature in focused, understandable changes.",
        "Add or update tests for normal, boundary, and error cases.",
        "Run relevant tests and inspect the results.",
        "Review the proposed diff against the acceptance criteria.",
    ),
    example_prompts=(
        "Add pagination support to the search endpoint.",
        "Implement filtering by status in the report generation module.",
    ),
    forbidden_actions=(
        "Do not introduce unrelated features or unnecessary abstractions.",
        "Do not silently change existing public interfaces.",
        "Do not bypass validation or authorization requirements.",
        "Do not claim that unimplemented acceptance criteria are complete.",
    ),
)


DEPENDENCY_UPDATE_SKILL = SkillDefinition(
    name="dependency_update_skill",
    instructions=(
        "Update dependencies conservatively. Inspect declared version "
        "constraints, compatibility requirements, and relevant lock or "
        "configuration files before changing dependency declarations."
    ),
    workflow_steps=(
        "Identify the dependency and requested version or update objective.",
        "Inspect dependency declarations, constraints, and lock files.",
        "Check available compatibility information and affected APIs.",
        "Make the smallest necessary dependency changes.",
        "Update related configuration or lock files when required.",
        "Run relevant tests and available dependency consistency checks.",
        "Review the diff for unrelated dependency changes.",
    ),
    example_prompts=(
        "Update the HTTP client dependency to a compatible patched release.",
        "Resolve a dependency compatibility issue without upgrading unrelated packages.",
    ),
    forbidden_actions=(
        "Do not install packages or access the network without authorization.",
        "Do not perform broad dependency upgrades unrelated to the issue.",
        "Do not bypass declared version constraints without justification.",
        "Do not claim compatibility without appropriate verification.",
    ),
)


DOCUMENTATION_SKILL = SkillDefinition(
    name="documentation_skill",
    instructions=(
        "Produce accurate, clear, and maintainable documentation based on "
        "the actual implementation. Preserve established documentation "
        "conventions and verify commands and examples where possible."
    ),
    workflow_steps=(
        "Identify the documentation requested and its intended audience.",
        "Inspect existing documentation and relevant implementation files.",
        "Determine which statements, examples, or instructions need changes.",
        "Update the documentation consistently with project conventions.",
        "Check links, formatting, commands, and examples where possible.",
        "Run relevant documentation checks or tests when available.",
        "Review the diff for accuracy, clarity, and completeness.",
    ),
    example_prompts=(
        "Document the configuration options and their default values.",
        "Update the README with accurate first-time setup instructions.",
    ),
    forbidden_actions=(
        "Do not document behavior that the implementation does not support.",
        "Do not include secrets, credentials, or private tokens.",
        "Do not remove important warnings or setup requirements without justification.",
        "Do not claim that examples were executed unless they were verified.",
    ),
)


CONFIG_CHANGE_SKILL = SkillDefinition(
    name="config_change_skill",
    instructions=(
        "Make configuration changes that preserve secure defaults and "
        "existing validation conventions. Check compatibility, required "
        "environment variables, and behavior for valid and invalid values."
    ),
    workflow_steps=(
        "Identify the requested configuration change and its acceptance criteria.",
        "Inspect existing settings, validation, defaults, and consumers.",
        "Determine compatibility and security implications.",
        "Implement the smallest appropriate configuration change.",
        "Update relevant examples and documentation when necessary.",
        "Test valid values, invalid values, and default behavior.",
        "Review the diff for accidental exposure of secrets or unsafe defaults.",
    ),
    example_prompts=(
        "Add a configurable timeout with a safe default.",
        "Introduce an environment variable for the maximum number of retries.",
    ),
    forbidden_actions=(
        "Do not commit secrets, API keys, or credentials.",
        "Do not weaken security controls to simplify configuration.",
        "Do not silently change defaults in a backward-incompatible way.",
        "Do not bypass configuration validation.",
    ),
)


_SKILLS_BY_CATEGORY: Mapping[IssueCategory, SkillDefinition] = MappingProxyType(
    {
        IssueCategory.BUG_FIX: BUG_FIX_SKILL,
        IssueCategory.FEATURE_ADDITION: FEATURE_ADDITION_SKILL,
        IssueCategory.DEPENDENCY_UPDATE: DEPENDENCY_UPDATE_SKILL,
        IssueCategory.DOCUMENTATION: DOCUMENTATION_SKILL,
        IssueCategory.CONFIG_CHANGE: CONFIG_CHANGE_SKILL,
    }
)


SKILLS: Mapping[str, SkillDefinition] = MappingProxyType(
    {skill.name: skill for skill in _SKILLS_BY_CATEGORY.values()}
)


def get_skill(category: IssueCategory) -> SkillDefinition:
    """Return the skill associated with an issue category."""
    if not isinstance(category, IssueCategory):
        raise TypeError("category must be an IssueCategory.")

    return _SKILLS_BY_CATEGORY[category]


def get_skill_by_name(name: str) -> SkillDefinition:
    """Return a skill by its registered name."""
    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    normalized_name = name.strip()

    try:
        return SKILLS[normalized_name]
    except KeyError as exc:
        raise ValueError(f"Unknown CodePilot skill: {normalized_name!r}.") from exc


def list_skills() -> tuple[SkillDefinition, ...]:
    """Return all registered skills in deterministic order."""
    return tuple(_SKILLS_BY_CATEGORY.values())
