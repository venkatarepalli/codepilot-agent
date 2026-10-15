"""Unit tests for the CodePilot skill registry."""

import pytest

from codepilot.agents.classifier import IssueCategory
from codepilot.skills.skill_registry import (
    BUG_FIX_SKILL,
    CONFIG_CHANGE_SKILL,
    DEPENDENCY_UPDATE_SKILL,
    DOCUMENTATION_SKILL,
    FEATURE_ADDITION_SKILL,
    SKILLS,
    SkillDefinition,
    get_skill,
    get_skill_by_name,
    list_skills,
)

EXPECTED_SKILLS = (
    BUG_FIX_SKILL,
    FEATURE_ADDITION_SKILL,
    DEPENDENCY_UPDATE_SKILL,
    DOCUMENTATION_SKILL,
    CONFIG_CHANGE_SKILL,
)


def test_registry_contains_five_skills() -> None:
    """The registry should contain all five required skills."""
    skills = list_skills()

    assert len(skills) == 5
    assert len(SKILLS) == 5
    assert {skill.name for skill in skills} == {
        "bug_fix_skill",
        "feature_addition_skill",
        "dependency_update_skill",
        "documentation_skill",
        "config_change_skill",
    }


@pytest.mark.parametrize("skill", EXPECTED_SKILLS)
def test_skill_has_all_required_fields(skill: SkillDefinition) -> None:
    """Every skill should contain all five required fields."""
    assert skill.name
    assert skill.instructions.strip()
    assert skill.workflow_steps
    assert skill.example_prompts
    assert skill.forbidden_actions


@pytest.mark.parametrize(
    ("category", "expected_skill"),
    [
        (IssueCategory.BUG_FIX, BUG_FIX_SKILL),
        (IssueCategory.FEATURE_ADDITION, FEATURE_ADDITION_SKILL),
        (IssueCategory.DEPENDENCY_UPDATE, DEPENDENCY_UPDATE_SKILL),
        (IssueCategory.DOCUMENTATION, DOCUMENTATION_SKILL),
        (IssueCategory.CONFIG_CHANGE, CONFIG_CHANGE_SKILL),
    ],
)
def test_get_skill_returns_correct_skill(
    category: IssueCategory,
    expected_skill: SkillDefinition,
) -> None:
    """Each classifier category should resolve to its matching skill."""
    assert get_skill(category) is expected_skill


@pytest.mark.parametrize("skill", EXPECTED_SKILLS)
def test_get_skill_by_name_returns_correct_skill(
    skill: SkillDefinition,
) -> None:
    """Each registered skill should be retrievable by name."""
    assert get_skill_by_name(skill.name) is skill


def test_get_skill_by_name_accepts_surrounding_whitespace() -> None:
    """Skill lookup should tolerate surrounding whitespace."""
    assert get_skill_by_name("  bug_fix_skill  ") is BUG_FIX_SKILL


def test_get_skill_by_name_rejects_unknown_skill() -> None:
    """Unknown skill names should raise a clear exception."""
    with pytest.raises(ValueError, match="Unknown CodePilot skill"):
        get_skill_by_name("unknown_skill")


def test_get_skill_by_name_rejects_non_string_input() -> None:
    """Skill lookup should reject values that are not strings."""
    with pytest.raises(TypeError, match="name must be a string"):
        get_skill_by_name(None)  # type: ignore[arg-type]


def test_get_skill_rejects_non_category_input() -> None:
    """Category lookup should reject values outside IssueCategory."""
    with pytest.raises(TypeError, match="category must be an IssueCategory"):
        get_skill("bug_fix")  # type: ignore[arg-type]


def test_skill_definitions_are_immutable() -> None:
    """A registered skill's fields should not be reassigned."""
    with pytest.raises(AttributeError):
        BUG_FIX_SKILL.name = "changed"  # type: ignore[misc]


def test_skill_registry_is_read_only() -> None:
    """Callers should not be able to modify the registry mapping."""
    with pytest.raises(TypeError):
        SKILLS["new_skill"] = BUG_FIX_SKILL  # type: ignore[index]


def test_list_skills_returns_a_tuple() -> None:
    """The registry should return an immutable collection."""
    assert isinstance(list_skills(), tuple)
