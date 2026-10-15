"""Tests for OpenAI-compatible GitHub Toolkit tool names."""

from dataclasses import dataclass

from codepilot.runner import _normalize_tool_names


@dataclass
class FakeTool:
    name: str

    def model_copy(self, update: dict[str, str]):
        return FakeTool(name=update["name"])


def test_normalizes_spaces_and_punctuation_without_mutating_original():
    original = FakeTool("create issue/comment!")
    result = _normalize_tool_names([original])

    assert result[0].name == "create_issue_comment_"
    assert original.name == "create issue/comment!"


def test_makes_duplicate_normalized_names_unique():
    result = _normalize_tool_names(
        [FakeTool("get issue"), FakeTool("get/issue"), FakeTool("get_issue")]
    )

    assert [tool.name for tool in result] == [
        "get_issue",
        "get_issue_2",
        "get_issue_3",
    ]


def test_leaves_already_valid_names_unchanged():
    original = FakeTool("get_issue-2")
    result = _normalize_tool_names([original])

    assert result[0] is original
    assert result[0].name == "get_issue-2"


def test_preserves_tool_with_blank_name_for_compatibility():
    original = FakeTool("   ")

    result = _normalize_tool_names([original])

    assert result[0] is original
    assert result[0].name == "   "
