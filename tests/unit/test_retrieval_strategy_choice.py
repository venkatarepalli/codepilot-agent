"""Component 2: the Orchestrator chooses keyword vs embedding retrieval per task."""

import logging
from unittest.mock import MagicMock

import pytest

from codepilot.agents.classifier import IssueCategory
from codepilot.agents.orchestrator import CodePilotOrchestrator, choose_retrieval_strategy
from codepilot.context.context_assembler import ContextAssembler
from codepilot.context.repo_map import FileSummary, RepoMap
from codepilot.github.models import GitHubIssue


def _repo_map() -> RepoMap:
    return RepoMap(
        root=".",
        directory_tree="src/\n  calculator.py\nREADME.md",
        files=(
            FileSummary(path="README.md", language="Markdown", summary="Installation steps."),
            FileSummary(
                path="src/calculator.py", language="Python", summary="Defines: calculate_total"
            ),
        ),
    )


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (IssueCategory.DOCUMENTATION, "keyword"),
        (IssueCategory.CONFIG_CHANGE, "keyword"),
        (IssueCategory.DEPENDENCY_UPDATE, "keyword"),
        (IssueCategory.BUG_FIX, "hybrid"),
        (IssueCategory.FEATURE_ADDITION, "hybrid"),
    ],
)
def test_every_category_has_a_strategy(category, expected) -> None:
    assert choose_retrieval_strategy(category) == expected


def test_assembler_strategy_override_controls_embedding_search() -> None:
    semantic = MagicMock()
    semantic.retrieve.return_value = []
    assembler = ContextAssembler(token_budget=1000, semantic_retriever=semantic)

    assembler.assemble(_repo_map(), "Fix README typo", retrieval_strategy="keyword")
    semantic.retrieve.assert_not_called()

    assembler.assemble(_repo_map(), "calculate_total is wrong", retrieval_strategy="hybrid")
    semantic.retrieve.assert_called_once()

    with pytest.raises(ValueError):
        assembler.assemble(_repo_map(), "x", retrieval_strategy="magic")


@pytest.mark.parametrize(
    ("issue", "strategy", "embedding_used"),
    [
        (
            GitHubIssue(
                number=5,
                title="Fix typo in README installation section",
                body="README says instalation",
                labels=("ai-assignable",),
            ),
            "keyword",
            False,
        ),
        (
            GitHubIssue(
                number=1,
                title="Fix calculate_total to multiply price by quantity",
                body="calculate_total returns the wrong total",
                labels=("bug",),
            ),
            "hybrid",
            True,
        ),
    ],
)
def test_orchestrator_chooses_strategy_and_logs_it(caplog, issue, strategy, embedding_used) -> None:
    semantic = MagicMock()
    semantic.retrieve.return_value = []
    orchestrator = CodePilotOrchestrator(
        model=object(),
        repository="venkatarepalli/codepilot-smoke-test",
        agent_factory=lambda **kwargs: MagicMock(),
        context_assembler=ContextAssembler(token_budget=1000, semantic_retriever=semantic),
        repo_map_provider=lambda *args: _repo_map(),
    )

    with caplog.at_level(logging.INFO, logger="codepilot"):
        orchestrator.prepare_task(issue)

    assert f"Retrieval strategy for issue #{issue.number}" in caplog.text
    assert f": {strategy}." in caplog.text
    assert semantic.retrieve.called is embedding_used
