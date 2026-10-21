"""Unit tests for the GitHub client."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from codepilot.config.settings import Settings
from codepilot.github.client import GitHubClient


@pytest.fixture
def settings() -> Settings:
    """Return settings that do not require external credentials."""
    return Settings(
        llm_provider=None,
        llm_model=None,
        openai_api_key=None,
        anthropic_api_key=None,
        google_api_key=None,
        github_token=None,
        github_repository_owner="example-owner",
        github_repository_name="example-repo",
        github_default_branch="main",
        poll_interval_minutes=5,
        complexity_threshold=7,
        repo_map_token_budget=4000,
        repo_map_top_k=10,
        max_test_retries=3,
        state_dir=".codepilot",
        chroma_dir=".codepilot/chroma",
    )


def make_issue(
    *,
    number: int,
    title: str,
    pull_request: bool = False,
    labels: tuple[str, ...] = (),
    reporter: str | None = "reporter",
    assignee: str | None = None,
) -> SimpleNamespace:
    """Create a lightweight GitHub issue test double."""
    return SimpleNamespace(
        number=number,
        title=title,
        body=f"Body for {title}",
        html_url=f"https://github.com/example-owner/example-repo/issues/{number}",
        pull_request={"url": "example"} if pull_request else None,
        labels=[SimpleNamespace(name=name) for name in labels],
        user=SimpleNamespace(login=reporter) if reporter else None,
        assignee=SimpleNamespace(login=assignee) if assignee else None,
    )


def make_client(
    settings: Settings,
    issues: list[SimpleNamespace] | None = None,
) -> GitHubClient:
    """Build a client with mocked GitHub dependencies."""
    wrapper = MagicMock()
    wrapper.github_repo_instance.get_issues.return_value = issues or []

    toolkit = MagicMock()

    return GitHubClient(
        settings,
        api_wrapper=wrapper,
        toolkit=toolkit,
    )


def test_get_open_issues_normalizes_issue_fields(settings: Settings) -> None:
    issue = make_issue(
        number=12,
        title="Fix login",
        labels=("bug", "ai-assignable"),
        assignee="developer",
    )
    client = make_client(settings, [issue])

    result = client.get_open_issues()

    assert len(result) == 1
    assert result[0].number == 12
    assert result[0].title == "Fix login"
    assert result[0].body == "Body for Fix login"
    assert result[0].labels == ("bug", "ai-assignable")
    assert result[0].reporter == "reporter"
    assert result[0].assignee == "developer"


def test_get_open_issues_excludes_pull_requests(settings: Settings) -> None:
    issues = [
        make_issue(number=1, title="Real issue"),
        make_issue(number=2, title="Pull request", pull_request=True),
    ]
    client = make_client(settings, issues)

    result = client.get_open_issues()

    assert [issue.number for issue in result] == [1]


def test_get_open_issues_handles_empty_repository(settings: Settings) -> None:
    client = make_client(settings)

    assert client.get_open_issues() == []


def test_get_open_issues_handles_missing_reporter(settings: Settings) -> None:
    client = make_client(
        settings,
        [make_issue(number=3, title="Issue without reporter", reporter=None)],
    )

    result = client.get_open_issues()

    assert result[0].reporter is None


def test_get_open_issues_requests_only_open_issues(settings: Settings) -> None:
    client = make_client(settings)

    client.get_open_issues()

    client._api_wrapper.github_repo_instance.get_issues.assert_called_once_with(state="open")


def test_repository_returns_configured_repository(settings: Settings) -> None:
    client = make_client(settings)

    assert client.repository is client._api_wrapper.github_repo_instance


def test_create_pull_request_creates_pr_with_labels_and_reviewer(
    settings: Settings,
) -> None:
    client = make_client(settings)
    pull_request = client.repository.create_pull.return_value

    result = client.create_pull_request(
        title="[CodePilot] Fix login",
        body="Fixes login validation.\n\nCloses #12",
        head="codepilot/issue-12-fix-login",
        base="main",
        reviewer="reporter",
    )

    assert result is pull_request

    client.repository.create_pull.assert_called_once_with(
        title="[CodePilot] Fix login",
        body="Fixes login validation.\n\nCloses #12",
        head="codepilot/issue-12-fix-login",
        base="main",
        draft=False,
    )
    pull_request.add_to_labels.assert_called_once_with(
        "codepilot-generated",
        "needs-review",
    )
    pull_request.create_review_request.assert_called_once_with(reviewers=["reporter"])


def test_create_pull_request_supports_custom_labels_and_draft(
    settings: Settings,
) -> None:
    client = make_client(settings)

    client.create_pull_request(
        title="Documentation update",
        body="Update the README.",
        head="docs/update-readme",
        base="main",
        labels=("documentation",),
        draft=True,
    )

    client.repository.create_pull.assert_called_once_with(
        title="Documentation update",
        body="Update the README.",
        head="docs/update-readme",
        base="main",
        draft=True,
    )
    client.repository.create_pull.return_value.add_to_labels.assert_called_once_with(
        "documentation"
    )
    client.repository.create_pull.return_value.create_review_request.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", ""),
        ("body", " "),
        ("head", ""),
        ("base", ""),
    ],
)
def test_create_pull_request_rejects_empty_required_fields(
    settings: Settings,
    field: str,
    value: str,
) -> None:
    client = make_client(settings)
    arguments = {
        "title": "Fix login",
        "body": "Fix the login bug.",
        "head": "codepilot/issue-12-fix-login",
        "base": "main",
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=f"{field} must not be empty"):
        client.create_pull_request(**arguments)

    client.repository.create_pull.assert_not_called()


def test_create_pull_request_rejects_identical_branches(
    settings: Settings,
) -> None:
    client = make_client(settings)

    with pytest.raises(ValueError, match="must be different"):
        client.create_pull_request(
            title="Fix login",
            body="Fix the login bug.",
            head="main",
            base="main",
        )

    client.repository.create_pull.assert_not_called()


@pytest.mark.parametrize("pull_number", [0, -1])
def test_get_pull_request_rejects_invalid_number(
    settings: Settings,
    pull_number: int,
) -> None:
    client = make_client(settings)

    with pytest.raises(ValueError, match="positive integer"):
        client.get_pull_request(pull_number)

    client.repository.get_pull.assert_not_called()


def test_get_pull_request_returns_requested_pull_request(
    settings: Settings,
) -> None:
    client = make_client(settings)
    expected = client.repository.get_pull.return_value

    result = client.get_pull_request(42)

    assert result is expected
    client.repository.get_pull.assert_called_once_with(42)


@pytest.mark.parametrize("pull_number", [0, -1])
def test_is_pull_request_merged_rejects_invalid_number(
    settings: Settings,
    pull_number: int,
) -> None:
    client = make_client(settings)

    with pytest.raises(ValueError, match="positive integer"):
        client.is_pull_request_merged(pull_number)

    client.repository.get_pull.assert_not_called()


@pytest.mark.parametrize(
    ("merged", "expected"),
    [(True, True), (False, False)],
)
def test_is_pull_request_merged_returns_merge_status(
    settings: Settings,
    merged: bool,
    expected: bool,
) -> None:
    client = make_client(settings)
    pull_request = client.repository.get_pull.return_value
    pull_request.is_merged.return_value = merged

    result = client.is_pull_request_merged(42)

    assert result is expected
    client.repository.get_pull.assert_called_once_with(42)
    pull_request.is_merged.assert_called_once_with()

def test_normalize_github_tool_names_preserves_tools_and_normalizes_names():
    """GitHub tool names must satisfy OpenAI function-calling requirements."""
    import re
    from unittest.mock import MagicMock

    from codepilot.github.client import _normalize_github_tool_names

    get_issues = MagicMock()
    get_issues.name = "Get Issues"

    create_pr = MagicMock()
    create_pr.name = "Create Pull Request"

    tools = [get_issues, create_pr]

    result = _normalize_github_tool_names(tools)

    assert result is tools
    assert result[0] is get_issues
    assert result[1] is create_pr
    assert get_issues.name == "Get_Issues"
    assert create_pr.name == "Create_Pull_Request"

    for tool in result:
        assert re.fullmatch(r"[a-zA-Z0-9_-]+", tool.name)
