"""GitHub integration for issues and pull-request operations."""

from __future__ import annotations

import logging
import re
from typing import Any

from github import Auth, Github
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper

from codepilot.config.settings import ConfigurationError, Settings
from codepilot.github.models import GitHubIssue

logger = logging.getLogger(__name__)

_CODEPILOT_ISSUE_BRANCH = re.compile(r"^codepilot/issue-(\d+)(?:-|$)", re.IGNORECASE)

def _normalize_github_tool_names(tools: list[Any]) -> list[Any]:
    """Normalize GitHub toolkit names for OpenAI-compatible function calling."""
    for tool in tools:
        name = getattr(tool, "name", None)
        if isinstance(name, str):
            normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
            if not normalized:
                raise ValueError("GitHub tool name cannot be empty after normalization.")
            tool.name = normalized
    return tools


class GitHubClient:
    """Retrieve issues and manage pull requests through GitHub's API."""

    def __init__(
        self,
        settings: Settings,
        *,
        api_wrapper: GitHubAPIWrapper | None = None,
        toolkit: GitHubToolkit | None = None,
    ) -> None:
        """Initialize the GitHub integration using a PAT-backed PyGithub client."""
        self._settings = settings

        if api_wrapper is None:
            if not settings.github_token:
                raise ConfigurationError("GITHUB_TOKEN is required for GitHub operations.")
            if not settings.github_repository_owner or not settings.github_repository_name:
                raise ConfigurationError(
                    "GITHUB_REPOSITORY_OWNER and GITHUB_REPOSITORY_NAME are required "
                    "for GitHub operations."
                )

            repository_name = (
                f"{settings.github_repository_owner}/{settings.github_repository_name}"
            )
            github = Github(auth=Auth.Token(settings.github_token))
            repository = github.get_repo(repository_name)

            # The LangChain wrapper's validator requires GitHub App credentials.
            # Supply its expected fields around the PAT-authenticated PyGithub client.
            api_wrapper = GitHubAPIWrapper.model_construct(
                github=github,
                github_repo_instance=repository,
                github_repository=repository_name,
                active_branch=settings.github_default_branch or repository.default_branch,
                github_base_branch=settings.github_default_branch or repository.default_branch,
            )

        self._api_wrapper = api_wrapper
        self._toolkit = toolkit or GitHubToolkit.from_github_api_wrapper(self._api_wrapper)
        self._tools = _normalize_github_tool_names(self._toolkit.get_tools())

    @property
    def toolkit(self) -> GitHubToolkit:
        """Expose the toolkit for integration with LangChain agents."""
        return self._toolkit

    @property
    def tools(self) -> list[Any]:
        """Return GitHub tools with OpenAI-compatible function names."""
        return self._tools

    @property
    def repository(self) -> Any:
        """Expose the configured PyGithub repository instance."""
        return self._api_wrapper.github_repo_instance

    def get_open_issues(self) -> list[GitHubIssue]:
        """Fetch open issues, excluding pull requests."""
        raw_issues = self.repository.get_issues(state="open")
        normalized: list[GitHubIssue] = []

        for issue in raw_issues:
            if getattr(issue, "pull_request", None):
                continue
            normalized.append(
                GitHubIssue(
                    number=issue.number,
                    title=issue.title,
                    body=issue.body or "",
                    url=issue.html_url,
                    labels=tuple(label.name for label in issue.labels),
                    reporter=issue.user.login if issue.user else None,
                    assignee=issue.assignee.login if issue.assignee else None,
                )
            )
        return normalized

    def get_open_codepilot_pr_issue_numbers(self) -> set[int]:
        """Return issue numbers with an existing open CodePilot pull request.

        CodePilot's PR workflow uses the stable branch convention
        ``codepilot/issue-{number}-{slug}``. Requiring the matching
        ``[CodePilot]`` title avoids matching unrelated branches.
        """
        issue_numbers: set[int] = set()
        for pull_request in self.repository.get_pulls(state="open"):
            title = getattr(pull_request, "title", "") or ""
            if not isinstance(title, str) or not title.casefold().startswith("[codepilot]"):
                continue

            head_ref = getattr(getattr(pull_request, "head", None), "ref", "")
            if not isinstance(head_ref, str):
                continue

            match = _CODEPILOT_ISSUE_BRANCH.match(head_ref)
            if match:
                issue_numbers.add(int(match.group(1)))

        return issue_numbers

    def create_pull_request(
        self,
        *,
        title: str,
        body: str,
        head: str,
        base: str,
        labels: tuple[str, ...] = ("codepilot-generated", "needs-review"),
        reviewer: str | None = None,
        warning_collector: list[str] | None = None,
        draft: bool = False,
    ) -> Any:
        """Create a pull request from an existing branch."""
        for name, value in (
            ("title", title),
            ("body", body),
            ("head", head),
            ("base", base),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty.")
        if head == base:
            raise ValueError("The head and base branches must be different.")

        pull_request = self.repository.create_pull(
            title=title, body=body, head=head, base=base, draft=draft
        )

        warnings = warning_collector if warning_collector is not None else []

        if labels:
            try:
                pull_request.add_to_labels(*labels)
            except Exception as exc:
                message = f"Failed to apply labels: {exc}"
                warnings.append(message)
                logger.warning(
                    "Pull request #%s was created, but applying labels failed: %s",
                    pull_request.number,
                    exc,
                )

        if reviewer:
            requested_reviewer = reviewer.strip()
            pull_request_author = getattr(
                getattr(pull_request, "user", None),
                "login",
                None,
            )

            if (
                requested_reviewer
                and pull_request_author
                and requested_reviewer.casefold() == pull_request_author.casefold()
            ):
                message = (
                    f"Skipped reviewer '{requested_reviewer}': "
                    "that account is the pull request author."
                )
                warnings.append(message)
                logger.warning(
                    "Pull request #%s was created; skipping review request for '%s' "
                    "because that account is the pull request author.",
                    pull_request.number,
                    requested_reviewer,
                )
            elif requested_reviewer:
                try:
                    pull_request.create_review_request(reviewers=[requested_reviewer])
                except Exception as exc:
                    message = f"Failed to request reviewer '{requested_reviewer}': {exc}"
                    warnings.append(message)
                    logger.warning(
                        "Pull request #%s was created, but requesting reviewer '%s' failed: %s",
                        pull_request.number,
                        requested_reviewer,
                        exc,
                    )

        return pull_request

    def is_pull_request_merged(self, pull_number: int) -> bool:
        """Return whether a pull request has been merged."""
        if pull_number < 1:
            raise ValueError("pull_number must be a positive integer.")
        return bool(self.repository.get_pull(pull_number).is_merged())

    def get_pull_request(self, pull_number: int) -> Any:
        """Retrieve a pull request by its number."""
        if pull_number < 1:
            raise ValueError("pull_number must be a positive integer.")
        return self.repository.get_pull(pull_number)
