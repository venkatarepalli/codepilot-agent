"""Prepare and publish CodePilot changes as GitHub pull requests."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from github import InputGitTreeElement

from codepilot.github.client import GitHubClient

ApprovalCallback = Callable[[str, Mapping[str, Any]], bool]


class PullRequestWorkflowError(RuntimeError):
    """Raised when a pull-request workflow cannot safely continue."""


class HumanApprovalRejectedError(PullRequestWorkflowError):
    """Raised when a required human approval is explicitly rejected."""


@dataclass(frozen=True, slots=True)
class PullRequestResult:
    """Summary of a published CodePilot pull request."""

    branch: str
    pull_number: int
    pull_url: str
    title: str
    base_branch: str
    metadata_warnings: tuple[str, ...] = ()


class PullRequestWorkflow:
    """Publish tested file changes and open a reviewable pull request.

    The workflow uses GitHub's API to create blobs, a tree, a commit,
    a branch reference, and a pull request.

    Human approval is required before publishing the branch, before
    publishing changes affecting more than five files, and before
    opening a pull request against main or master.
    """

    def __init__(
        self,
        *,
        github_client: GitHubClient,
        default_branch: str,
        approval_callback: ApprovalCallback,
    ) -> None:
        """Initialize the workflow with GitHub access and approval handling."""
        if not default_branch.strip():
            raise ValueError("default_branch must not be empty.")

        if not callable(approval_callback):
            raise TypeError("approval_callback must be callable.")

        self._github_client = github_client
        self._default_branch = default_branch.strip()
        self._approval_callback = approval_callback

    def is_merged(self, pull_number: int) -> bool:
        """Return whether a pull request has been merged on GitHub."""
        return self._github_client.is_pull_request_merged(pull_number)

    @staticmethod
    def _slugify(value: str) -> str:
        """Convert an issue title into a branch-friendly slug."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
        return slug[:60].rstrip("-") or "task"

    @staticmethod
    def _require_approval(
        callback: ApprovalCallback,
        operation: str,
        details: Mapping[str, Any],
    ) -> None:
        """Stop the workflow unless the approval callback explicitly approves."""
        try:
            approved = callback(operation, details)
        except Exception as exc:
            raise PullRequestWorkflowError(
                f"Approval callback failed for {operation}: {type(exc).__name__}: {exc}"
            ) from exc
        if approved is not True:
            raise HumanApprovalRejectedError(f"Human approval was not granted for: {operation}.")

    def publish(
        self,
        *,
        issue_number: int,
        issue_title: str,
        issue_url: str,
        changes: Mapping[str, str],
        commit_message: str,
        summary: str,
        test_results: str,
        reviewer: str | None = None,
        base_branch: str | None = None,
    ) -> PullRequestResult:
        """Publish proposed text-file changes and create a pull request.

        Args:
            issue_number: GitHub issue number.
            issue_title: Original issue title.
            issue_url: URL of the original issue.
            changes: Mapping of repository-relative paths to complete file
                contents. These are complete replacement contents, not diffs.
            commit_message: Commit message describing the changes.
            summary: Human-readable summary for the pull-request body.
            test_results: Test results to include in the pull-request body.
            reviewer: Optional GitHub username to request as reviewer.
            base_branch: Target branch; defaults to the configured branch.

        Returns:
            Details of the created pull request.

        Raises:
            ValueError: If required input is invalid.
            PullRequestWorkflowError: If approval is denied or the workflow
                cannot safely publish the proposed changes.
        """
        if issue_number < 1:
            raise ValueError("issue_number must be a positive integer.")

        for name, value in (
            ("issue_title", issue_title),
            ("issue_url", issue_url),
            ("commit_message", commit_message),
            ("summary", summary),
            ("test_results", test_results),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty.")

        if not changes:
            raise ValueError("At least one file change is required.")

        normalized_changes: dict[str, str] = {}

        for path, content in changes.items():
            normalized_path = path.replace("\\", "/").strip()

            if (
                not normalized_path
                or normalized_path.startswith("/")
                or any(part in {"", ".", ".."} for part in normalized_path.split("/"))
                or ":" in normalized_path
            ):
                raise ValueError(f"Invalid repository-relative path: {path!r}")

            if not isinstance(content, str):
                raise TypeError(f"File content must be text: {path!r}")

            normalized_changes[normalized_path] = content

        if len(normalized_changes) != len(changes):
            raise ValueError("File paths must be unique after normalization.")

        base = (base_branch or self._default_branch).strip()

        if not base:
            raise ValueError("base_branch must not be empty.")

        branch = f"codepilot/issue-{issue_number}-{self._slugify(issue_title)}"

        changed_files = sorted(normalized_changes)
        approval_details = {
            "issue_number": issue_number,
            "branch": branch,
            "base_branch": base,
            "files_changed": len(changed_files),
            # Include the exact publication payload in the approval prompt so a
            # human can verify scope instead of trusting a count alone.
            "changed_files": changed_files,
        }

        # A larger change set must be explicitly reviewed before publication.
        if len(normalized_changes) > 5:
            self._require_approval(
                self._approval_callback,
                "publish_changes_over_five_files",
                approval_details,
            )

        # Obtain every required approval before creating any remote Git object.
        # Blob/tree/commit creation is itself a write, even though it does not
        # immediately publish a branch reference. Denials must have no remote
        # side effects.
        self._require_approval(
            self._approval_callback,
            "publish_branch",
            approval_details,
        )

        if base.lower() in {"main", "master"}:
            self._require_approval(
                self._approval_callback,
                "create_pull_request_against_protected_branch",
                approval_details,
            )

        repository = self._github_client.repository

        try:
            base_ref = repository.get_git_ref(f"heads/{base}")
            parent_commit = repository.get_git_commit(base_ref.object.sha)

            tree_entries = []

            for path, content in sorted(normalized_changes.items()):
                blob = repository.create_git_blob(content, "utf-8")

                tree_entries.append(
                    InputGitTreeElement(
                        path=path,
                        mode="100644",
                        type="blob",
                        sha=blob.sha,
                    )
                )

            tree = repository.create_git_tree(
                tree_entries,
                base_tree=parent_commit.tree,
            )

            commit = repository.create_git_commit(
                commit_message,
                tree,
                [parent_commit],
            )

            # Publishing the approved commit starts with creating the branch ref.
            repository.create_git_ref(
                ref=f"refs/heads/{branch}",
                sha=commit.sha,
            )

            pull_title = f"[CodePilot] {issue_title}"

            pull_body = (
                f"## Summary\n{summary}\n\n"
                f"## Approach\n"
                f"Changes were prepared for issue #{issue_number}.\n\n"
                f"## Files changed\n"
                + "\n".join(f"- `{path}`" for path in sorted(normalized_changes))
                + f"\n\n## Tests\n{test_results}\n\n"
                f"## Related issue\nCloses #{issue_number}\n\n"
                f"Issue: {issue_url}"
            )

            metadata_warnings: list[str] = []

            pull_request = self._github_client.create_pull_request(
                title=pull_title,
                body=pull_body,
                head=branch,
                base=base,
                reviewer=reviewer,
                warning_collector=metadata_warnings,
            )

            return PullRequestResult(
                branch=branch,
                pull_number=pull_request.number,
                pull_url=pull_request.html_url,
                title=pull_title,
                base_branch=base,
                metadata_warnings=tuple(metadata_warnings),
            )

        except PullRequestWorkflowError:
            raise
        except Exception as exc:
            raise PullRequestWorkflowError(
                f"Could not publish the pull request for issue #{issue_number}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
