"""Component 6: merge conflicts are detected, never auto-resolved, and fail the task."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from github import GithubException

from codepilot.github.pr_workflow import MergeConflictError, PullRequestWorkflow

ORIGINAL = "def total(p, q):\n    return p + q\n"
CODEPILOT = "def total(p, q):\n    return p * q\n"


def _workflow(remote_files: dict[str, str]):
    client = MagicMock()
    repository = client.repository
    repository.get_git_ref.return_value.object.sha = "head-sha"
    repository.create_git_blob.return_value.sha = "blob"
    repository.create_git_commit.return_value.sha = "commit"
    client.create_pull_request.return_value = SimpleNamespace(number=9, html_url="u/9")

    def get_contents(path, ref):
        assert ref == "head-sha"
        if path not in remote_files:
            raise GithubException(404, {"message": "Not Found"}, None)
        return SimpleNamespace(decoded_content=remote_files[path].encode("utf-8"))

    repository.get_contents.side_effect = get_contents
    approvals = MagicMock(return_value=True)
    workflow = PullRequestWorkflow(
        github_client=client, default_branch="main", approval_callback=approvals
    )
    return workflow, client, approvals


def _publish(workflow, base_contents, changes=None):
    return workflow.publish(
        issue_number=3,
        issue_title="Fix total",
        issue_url="https://x/issues/3",
        changes=changes or {"src/calc.py": CODEPILOT},
        commit_message="fix(#3): x",
        summary="s",
        test_results="ok",
        base_contents=base_contents,
    )


def test_unchanged_base_publishes_normally() -> None:
    workflow, client, _ = _workflow({"src/calc.py": ORIGINAL})
    assert _publish(workflow, {"src/calc.py": ORIGINAL}).pull_number == 9
    client.repository.create_git_commit.assert_called_once()


def test_base_changed_since_clone_raises_before_any_approval_or_write() -> None:
    someone_elses_edit = "def total(p, q):\n    return round(p + q, 2)\n"
    workflow, client, approvals = _workflow({"src/calc.py": someone_elses_edit})

    with pytest.raises(MergeConflictError) as raised:
        _publish(workflow, {"src/calc.py": ORIGINAL})

    assert raised.value.paths == ("src/calc.py",)
    assert "Not resolved automatically" in str(raised.value)
    approvals.assert_not_called()
    repository = client.repository
    repository.create_git_blob.assert_not_called()
    repository.create_git_commit.assert_not_called()
    repository.create_git_ref.assert_not_called()
    client.create_pull_request.assert_not_called()


def test_crlf_checkout_is_not_a_conflict() -> None:
    workflow, client, _ = _workflow({"src/calc.py": ORIGINAL})
    _publish(workflow, {"src/calc.py": ORIGINAL.replace("\n", "\r\n")})
    client.create_pull_request.assert_called_once()


def test_new_file_conflicts_only_if_someone_else_added_it() -> None:
    changes = {"docs/new.md": "hello\n"}
    workflow, client, _ = _workflow({})
    _publish(workflow, {"docs/new.md": None}, changes)
    client.create_pull_request.assert_called_once()

    workflow, _, _ = _workflow({"docs/new.md": "someone else's file\n"})
    with pytest.raises(MergeConflictError):
        _publish(workflow, {"docs/new.md": None}, changes)


def test_unverifiable_file_fails_closed() -> None:
    workflow, client, _ = _workflow({})
    client.repository.get_contents.side_effect = GithubException(500, {"message": "boom"}, None)
    with pytest.raises(Exception, match="Could not verify"):
        _publish(workflow, {"src/calc.py": ORIGINAL})
    client.repository.create_git_commit.assert_not_called()


def test_orchestrator_passes_checkout_snapshot_as_base_contents(tmp_path) -> None:
    from codepilot.agents.coder_models import CoderResult, CoderStatus
    from codepilot.agents.orchestrator import CodePilotOrchestrator
    from codepilot.github.models import GitHubIssue
    from codepilot.sandbox.execution_models import CommandExecutionResult, CommandStatus

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text(ORIGINAL, encoding="utf-8")
    coder_workspace = MagicMock()
    coder_workspace.get_changes.return_value = {
        "added": ["docs/new.md"],
        "modified": ["src/calc.py"],
        "deleted": [],
    }
    coder_workspace.read_file.side_effect = lambda _ws, path: f"new {path}"
    pr_workflow = MagicMock()
    orchestrator = CodePilotOrchestrator(
        model="m",
        agent_factory=MagicMock(return_value=MagicMock()),
        pr_workflow=pr_workflow,
        coder_workspace=coder_workspace,
        test_command=("pytest", "-q"),
    )
    issue = GitHubIssue(number=3, title="Fix total", url="https://github.com/o/r/issues/3")
    result = CoderResult(
        status=CoderStatus.SUCCEEDED,
        issue_number=3,
        workspace=MagicMock(source_repository=tmp_path),
        plan="p",
        attempts=1,
        diff_path=tmp_path / "proposed_diff.txt",
        test_result=CommandExecutionResult(
            status=CommandStatus.SUCCEEDED,
            command=("pytest",),
            return_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
        ),
    )

    orchestrator.publish_coder_result(issue, result, commit_message="c", summary="s")

    assert pr_workflow.publish.call_args.kwargs["base_contents"] == {
        "docs/new.md": None,
        "src/calc.py": ORIGINAL,
    }


def test_runner_marks_conflict_failed_and_notifies(monkeypatch, tmp_path) -> None:
    from codepilot import runner
    from codepilot.config.settings import Settings

    orchestrator = MagicMock()
    orchestrator.merge_monitor = None
    orchestrator.run_coding_task.return_value = SimpleNamespace(
        status=runner.CoderStatus.SUCCEEDED,
        attempts=1,
        plan="p",
        test_result=SimpleNamespace(succeeded=True, return_code=0),
        diff_path=tmp_path / "proposed_diff.txt",
        error=None,
    )
    orchestrator.publish_coder_result.side_effect = MergeConflictError(["src/calc.py"], "main")
    polling = MagicMock()
    monkeypatch.setattr(runner, "GitHubClient", lambda _s: MagicMock())
    monkeypatch.setattr(runner, "create_orchestrator", lambda **_k: orchestrator)
    monkeypatch.setattr(runner, "GitHubPollingService", lambda *_a, **_k: polling)
    monkeypatch.setattr(runner, "_clone_repository", lambda _s, d: d.mkdir(parents=True))
    settings = Settings(
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        openai_api_key="k",
        anthropic_api_key=None,
        google_api_key=None,
        github_token="t",
        github_repository_owner="o",
        github_repository_name="r",
        github_default_branch="main",
        poll_interval_minutes=5,
        complexity_threshold=7,
        repo_map_token_budget=4000,
        repo_map_top_k=10,
        max_test_retries=3,
        state_dir=".codepilot",
        chroma_dir=".codepilot/chroma",
    )
    runtime = runner.CodePilotRuntime(settings)
    logs = []
    runtime._log = lambda message, source="Runtime": logs.append((source, message))

    runtime.process_issue(
        SimpleNamespace(number=3, title="Fix total", body="", reporter=None, url="")
    )

    assert runtime._issue_statuses[3] == "FAILED"
    assert any(src == "PR Agent" and "MERGE CONFLICT" in msg for src, msg in logs)
    polling.release_issue.assert_called_once_with(3)
