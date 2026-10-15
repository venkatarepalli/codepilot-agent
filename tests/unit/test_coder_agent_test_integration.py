"""Integration tests for Coder Agent and Test Agent coordination."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from codepilot.agents.coder import CoderAgent
from codepilot.agents.coder_models import CoderStatus
from codepilot.agents.test_agent import TestAgent
from codepilot.agents.test_models import TestResult, TestStatus
from codepilot.github.models import GitHubIssue
from codepilot.sandbox.execution_models import CommandExecutionResult, CommandStatus
from codepilot.sandbox.workspace import WorkspaceInfo


@pytest.fixture
def issue() -> GitHubIssue:
    return GitHubIssue(
        number=42,
        title="Fix calculation bug",
        body="Correct the calculation and add regression coverage.",
        labels=("bug",),
    )


@pytest.fixture
def workspace_info(tmp_path: Path) -> WorkspaceInfo:
    source_repository = tmp_path / "repository"
    source_repository.mkdir()
    (source_repository / "app.py").write_text("def calculate():\n    return 1\n", encoding="utf-8")
    workspace_path = tmp_path / "workspaces" / "issue-42"
    workspace_path.mkdir(parents=True)
    (workspace_path / "app.py").write_text("def calculate():\n    return 2\n", encoding="utf-8")
    return WorkspaceInfo(
        issue_number=42,
        path=workspace_path,
        source_repository=source_repository,
    )


@pytest.fixture
def workspace_manager(workspace_info: WorkspaceInfo) -> Mock:
    manager = Mock()
    manager.create.return_value = workspace_info
    manager.resolve_path.return_value = workspace_info.path / "working" / "proposed_diff.txt"
    manager.get_changes.return_value = {"modified": ["app.py"], "added": [], "deleted": []}

    def write_file(workspace: WorkspaceInfo, relative_path: str, content: str) -> Path:
        destination = workspace.path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return destination

    manager.write_file.side_effect = write_file
    return manager


@pytest.fixture
def test_agent() -> Mock:
    return Mock(spec=TestAgent)


@pytest.fixture
def coder_agent(workspace_manager: Mock, test_agent: Mock) -> CoderAgent:
    return CoderAgent(
        workspace_manager,
        planner=lambda issue, workspace: "Update calculate() and verify the result.",
        implementer=Mock(),
        test_command=("pytest", "-q"),
        test_agent=test_agent,
    )


def command_result(
    status: CommandStatus,
    *,
    return_code: int | None,
    stdout: str = "",
    stderr: str = "",
    reason: str | None = None,
) -> CommandExecutionResult:
    return CommandExecutionResult(
        status=status,
        command=("pytest", "-q"),
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=0.2,
        reason=reason,
    )


def successful_command_result() -> CommandExecutionResult:
    return command_result(CommandStatus.SUCCEEDED, return_code=0, stdout="3 passed")


def failed_command_result() -> CommandExecutionResult:
    return command_result(CommandStatus.FAILED, return_code=1, stderr="1 failed")


def make_test_result(
    status: TestStatus,
    *,
    attempt: int = 1,
    command_result: CommandExecutionResult | None = None,
    error: str | None = None,
) -> TestResult:
    if status == TestStatus.BLOCKED:
        command_result = command_result or globals()["command_result"](
            CommandStatus.BLOCKED, return_code=None, reason="Blocked by policy"
        )
    elif status == TestStatus.REQUIRES_APPROVAL:
        command_result = command_result or globals()["command_result"](
            CommandStatus.REQUIRES_APPROVAL, return_code=None, reason="Approval required"
        )
    elif status == TestStatus.TIMED_OUT:
        command_result = command_result or globals()["command_result"](
            CommandStatus.TIMED_OUT, return_code=None, reason="Timed out"
        )
    elif status == TestStatus.ERROR:
        command_result = None
        error = error or "Test execution raised an exception."

    return TestResult(
        issue_number=42,
        attempt=attempt,
        status=status,
        command_result=command_result,
        summary=f"Tests {status.value}.",
        error=error,
    )


def test_coder_agent_uses_default_test_agent_when_not_injected(
    workspace_manager: Mock,
) -> None:
    agent = CoderAgent(
        workspace_manager,
        planner=lambda issue, workspace: "Plan",
        implementer=Mock(),
        test_command=("pytest", "-q"),
    )
    assert isinstance(agent.test_agent, TestAgent)


def test_successful_test_result_produces_successful_coding_result(
    coder_agent: CoderAgent,
    test_agent: Mock,
    issue: GitHubIssue,
    workspace_info: WorkspaceInfo,
) -> None:
    test_agent.run.return_value = make_test_result(
        TestStatus.PASSED, command_result=successful_command_result()
    )
    result = coder_agent.run(issue, workspace_info.source_repository)
    assert result.status == CoderStatus.SUCCEEDED
    assert result.attempts == 1
    assert result.diff_path is not None and result.diff_path.exists()
    assert result.test_assessment is not None
    test_agent.run.assert_called_once_with(
        issue_number=42,
        attempt=1,
        workspace_info=workspace_info,
        command=("pytest", "-q"),
    )


def test_failed_test_result_triggers_an_implementation_retry(
    coder_agent: CoderAgent,
    test_agent: Mock,
    issue: GitHubIssue,
    workspace_info: WorkspaceInfo,
) -> None:
    test_agent.run.side_effect = [
        make_test_result(TestStatus.FAILED, command_result=failed_command_result()),
        make_test_result(TestStatus.PASSED, attempt=2, command_result=successful_command_result()),
    ]
    result = coder_agent.run(issue, workspace_info.source_repository)
    assert result.status == CoderStatus.SUCCEEDED
    assert result.attempts == 2
    assert test_agent.run.call_count == 2


def test_three_failed_test_attempts_stop_the_coding_workflow(
    coder_agent: CoderAgent,
    test_agent: Mock,
    issue: GitHubIssue,
    workspace_info: WorkspaceInfo,
) -> None:
    test_agent.run.side_effect = [
        make_test_result(
            TestStatus.FAILED,
            attempt=attempt,
            command_result=failed_command_result(),
        )
        for attempt in range(1, 4)
    ]
    result = coder_agent.run(issue, workspace_info.source_repository)
    assert result.status == CoderStatus.FAILED
    assert result.attempts == 3
    assert test_agent.run.call_count == 3
    assert "3 implementation attempts" in result.error


@pytest.mark.parametrize(
    "status",
    [
        TestStatus.BLOCKED,
        TestStatus.REQUIRES_APPROVAL,
        TestStatus.TIMED_OUT,
        TestStatus.ERROR,
    ],
)
def test_non_retryable_test_outcomes_stop_the_workflow(
    coder_agent: CoderAgent,
    test_agent: Mock,
    issue: GitHubIssue,
    workspace_info: WorkspaceInfo,
    status: TestStatus,
) -> None:
    test_agent.run.return_value = make_test_result(status)
    result = coder_agent.run(issue, workspace_info.source_repository)
    assert result.status == CoderStatus.FAILED
    assert result.attempts == 1
    assert test_agent.run.call_count == 1


def test_implementer_receives_previous_failed_test_result(
    coder_agent: CoderAgent,
    test_agent: Mock,
    issue: GitHubIssue,
    workspace_info: WorkspaceInfo,
) -> None:
    implementer = coder_agent.implementer
    failed_result = make_test_result(TestStatus.FAILED, command_result=failed_command_result())
    test_agent.run.side_effect = [
        failed_result,
        make_test_result(TestStatus.PASSED, attempt=2, command_result=successful_command_result()),
    ]
    coder_agent.run(issue, workspace_info.source_repository)
    assert implementer.call_count == 2
    assert implementer.call_args_list[0].args[3] == 1
    assert implementer.call_args_list[0].args[4] is None
    assert implementer.call_args_list[1].args[3] == 2
    assert implementer.call_args_list[1].args[4] == failed_result
