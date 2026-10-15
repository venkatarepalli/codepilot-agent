"""Tests for the Test Agent."""

from unittest.mock import Mock

import pytest

from codepilot.agents.test_agent import TestAgent
from codepilot.agents.test_models import TestResult, TestStatus
from codepilot.sandbox.execution_models import (
    CommandExecutionResult,
    CommandStatus,
)


def make_command_result(
    status: CommandStatus,
) -> CommandExecutionResult:
    """Create a valid command execution result."""
    succeeded = status == CommandStatus.SUCCEEDED
    failed = status == CommandStatus.FAILED

    reason = (
        "Command could not execute normally."
        if status
        in {
            CommandStatus.BLOCKED,
            CommandStatus.REQUIRES_APPROVAL,
            CommandStatus.TIMED_OUT,
        }
        else ""
    )

    return CommandExecutionResult(
        status=status,
        command=("pytest", "-q"),
        return_code=0 if succeeded else 1 if failed else None,
        stdout="4 passed" if succeeded else "",
        stderr="2 tests failed" if failed else "",
        duration_seconds=0.25,
        reason=reason,
    )


@pytest.fixture
def workspace() -> Mock:
    """Create a mock workspace with a controllable execution result."""
    return Mock()


@pytest.fixture
def test_agent(workspace: Mock) -> TestAgent:
    """Create a Test Agent using the mock workspace."""
    return TestAgent(workspace=workspace)


def test_successful_execution_returns_passed_result(
    test_agent: TestAgent,
    workspace: Mock,
) -> None:
    """Successful test execution produces a structured passed result."""
    workspace.execute.return_value = make_command_result(CommandStatus.SUCCEEDED)

    result = test_agent.run(
        issue_number=42,
        attempt=1,
        workspace_path=Mock(),
        test_command=("pytest", "-q"),
    )

    assert isinstance(result, TestResult)
    assert result.status == TestStatus.PASSED
    assert result.issue_number == 42
    assert result.attempt == 1
    assert result.command_result is not None
    assert result.command_result.stdout == "4 passed"
    workspace.execute.assert_called_once()


def test_failed_execution_returns_failed_result(
    test_agent: TestAgent,
    workspace: Mock,
) -> None:
    """A failed test command produces a structured failed result."""
    workspace.execute.return_value = make_command_result(CommandStatus.FAILED)

    result = test_agent.run(
        issue_number=42,
        attempt=2,
        workspace_path=Mock(),
        test_command=("pytest", "-q"),
    )

    assert result.status == TestStatus.FAILED
    assert result.issue_number == 42
    assert result.attempt == 2
    assert result.command_result is not None
    assert result.command_result.return_code == 1


@pytest.mark.parametrize(
    ("command_status", "expected_test_status"),
    [
        (CommandStatus.BLOCKED, TestStatus.BLOCKED),
        (
            CommandStatus.REQUIRES_APPROVAL,
            TestStatus.REQUIRES_APPROVAL,
        ),
        (CommandStatus.TIMED_OUT, TestStatus.TIMED_OUT),
    ],
)
def test_non_success_execution_statuses_are_preserved(
    test_agent: TestAgent,
    workspace: Mock,
    command_status: CommandStatus,
    expected_test_status: TestStatus,
) -> None:
    """The Test Agent preserves sandbox restrictions and timeout outcomes."""
    workspace.execute.return_value = make_command_result(command_status)

    result = test_agent.run(
        issue_number=17,
        attempt=1,
        workspace_path=Mock(),
        test_command=("pytest", "-q"),
    )

    assert result.status == expected_test_status
    assert result.command_result is not None
    assert result.command_result.status == command_status


def test_execution_exception_returns_error_result(
    test_agent: TestAgent,
    workspace: Mock,
) -> None:
    """An execution exception becomes a structured error result."""
    workspace.execute.side_effect = RuntimeError("Sandbox unavailable")

    result = test_agent.run(
        issue_number=42,
        attempt=1,
        workspace_path=Mock(),
        test_command=("pytest", "-q"),
    )

    assert result.status == TestStatus.ERROR
    assert result.error is not None
    assert "Sandbox unavailable" in result.error
    assert result.command_result is None


def test_test_agent_does_not_retry(
    test_agent: TestAgent,
    workspace: Mock,
) -> None:
    """Retry decisions belong to the Coder Agent, not the Test Agent."""
    workspace.execute.return_value = make_command_result(CommandStatus.FAILED)

    result = test_agent.run(
        issue_number=42,
        attempt=1,
        workspace_path=Mock(),
        test_command=("pytest", "-q"),
    )

    assert result.status == TestStatus.FAILED
    workspace.execute.assert_called_once()
