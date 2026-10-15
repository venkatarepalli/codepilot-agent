"""Tests for structured Test Agent result models."""

from dataclasses import FrozenInstanceError

import pytest

from codepilot.agents.test_models import TestResult, TestStatus
from codepilot.sandbox.execution_models import (
    CommandExecutionResult,
    CommandStatus,
)


def make_command_result(
    status: CommandStatus,
) -> CommandExecutionResult:
    """Create a valid command result for testing."""
    succeeded = status == CommandStatus.SUCCEEDED
    failed = status == CommandStatus.FAILED

    return CommandExecutionResult(
        status=status,
        command=("pytest", "-q"),
        return_code=0 if succeeded else 1 if failed else None,
        stdout="All tests passed" if succeeded else "",
        stderr="Test execution failed" if failed else "",
        duration_seconds=0.5,
        reason=(
            "Command could not execute normally."
            if status
            in {
                CommandStatus.BLOCKED,
                CommandStatus.REQUIRES_APPROVAL,
                CommandStatus.TIMED_OUT,
            }
            else ""
        ),
    )


def test_passed_result_requires_successful_command() -> None:
    """A passed test result must contain a successful command result."""
    command_result = make_command_result(CommandStatus.SUCCEEDED)

    result = TestResult(
        issue_number=42,
        attempt=1,
        status=TestStatus.PASSED,
        command_result=command_result,
        summary="All tests passed.",
    )

    assert result.status == TestStatus.PASSED
    assert result.issue_number == 42
    assert result.attempt == 1
    assert result.command_result is not None
    assert result.command_result.succeeded


def test_passed_result_rejects_missing_command_result() -> None:
    """A passed result cannot exist without command execution evidence."""
    with pytest.raises(ValueError, match="command result"):
        TestResult(
            issue_number=42,
            attempt=1,
            status=TestStatus.PASSED,
            command_result=None,
            summary="All tests passed.",
        )


def test_failed_result_requires_failed_command() -> None:
    """A failed test result must match a failed command execution."""
    result = TestResult(
        issue_number=42,
        attempt=1,
        status=TestStatus.FAILED,
        command_result=make_command_result(CommandStatus.FAILED),
        summary="Two tests failed.",
    )

    assert result.status == TestStatus.FAILED
    assert result.command_result is not None
    assert not result.command_result.succeeded


def test_failed_result_rejects_successful_command() -> None:
    """A failed test result cannot contradict a successful command."""
    with pytest.raises(ValueError, match="match"):
        TestResult(
            issue_number=42,
            attempt=1,
            status=TestStatus.FAILED,
            command_result=make_command_result(CommandStatus.SUCCEEDED),
            summary="Tests failed.",
        )


@pytest.mark.parametrize(
    ("test_status", "command_status"),
    [
        (TestStatus.BLOCKED, CommandStatus.BLOCKED),
        (
            TestStatus.REQUIRES_APPROVAL,
            CommandStatus.REQUIRES_APPROVAL,
        ),
        (TestStatus.TIMED_OUT, CommandStatus.TIMED_OUT),
    ],
)
def test_non_success_statuses_match_command_status(
    test_status: TestStatus,
    command_status: CommandStatus,
) -> None:
    """Blocked, approval, and timeout results preserve execution status."""
    result = TestResult(
        issue_number=42,
        attempt=1,
        status=test_status,
        command_result=make_command_result(command_status),
        summary="Test command could not complete normally.",
    )

    assert result.status == test_status
    assert result.command_result is not None
    assert result.command_result.status == command_status


def test_blocked_result_rejects_mismatched_command_status() -> None:
    """A blocked result cannot describe a command that succeeded."""
    with pytest.raises(ValueError, match="match"):
        TestResult(
            issue_number=42,
            attempt=1,
            status=TestStatus.BLOCKED,
            command_result=make_command_result(CommandStatus.SUCCEEDED),
            summary="Command was blocked.",
        )


def test_error_result_requires_error_details() -> None:
    """An execution error must explain what went wrong."""
    result = TestResult(
        issue_number=42,
        attempt=1,
        status=TestStatus.ERROR,
        command_result=None,
        summary="Test execution encountered an exception.",
        error="Sandbox executor raised RuntimeError.",
    )

    assert result.status == TestStatus.ERROR
    assert result.error == "Sandbox executor raised RuntimeError."


def test_error_result_rejects_empty_error_details() -> None:
    """An error result cannot contain empty error details."""
    with pytest.raises(ValueError, match="error"):
        TestResult(
            issue_number=42,
            attempt=1,
            status=TestStatus.ERROR,
            command_result=None,
            summary="Test execution failed.",
            error=" ",
        )


@pytest.mark.parametrize("issue_number", [0, -1])
def test_issue_number_must_be_positive(issue_number: int) -> None:
    """Issue numbers must be positive."""
    with pytest.raises(ValueError, match="issue"):
        TestResult(
            issue_number=issue_number,
            attempt=1,
            status=TestStatus.ERROR,
            command_result=None,
            summary="Execution error.",
            error="Executor failed.",
        )


@pytest.mark.parametrize("attempt", [0, 4, -1])
def test_attempt_must_be_between_one_and_three(attempt: int) -> None:
    """Test attempts must respect the Coder Agent's three-attempt limit."""
    with pytest.raises(ValueError, match="attempt"):
        TestResult(
            issue_number=42,
            attempt=attempt,
            status=TestStatus.ERROR,
            command_result=None,
            summary="Execution error.",
            error="Executor failed.",
        )


def test_result_is_immutable() -> None:
    """Test results should be immutable once created."""
    result = TestResult(
        issue_number=42,
        attempt=1,
        status=TestStatus.ERROR,
        command_result=None,
        summary="Execution error.",
        error="Executor failed.",
    )

    with pytest.raises(FrozenInstanceError):
        result.attempt = 2  # type: ignore[misc]
