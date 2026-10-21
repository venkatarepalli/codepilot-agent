"""Tests for the non-executing sandbox guardrail preview."""

from codepilot import runner


def _checker():
    checker = getattr(runner, "_check_guardrail_command", None)
    assert callable(checker), "guardrail preview helper has not been implemented"
    return checker


def test_curl_is_blocked_without_launching_a_subprocess(monkeypatch) -> None:
    """The prohibited command is rejected by policy before any process launch."""

    def must_not_launch(*args, **kwargs):
        raise AssertionError("guardrail preview must never launch a subprocess")

    monkeypatch.setattr("codepilot.sandbox.executor.subprocess.run", must_not_launch)

    result = _checker()("curl https://example.com")

    assert "BLOCKED" in result
    assert "curl" in result.lower()
    assert "no subprocess was launched" in result.lower()


def test_allowed_command_is_reported_but_never_executed(monkeypatch) -> None:
    """The guardrail UI is a policy preview, not a general-purpose shell."""

    def must_not_launch(*args, **kwargs):
        raise AssertionError("guardrail preview must never launch a subprocess")

    monkeypatch.setattr("codepilot.sandbox.executor.subprocess.run", must_not_launch)

    result = _checker()("python --version")

    assert "allows" in result.lower()
    assert "not executed" in result.lower()


def test_approval_required_command_uses_hitl_but_is_not_executed(monkeypatch) -> None:
    """Approval is surfaced through HITL, but preview mode still never runs it."""

    def must_not_launch(*args, **kwargs):
        raise AssertionError("guardrail preview must never launch a subprocess")

    monkeypatch.setattr("codepilot.sandbox.executor.subprocess.run", must_not_launch)
    requests = []

    def approve(operation, details):
        requests.append((operation, details))
        return True

    result = _checker()("git push origin main", approval_callback=approve)

    assert requests
    assert requests[0][0] == "execute_command_requires_approval"
    assert "approval granted" in result.lower()
    assert "not executed" in result.lower()
