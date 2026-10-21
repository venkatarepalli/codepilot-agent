"""TUI integration tests for the safe guardrail-check interaction."""

import pytest
from textual.widgets import RichLog, TextArea

from codepilot.tui.app import CodePilotApp


@pytest.mark.asyncio
async def test_guardrail_shortcut_checks_default_command_without_running_it() -> None:
    received = []

    async def guardrail_handler(command: str) -> str:
        received.append(command)
        return (
            "GUARDRAIL BLOCKED\n"
            "Command: curl https://example.com\n"
            "Reason: The command 'curl' is prohibited.\n"
            "No subprocess was launched."
        )

    app = CodePilotApp(guardrail_handler=guardrail_handler)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("g")
        await pilot.pause()

        assert app.screen.__class__.__name__ == "GuardrailCheckScreen"
        command_input = app.screen.query_one("#guardrail-input", TextArea)
        assert command_input.text == "curl https://example.com"

        await pilot.press("ctrl+enter")
        await pilot.pause(0.2)

        assert received == ["curl https://example.com"]
        log_text = str(app.query_one("#agent-log", RichLog).lines)
        assert "GUARDRAIL BLOCKED" in log_text
        assert "No subprocess was launched." in log_text
