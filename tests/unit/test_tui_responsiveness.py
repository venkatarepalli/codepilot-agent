"""TUI stays responsive: incremental log view and focus restored after approvals."""

import threading

from textual.widgets import TextArea

from codepilot.tui.app import CodePilotApp, LogsScreen


async def test_open_log_view_appends_incrementally_without_full_rebuild(monkeypatch) -> None:
    app = CodePilotApp()
    async with app.run_test(size=(120, 40)) as pilot:
        for index in range(50):
            app.append_log(f"line {index}", source="test")
        await pilot.press("l")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, LogsScreen)

        rebuilds = []
        monkeypatch.setattr(screen, "refresh_entries", lambda entries: rebuilds.append(1))
        for index in range(50, 60):
            app.append_log(f"line {index}", source="test")
        await pilot.pause()

        text = screen.query_one("#full-agent-log", TextArea).text
        assert rebuilds == []  # no full re-render per streamed line
        assert text.splitlines()[0] == "[test] line 0"
        assert text.splitlines()[-1] == "[test] line 59"
        assert len(text.splitlines()) == 60


async def test_log_view_opened_on_empty_history_then_receives_entries() -> None:
    app = CodePilotApp()
    async with app.run_test() as pilot:
        await pilot.press("l")
        await pilot.pause()
        app.append_log("first", source="test")
        app.append_log("second", source="test")
        await pilot.pause()
        text = app.screen.query_one("#full-agent-log", TextArea).text
        assert text.splitlines() == ["[test] first", "[test] second"]


async def test_shortcuts_work_after_a_rejected_approval() -> None:
    app = CodePilotApp()
    async with app.run_test(size=(120, 40)) as pilot:
        result: dict[str, bool] = {}
        worker = threading.Thread(
            target=lambda: result.setdefault(
                "approved", app.request_approval_from_thread("publish_branch", {"x": 1})
            )
        )
        worker.start()
        await pilot.pause(0.2)
        for character in "reject":
            await pilot.press(character)
        await pilot.press("enter")
        await pilot.pause(0.2)
        worker.join(2)

        assert result == {"approved": False}
        assert app.focused is not None and app.focused.id == "agent-log"
        await pilot.press("l")
        await pilot.pause()
        assert isinstance(app.screen, LogsScreen)
