"""Tests for the CodePilot four-panel Textual interface."""

from __future__ import annotations

import asyncio
from threading import Thread

import pytest
from textual.widgets import Input, RichLog, Static, TextArea

from codepilot.tui.app import CodePilotApp, LogsScreen


@pytest.mark.asyncio
async def test_dashboard_has_four_required_panels_with_uniform_dimensions() -> None:
    app = CodePilotApp()
    async with app.run_test(size=(120, 40)) as pilot:
        assert app.query_one("#issues-content", Static)
        assert app.query_one("#task-content", Static)
        assert app.query_one("#agent-log", RichLog)
        assert app.query_one("#approval-content", Static)
        assert app.query_one("#approval-command", Input).disabled
        await pilot.pause()

        issue_panel = app.query_one("#issues-panel")
        task_panel = app.query_one("#task-panel")
        logs_panel = app.query_one("#logs-panel")
        approval_panel = app.query_one("#approval-panel")
        assert issue_panel.size.width == task_panel.size.width
        assert logs_panel.size.width == approval_panel.size.width
        assert issue_panel.size.height == logs_panel.size.height
        assert task_panel.size.height == approval_panel.size.height


@pytest.mark.asyncio
async def test_issue_task_and_agent_log_updates_render() -> None:
    app = CodePilotApp()
    async with app.run_test() as pilot:
        app.update_issues(
            [
                {"number": 42, "title": "Fix null pointer", "status": "open"},
                {"number": 38, "title": "Improve docs", "status": "in-progress"},
            ]
        )
        app.update_active_task(
            {
                "title": "Issue #42: Fix null pointer",
                "status": "IMPLEMENTING",
                "agent": "Coder (retry 1/3)",
                "skill": "bug_fix_skill",
                "todos": [{"text": "Reproduce", "done": True}, {"text": "Fix", "done": False}],
            }
        )
        app.append_log("Found 8 relevant files", source="RepoExplorer")
        await pilot.pause()
        assert "#42" in str(app.query_one("#issues-content", Static).render())
        task_text = str(app.query_one("#task-content", Static).render())
        assert "IMPLEMENTING" in task_text
        assert "bug_fix_skill" in task_text
        assert "Reproduce" in task_text
        assert "Found 8 relevant files" in str(app.query_one("#agent-log", RichLog).lines)


@pytest.mark.asyncio
async def test_approval_requires_explicit_approve_or_reject() -> None:
    app = CodePilotApp()
    async with app.run_test() as pilot:
        approval = app.run_worker(
            app.request_approval("open_pull_request", {"base": "main", "files_changed": 5}),
            name="test-approval",
        )
        await pilot.pause()
        command = app.query_one("#approval-command", Input)
        assert not command.disabled
        command.value = "inspect"
        await pilot.press("enter")
        await pilot.pause()
        assert not approval.is_finished
        command.value = "reject"
        await pilot.press("enter")
        await pilot.pause()
        assert approval.is_finished
        assert approval.result is False


@pytest.mark.asyncio
async def test_new_task_shortcut_opens_multiline_task_modal() -> None:
    app = CodePilotApp()
    async with app.run_test() as pilot:
        await pilot.press("i")
        await pilot.pause()
        assert app.screen.__class__.__name__ == "NewTaskScreen"
        assert app.screen.query_one("#task-input", TextArea)


@pytest.mark.asyncio
async def test_direct_task_accepts_multiline_input_in_one_submission() -> None:
    submitted: list[str] = []

    async def task_handler(task: str) -> str:
        submitted.append(task)
        return "Task was received."

    app = CodePilotApp(task_handler=task_handler)
    task_text = (
        "Fix src/calculator.py.\n"
        "Keep tests unchanged.\n"
        "Run the test suite.\n"
        "Show the proposed diff."
    )

    async with app.run_test() as pilot:
        await pilot.press("i")
        await pilot.pause()
        app.screen.query_one("#task-input", TextArea).text = task_text
        await pilot.press("ctrl+enter")
        await pilot.pause(0.1)

        assert submitted == [task_text]


@pytest.mark.asyncio
async def test_runtime_thread_can_wait_for_tui_approval() -> None:
    app = CodePilotApp()
    decision: list[bool] = []

    async with app.run_test() as pilot:
        worker = Thread(
            target=lambda: decision.append(
                app.request_approval_from_thread("publish_branch", {"base_branch": "main"})
            ),
            daemon=True,
        )
        worker.start()
        await pilot.pause()

        command = app.query_one("#approval-command", Input)
        assert not command.disabled
        command.value = "approve"
        await pilot.press("enter")
        await asyncio.to_thread(worker.join, 2)

        assert not worker.is_alive()
        assert decision == [True]


@pytest.mark.asyncio
async def test_logs_shortcut_opens_full_screen_view_and_keeps_live_history(tmp_path) -> None:
    app = CodePilotApp(log_directory=tmp_path)

    async with app.run_test(size=(120, 40)) as pilot:
        app.append_log('Response preview: {"changes": []}', source="Coder")
        await pilot.press("l")
        await pilot.pause()

        assert isinstance(app.screen, LogsScreen)
        log_view = app.screen.query_one("#full-agent-log", TextArea)
        assert log_view.read_only is True
        assert "Response preview" in log_view.text

        app.append_log("Latest runtime detail", source="Runtime")
        app.update_active_task({"title": "Background task", "status": "IMPLEMENTING"})
        await pilot.pause()
        assert "Latest runtime detail" in log_view.text
        assert "IMPLEMENTING" in str(app._dashboard_task_widget.render())

        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, LogsScreen)


@pytest.mark.asyncio
async def test_full_screen_logs_can_copy_all_entries(monkeypatch, tmp_path) -> None:
    app = CodePilotApp(log_directory=tmp_path)
    copied: list[str] = []
    monkeypatch.setattr(CodePilotApp, "copy_to_clipboard", lambda _self, text: copied.append(text))

    async with app.run_test() as pilot:
        app.append_log("Implementation failed: multiple JSON change sets", source="Coder")
        await pilot.press("l")
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()

        assert len(copied) == 1
        assert "multiple JSON change sets" in copied[0]


@pytest.mark.asyncio
async def test_full_screen_logs_can_save_complete_log_file(tmp_path) -> None:
    app = CodePilotApp(log_directory=tmp_path)

    async with app.run_test() as pilot:
        app.append_log("First diagnostic line", source="Coder")
        app.append_log('Response preview: {"changes": []}', source="Coder")
        await pilot.press("l")
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()

        saved_files = list(tmp_path.glob("session-*.log"))
        assert len(saved_files) == 1
        saved_text = saved_files[0].read_text(encoding="utf-8")
        assert "First diagnostic line" in saved_text
        assert "Response preview" in saved_text
