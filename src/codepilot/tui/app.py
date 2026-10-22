"""Four-panel Textual interface for CodePilot.

The UI is deliberately decoupled from the runtime. The application runner can
feed issue/task/log updates and provide callbacks for direct tasks, issue skips,
and human-approval decisions without coupling agents to Textual widgets.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.events import Resize
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Header, Input, Label, RichLog, Static, TextArea

TaskHandler = Callable[[str], Any | Awaitable[Any]]
GuardrailHandler = Callable[[str], Any | Awaitable[Any]]
ApprovalHandler = Callable[[str, Mapping[str, Any]], Any | Awaitable[Any]]
SkipHandler = Callable[[], Any | Awaitable[Any]]
LifecycleHandler = Callable[[], Any]

logger = logging.getLogger(__name__)


class NewTaskScreen(ModalScreen[str | None]):
    """Modal prompt for a free-form task that is not tied to a GitHub issue."""

    CSS = """
    NewTaskScreen { align: center middle; }
    #task-dialog {
        width: 88%; max-width: 112; height: 70%; min-height: 16;
        padding: 1 2; border: round $accent; background: $panel;
    }
    #task-input { height: 1fr; min-height: 6; margin-top: 1; }
    #task-help { height: 2; color: $text-muted; margin-top: 1; }
    """
    BINDINGS = [
        Binding("ctrl+enter", "submit_task", "Submit task", priority=True),
        Binding("escape", "cancel_task", "Cancel", priority=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="task-dialog"):
            yield Label("Enter a direct coding task")
            yield TextArea(
                "",
                placeholder="Describe the task… (multiple lines are supported)",
                soft_wrap=True,
                id="task-input",
            )
            yield Label(
                "Enter adds a line · Ctrl+Enter submits · Escape cancels",
                id="task-help",
            )

    def action_submit_task(self) -> None:
        """Submit the complete, potentially multi-line task in one action."""
        task = self.query_one("#task-input", TextArea).text.strip()
        if task:
            self.dismiss(task)

    def action_cancel_task(self) -> None:
        self.dismiss(None)


class GuardrailCheckScreen(ModalScreen[str | None]):
    """Collect a command for a policy-only guardrail check; never execute it."""

    CSS = """
    GuardrailCheckScreen { align: center middle; }
    #guardrail-dialog {
        width: 88%; max-width: 112; height: 60%; min-height: 14;
        padding: 1 2; border: round $accent; background: $panel;
    }
    #guardrail-input { height: 1fr; min-height: 4; margin-top: 1; }
    #guardrail-help { height: 3; color: $text-muted; margin-top: 1; }
    """
    BINDINGS = [
        Binding("ctrl+enter", "check_command", "Check command", priority=True),
        Binding("escape", "cancel_check", "Cancel", priority=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="guardrail-dialog"):
            yield Label("Sandbox guardrail check — commands are never executed")
            yield TextArea(
                "curl https://example.com",
                placeholder="Enter one command to check against sandbox policy",
                soft_wrap=True,
                id="guardrail-input",
            )
            yield Label(
                "Ctrl+Enter checks policy · no subprocess is launched · Esc cancels",
                id="guardrail-help",
            )

    def action_check_command(self) -> None:
        """Return the entered command to the application for policy inspection."""
        command = self.query_one("#guardrail-input", TextArea).text.strip()
        if command:
            self.dismiss(command)

    def action_cancel_check(self) -> None:
        self.dismiss(None)


class LogsScreen(Screen[None]):
    """Full-screen, selectable log history with copy and export shortcuts."""

    CSS = """
    LogsScreen { layout: vertical; }
    #logs-screen-title { height: 1; padding: 0 1; color: $accent; text-style: bold; }
    #full-agent-log { height: 1fr; border: round $primary; }
    #logs-screen-status { height: 2; padding: 0 1; color: $text-muted; }
    """
    BINDINGS = [
        Binding("escape", "close_logs", "Back to dashboard"),
        Binding("l", "close_logs", "Back to dashboard", priority=True),
        Binding("c", "copy_all", "Copy all logs", priority=True),
        Binding("s", "save_logs", "Save log", priority=True),
    ]

    def __init__(
        self,
        entries: Sequence[str],
        *,
        log_directory: Path,
    ) -> None:
        super().__init__()
        self._entries = list(entries)
        self._log_directory = log_directory

    @property
    def log_text(self) -> str:
        """Return the complete log text, including multiline entries."""
        return "\n".join(self._entries) or "No logs recorded yet."

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("Agent Logs — full screen", id="logs-screen-title")
        yield TextArea(
            self.log_text,
            read_only=True,
            show_cursor=False,
            show_line_numbers=True,
            soft_wrap=True,
            id="full-agent-log",
        )
        yield Static(
            "Esc Back · c Copy all logs · s Save logs · "
            "Select text with the mouse to copy a portion",
            id="logs-screen-status",
        )
        yield Footer()

    def refresh_entries(self, entries: Sequence[str]) -> None:
        """Replace the whole viewer content (used only when history is trimmed)."""
        self._entries = list(entries)
        if self.is_mounted:
            self.query_one("#full-agent-log", TextArea).text = self.log_text

    def append_entry(self, entry: str) -> None:
        """Append one new log entry without re-rendering the whole history.

        Rebuilding the full TextArea on every streamed line is O(n) per line,
        which froze the UI for minutes on long sessions. Appending is O(1).
        """
        was_empty = not self._entries
        self._entries.append(entry)
        if not self.is_mounted:
            return
        text_area = self.query_one("#full-agent-log", TextArea)
        if was_empty:
            text_area.text = entry
        else:
            text_area.insert("\n" + entry, location=text_area.document.end)

    def action_close_logs(self) -> None:
        self.app.pop_screen()

    def action_copy_all(self) -> None:
        try:
            self.app.copy_to_clipboard(self.log_text)
        except Exception as exc:
            status = f"Clipboard copy failed: {exc}. Use s to save the logs instead."
        else:
            status = (
                "Copy request sent to the terminal clipboard. "
                "If paste does not work, use s to save the logs."
            )
        self.query_one("#logs-screen-status", Static).update(status)

    def action_save_logs(self) -> None:
        filename = f"session-{datetime.now():%Y%m%d-%H%M%S-%f}.log"
        destination = self._log_directory / filename
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(self.log_text + "\n", encoding="utf-8")
        except OSError as exc:
            status = f"Could not save logs: {exc}"
        else:
            status = f"Saved {len(self._entries)} log entries to {destination}"
        self.query_one("#logs-screen-status", Static).update(status)


class CodePilotApp(App[None]):
    """Display issue feed, active task, streaming logs, and approval requests."""

    TITLE = "CodePilot"
    SUB_TITLE = "Multi-Agent Coding Platform"
    CSS = """
    Screen { layout: vertical; }
    #dashboard {
        height: 1fr; layout: grid; grid-size: 2 2;
        grid-columns: 1fr 1fr; grid-rows: 1fr 1fr;
    }
    .panel { border: round $primary; padding: 0 1; height: 100%; background: $background; }
    .panel-title { text-style: bold; color: $accent; height: 1; }
    #issues-panel { column-span: 1; }
    #task-panel { column-span: 1; }
    #logs-panel { column-span: 1; }
    #approval-panel { column-span: 1; }
    #issues-content, #task-content, #approval-content { height: 1fr; overflow-y: auto; }
    #agent-log { height: 1fr; border: none; padding: 0; background: $background; }
    #approval-command { dock: bottom; }
    #status-bar { height: 1; padding: 0 1; background: $panel; }
    """
    BINDINGS = [
        Binding("i", "new_task", "New task"),
        Binding("g", "guardrail_check", "Guardrail check"),
        Binding("s", "skip_issue", "Skip issue"),
        Binding("l", "focus_logs", "Full logs"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        task_handler: TaskHandler | None = None,
        guardrail_handler: GuardrailHandler | None = None,
        approval_handler: ApprovalHandler | None = None,
        skip_handler: SkipHandler | None = None,
        startup_handler: LifecycleHandler | None = None,
        shutdown_handler: LifecycleHandler | None = None,
        initial_issues: Sequence[Mapping[str, Any]] = (),
        log_directory: Path | None = None,
    ) -> None:
        super().__init__()
        self._task_handler = task_handler
        self._guardrail_handler = guardrail_handler
        self._approval_handler = approval_handler
        self._skip_handler = skip_handler
        self._startup_handler = startup_handler
        self._shutdown_handler = shutdown_handler
        self._shutdown_called = False
        self._issues: list[dict[str, Any]] = [dict(issue) for issue in initial_issues]
        self._active_task: dict[str, Any] = {"title": "No active task", "status": "IDLE"}
        self._pending_approval: tuple[str, dict[str, Any]] | None = None
        self._approval_future: asyncio.Future[bool] | None = None
        self._log_entries: list[str] = []
        self._log_directory = log_directory or (Path.cwd() / ".codepilot" / "logs")
        self._dashboard_container: Container | None = None
        self._dashboard_issues_widget: Static | None = None
        self._dashboard_task_widget: Static | None = None
        self._dashboard_approval_widget: Static | None = None
        self._dashboard_approval_command: Input | None = None
        self._dashboard_log_widget: RichLog | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="dashboard"):
            with Vertical(id="issues-panel", classes="panel"):
                yield Static("GitHub Issues", classes="panel-title")
                yield Static("No issues received yet.", id="issues-content")
            with Vertical(id="task-panel", classes="panel"):
                yield Static("Active Task", classes="panel-title")
                yield Static("No active task", id="task-content")
            with Vertical(id="logs-panel", classes="panel"):
                yield Static("Agent Logs", classes="panel-title")
                yield RichLog(id="agent-log", wrap=True, markup=False, highlight=False)
            with Vertical(id="approval-panel", classes="panel"):
                yield Static("Human Approval", classes="panel-title")
                yield Static("No approval requested.", id="approval-content")
                yield Input(placeholder="approve / reject / inspect", id="approval-command")
        yield Static(
            "Ready · [i] New task  [g] Guardrail check  [s] Skip issue  [l] Full logs  [q] Quit",
            id="status-bar",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._dashboard_container = self.query_one("#dashboard", Container)
        self._dashboard_issues_widget = self.query_one("#issues-content", Static)
        self._dashboard_task_widget = self.query_one("#task-content", Static)
        self._dashboard_approval_widget = self.query_one("#approval-content", Static)
        self._dashboard_approval_command = self.query_one("#approval-command", Input)
        self._dashboard_log_widget = self.query_one("#agent-log", RichLog)
        self._dashboard_approval_command.disabled = True
        self.update_issues(self._issues)
        self.update_active_task(self._active_task)
        if self._startup_handler is not None:
            try:
                self._startup_handler()
            except Exception as exc:
                self.append_log(f"Could not start runtime: {exc}", source="Runtime")
                logger.exception("Could not start the CodePilot runtime")
        # A Textual grid has to allocate whole terminal rows. If the dashboard
        # has an odd number of rows, one track receives the remainder. Reserve
        # one row when needed so the two panel rows remain exactly equal.
        self.call_after_refresh(self._balance_dashboard_rows)

    def on_resize(self, event: Resize) -> None:
        """Recalculate equal dashboard rows after a terminal resize."""
        dashboard = self._dashboard_container
        if dashboard is not None:
            # Re-enable flexible tracks before measuring the new available height.
            dashboard.styles.grid_rows = "1fr 1fr"
            self.call_after_refresh(self._balance_dashboard_rows)

    def _balance_dashboard_rows(self) -> None:
        """Use two equal integer-height grid rows, leaving any spare row unused."""
        dashboard = self._dashboard_container
        if dashboard is None:
            return

        panel_ids = (
            "#issues-panel",
            "#task-panel",
            "#logs-panel",
            "#approval-panel",
        )
        heights = [self.query_one(panel_id, Vertical).size.height for panel_id in panel_ids]
        if not heights or any(height <= 0 for height in heights):
            return

        # Grid tracks use whole terminal cells. If fractional tracks split unevenly,
        # use their smaller rendered height for both rows instead of accepting a
        # one-cell discrepancy between the top and bottom panels.
        row_height = min(heights)
        dashboard.styles.grid_rows = f"{row_height} {row_height}"

    def update_issues(self, issues: Sequence[Mapping[str, Any]]) -> None:
        """Replace the issue panel with the latest polling snapshot."""
        self._issues = [dict(issue) for issue in issues]
        widget = self._dashboard_issues_widget or self.query_one("#issues-content", Static)
        if not self._issues:
            widget.update("No open issues currently available.")
            return
        lines = []
        for issue in self._issues:
            number = issue.get("number", "?")
            title = str(issue.get("title", "Untitled issue"))
            status = str(issue.get("status", issue.get("state", "open"))).upper()
            lines.append(f"#{number}  {status}  {title}")
        widget.update("\n".join(lines))

    def update_active_task(self, task: Mapping[str, Any]) -> None:
        """Update active-task details, including agent, status, skill, and todos."""
        self._active_task = dict(task)
        lines = [
            str(task.get("title", "No active task")),
            f"Status: {task.get('status', 'IDLE')}",
            f"Agent: {task.get('agent', '—')}",
            f"Skill: {task.get('skill', '—')}",
        ]
        todos = task.get("todos", ())
        if todos:
            lines.append("Todos:")
            for todo in todos:
                if isinstance(todo, Mapping):
                    done = bool(todo.get("done", False))
                    label = todo.get("text", todo.get("title", "Task"))
                else:
                    done = False
                    label = str(todo)
                lines.append(f"  [{'✓' if done else ' '}] {label}")
        widget = self._dashboard_task_widget or self.query_one("#task-content", Static)
        widget.update("\n".join(lines))

    def append_log(self, message: str, *, source: str = "CodePilot") -> None:
        """Append one streamed event to the dashboard and full-screen log history."""
        entry = f"[{source}] {message}"
        self._log_entries.append(entry)
        # Bound in-memory history so a long-running polling session stays predictable.
        if len(self._log_entries) > 10_000:
            del self._log_entries[: len(self._log_entries) - 10_000]
        if self._dashboard_log_widget is not None:
            self._dashboard_log_widget.write(entry)
        if isinstance(self.screen, LogsScreen):
            self.screen.append_entry(entry)

    async def request_approval(self, operation: str, details: Mapping[str, Any]) -> bool:
        """Display an approval request and await approve/reject from the user."""
        if self._approval_future is not None and not self._approval_future.done():
            self.append_log(
                "A second approval request was rejected while one is pending.",
                source="HITL",
            )
            return False
        self._pending_approval = (operation, dict(details))
        self._approval_future = asyncio.get_running_loop().create_future()
        rendered_details = "\n".join(f"{key}: {value}" for key, value in details.items())
        body = (
            f"Approval required: {operation}\n{rendered_details}"
            "\n\nType approve, reject, or inspect."
        )
        approval_widget = self._dashboard_approval_widget or self.query_one(
            "#approval-content", Static
        )
        approval_widget.update(body)
        command = self._dashboard_approval_command or self.query_one("#approval-command", Input)
        command.disabled = False
        command.value = ""
        command.focus()
        self.append_log(f"Waiting for human approval: {operation}", source="HITL")
        try:
            return await self._approval_future
        finally:
            command.disabled = True
            self._pending_approval = None
            self._approval_future = None
            # Disabling the focused input leaves nothing focused; give focus
            # back to the dashboard so i / g / s / l / q keep responding.
            if self._dashboard_log_widget is not None:
                try:
                    self._dashboard_log_widget.focus()
                except Exception:
                    pass
            approval_widget = self._dashboard_approval_widget or self.query_one(
                "#approval-content", Static
            )
            approval_widget.update("No approval requested.")

    def request_approval_from_thread(
        self,
        operation: str,
        details: Mapping[str, Any],
    ) -> bool:
        """Synchronously bridge a runtime-thread approval request into the TUI loop.

        The coding and polling workers are synchronous. They must wait for an
        explicit UI decision without attempting to manipulate Textual widgets
        from their own threads. A closed UI or a cancelled worker fails closed.
        """
        decision: Future[bool] = Future()

        def launch_approval() -> None:
            async def wait_for_decision() -> None:
                try:
                    approved = await self.request_approval(operation, details)
                except asyncio.CancelledError:
                    if not decision.done():
                        decision.set_result(False)
                    raise
                except Exception as exc:
                    if not decision.done():
                        decision.set_exception(exc)
                else:
                    if not decision.done():
                        decision.set_result(approved)

            self.run_worker(
                wait_for_decision(),
                name="approval-bridge",
                group="approval-bridge",
                exclusive=False,
            )

        try:
            self.call_from_thread(launch_approval)
            return decision.result()
        except Exception as exc:
            logger.warning("Human approval could not be obtained from the TUI: %s", exc)
            return False

    @on(Input.Submitted, "#approval-command")
    async def handle_approval_command(self, event: Input.Submitted) -> None:
        command = event.value.strip().lower()
        event.input.value = ""
        if self._pending_approval is None or self._approval_future is None:
            self.append_log("There is no pending approval request.", source="HITL")
            return
        if command == "inspect":
            operation, details = self._pending_approval
            self.append_log(f"Inspection — {operation}: {details!r}", source="HITL")
            return
        if command not in {"approve", "reject"}:
            self.append_log("Enter approve, reject, or inspect.", source="HITL")
            return
        approved = command == "approve"
        if self._approval_handler is not None:
            try:
                response = self._approval_handler(
                    self._pending_approval[0], self._pending_approval[1]
                )
                if inspect.isawaitable(response):
                    response = await response
                approved = approved and response is not False
            except Exception as exc:  # fail closed on approval callback errors
                approved = False
                self.append_log(f"Approval handler failed: {exc}", source="HITL")
        if not self._approval_future.done():
            self._approval_future.set_result(approved)
        self.append_log("Approval granted." if approved else "Approval rejected.", source="HITL")

    def action_new_task(self) -> None:
        self.push_screen(NewTaskScreen(), self._receive_new_task)

    def _receive_new_task(self, task: str | None) -> None:
        if not task:
            return
        self.update_active_task({"title": task, "status": "QUEUED", "agent": "Orchestrator"})
        self.append_log(f"Direct task submitted: {task}", source="TUI")
        if self._task_handler is None:
            self.append_log("No direct-task handler is connected to the runtime yet.", source="TUI")
            return
        self.run_worker(
            self._run_task_handler(task),
            name="direct-task",
            group="direct-task",
            exclusive=False,
        )

    async def _run_task_handler(self, task: str) -> None:
        try:
            result = self._task_handler(task) if self._task_handler else None
            if inspect.isawaitable(result):
                result = await result
            self.update_active_task({"title": task, "status": "COMPLETED", "agent": "Orchestrator"})
            if result is not None:
                self.append_log(str(result), source="Task result")
        except Exception as exc:
            self.update_active_task({"title": task, "status": "FAILED", "agent": "Orchestrator"})
            self.append_log(f"Direct task failed: {exc}", source="TUI")

    def action_guardrail_check(self) -> None:
        """Open the safe, non-executing sandbox-policy check."""
        self.push_screen(GuardrailCheckScreen(), self._receive_guardrail_command)

    def _receive_guardrail_command(self, command: str | None) -> None:
        if not command:
            return
        self.append_log(
            f"Checking command against sandbox policy (check-only): {command}",
            source="Guardrail",
        )
        if self._guardrail_handler is None:
            self.append_log(
                "No guardrail-check handler is connected to the runtime.",
                source="Guardrail",
            )
            return
        self.run_worker(
            self._run_guardrail_handler(command),
            name="guardrail-check",
            group="guardrail-check",
            exclusive=False,
        )

    async def _run_guardrail_handler(self, command: str) -> None:
        try:
            result = self._guardrail_handler(command) if self._guardrail_handler else None
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                self.append_log(str(result), source="Guardrail")
        except Exception as exc:
            self.append_log(f"Guardrail check failed safely: {exc}", source="Guardrail")

    def action_skip_issue(self) -> None:
        if self._skip_handler is None:
            self.append_log(
                "No issue-skip handler is connected to the polling runtime.", source="TUI"
            )
            return
        self.run_worker(
            self._run_skip_handler(),
            name="skip-issue",
            group="skip-issue",
            exclusive=True,
        )

    async def _run_skip_handler(self) -> None:
        try:
            result = self._skip_handler() if self._skip_handler else None
            if inspect.isawaitable(result):
                await result
            self.append_log("Skip request sent to issue polling runtime.", source="TUI")
        except Exception as exc:
            self.append_log(f"Could not skip issue: {exc}", source="TUI")

    def action_focus_logs(self) -> None:
        """Open the log history in a dedicated full-screen, copyable view."""
        self.push_screen(LogsScreen(self._log_entries, log_directory=self._log_directory))

    def action_quit(self) -> None:
        """Stop runtime workers before exiting the Textual application."""
        self._stop_runtime_once()
        self.exit()

    def on_unmount(self) -> None:
        """Also stop the runtime if the application closes by another route."""
        self._stop_runtime_once()

    def _stop_runtime_once(self) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True
        if self._shutdown_handler is None:
            return
        try:
            self._shutdown_handler()
        except Exception:
            logger.exception("Could not stop the CodePilot runtime cleanly")


def main() -> None:
    """Launch the standalone dashboard shell."""
    CodePilotApp().run()


if __name__ == "__main__":
    main()
