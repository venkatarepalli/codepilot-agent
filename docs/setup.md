# Setup Guide

The README is the canonical, executable first-time and subsequent-run setup guide. Follow its **Prerequisites**, **First-Time Setup**, **Subsequent Runs**, and **Testing** sections. Avoid maintaining a second divergent command sequence here.

## State and recovery

- `.env` contains local credentials and must not be committed.
- `.codepilot/` contains local workspace and memory state. Inspect and back up retained work before deleting it.
- The runner's issue checkout is configured outside the current project checkout to avoid recursively copying project state into the task workspace.
- `python -m codepilot.runner` starts the four-panel Textual interface; use `python -m codepilot.runner --headless` for console mode.
- The TUI's `i` shortcut submits a direct task to the sandboxed Coder Agent workflow. It retains a proposed diff and does not automatically open a PR without a linked GitHub issue.
- The TUI's `s` shortcut skips the oldest queued issue; it does not force-cancel an active task. The `q` shortcut requests shutdown.
- If a task stops because approval was denied or unavailable, inspect its test output and retained workspace before deciding whether to rerun.

## Fresh-directory verification

Before submission, follow the README from a clean clone in a different directory, using a fresh virtual environment. Verify dependency installation, test/lint commands, and startup configuration without committing `.env` or local state. A live GitHub issue-to-PR check should use a disposable test repository and explicit approval.
