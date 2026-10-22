# Setup Guide

The README's [Setup](../README.md#setup) section is the canonical, step-by-step setup guide: prerequisites, virtual environment, install, offline test run, test-repository preparation (labels and token permissions), `.env`, and start. This page only adds operational notes.

## State and recovery

- `.env` contains local credentials and must not be committed (it is git-ignored).
- `.codepilot/` holds local state: the Repo Map cache, ChromaDB stores and episodic session summaries (`memory/sessions.jsonl`).
- Task workspaces are created outside the checkout (`../.<checkout>-codepilot-workspaces/`), to avoid recursively copying project state.
- `python -m codepilot.runner` starts the four-panel Textual interface; `--headless` runs the console loop.
- `i` submits a direct task (sandboxed, tested, no PR). `g` checks a command against both guardrail layers without running it. `s` skips the oldest queued issue. `l` opens the full log. `q` quits and writes the session summary.
- If a task stops because an approval was rejected, inspect its retained workspace and `working/proposed_diff.txt` before re-running.

## Fresh-clone verification

From a clean clone in a new directory with a fresh virtual environment: `pip install -e .`, then `python -m pytest -q` (all tests pass offline, with no keys required), then configure `.env` and run against a disposable test repository.
