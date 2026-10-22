"""Component 2: the Coder's bounded context starts with Repo Map-relevant files."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

from codepilot.agents.deepagents_coder_adapter import DeepAgentsCoderAdapter
from codepilot.sandbox.workspace import WorkspaceInfo


def _workspace(tmp_path: Path, count: int = 20) -> WorkspaceInfo:
    root = tmp_path / "ws"
    for index in range(count):
        path = root / "aaa" / f"filler_{index:02d}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"X = {index}\n", encoding="utf-8")
    target = root / "zzz" / "billing" / "calculator.py"
    target.parent.mkdir(parents=True)
    target.write_text("def calculate_total(p, q):\n    return p + q\n", encoding="utf-8")
    return WorkspaceInfo(issue_number=1, path=root, source_repository=tmp_path)


def test_without_priority_alphabetical_cutoff_drops_the_relevant_file(tmp_path) -> None:
    adapter = DeepAgentsCoderAdapter(MagicMock(), MagicMock())
    files = adapter._collect_workspace_files(_workspace(tmp_path))
    assert len(files) == DeepAgentsCoderAdapter.MAX_CONTEXT_FILES
    assert "zzz/billing/calculator.py" not in files  # the bug this fix addresses


def test_relevant_files_are_loaded_first_even_if_they_sort_last(tmp_path, caplog) -> None:
    adapter = DeepAgentsCoderAdapter(
        MagicMock(),
        MagicMock(),
        priority_paths=["zzz\\billing\\calculator.py", "missing/file.py"],
    )
    with caplog.at_level(logging.INFO, logger="codepilot"):
        files = adapter._collect_workspace_files(_workspace(tmp_path))

    assert next(iter(files)) == "zzz/billing/calculator.py"
    assert len(files) == DeepAgentsCoderAdapter.MAX_CONTEXT_FILES
    assert "1 relevant file(s) from Repo Map retrieval loaded first" in caplog.text


def test_selection_is_logged_once_per_task(tmp_path, caplog) -> None:
    adapter = DeepAgentsCoderAdapter(
        MagicMock(), MagicMock(), priority_paths=["zzz/billing/calculator.py"]
    )
    workspace = _workspace(tmp_path)
    with caplog.at_level(logging.INFO, logger="codepilot"):
        adapter._collect_workspace_files(workspace)
        adapter._collect_workspace_files(workspace)
    assert caplog.text.count("Coder context:") == 1
