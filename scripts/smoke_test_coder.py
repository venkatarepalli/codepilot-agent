"""Smoke test the Gemini-to-Coder integration in an isolated sample repository.

This script:
1. Creates a disposable sample repository.
2. Introduces a deliberately incorrect implementation.
3. Invokes the CodePilot Coder workflow.
4. Reports the coding, testing, and proposed-diff results.
5. Verifies that the original sample repository was not modified.

Run from the project root:
    python scripts/smoke_test_coder.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from codepilot.config.bootstrap import create_orchestrator


def create_sample_repository(repository_path: Path) -> None:
    """Create a minimal repository containing a deliberately failing test."""
    source_directory = repository_path / "src"
    test_directory = repository_path / "tests"

    source_directory.mkdir(parents=True)
    test_directory.mkdir(parents=True)

    (source_directory / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )

    (source_directory / "calculator.py").write_text(
        "def calculate_total(price: float, quantity: int) -> float:\n"
        '    """Calculate the total price for the requested quantity."""\n'
        "    return price + quantity\n",
        encoding="utf-8",
    )

    (test_directory / "test_calculator.py").write_text(
        "import pytest\n"
        "from src.calculator import calculate_total\n"
        "\n"
        "\n"
        "def test_calculate_total_multiplies_price_by_quantity():\n"
        "    assert calculate_total(10.0, 3) == 30.0\n"
        "\n"
        "\n"
        "def test_calculate_total_handles_single_item():\n"
        "    assert calculate_total(10.0, 1) == 10.0\n",
        encoding="utf-8",
    )


def get_status(value: Any) -> str:
    """Normalize a string or enum status to lowercase text."""
    if value is None:
        return "unknown"

    return str(getattr(value, "value", value)).lower()


def report_result(result: Any) -> None:
    """Print the relevant information returned by the Coder Agent."""
    print("\n" + "=" * 65)
    print("CODEPILOT CODER SMOKE TEST RESULTS")
    print("=" * 65)

    status = get_status(getattr(result, "status", None))
    print(f"Coding status : {status}")
    print(f"Attempts      : {getattr(result, 'attempts', 'N/A')}")

    plan = getattr(result, "plan", None)
    if plan:
        print("\nPlan:")
        print(plan)

    test_result = getattr(result, "test_result", None)

    if test_result is not None:
        test_status = get_status(getattr(test_result, "status", None))
        print(f"\nTest status   : {test_status}")

        test_output = getattr(test_result, "output", None)
        if test_output:
            print("\nTest output:")
            print(test_output)

    diff_path = getattr(result, "diff_path", None)

    if diff_path:
        diff_file = Path(diff_path)
        print(f"\nProposed diff : {diff_file}")

        if diff_file.is_file():
            print("\nProposed diff contents:")
            print("-" * 65)
            print(diff_file.read_text(encoding="utf-8"))
            print("-" * 65)
        else:
            print("The reported diff file does not exist.")

    error = getattr(result, "error", None)
    if error:
        print(f"\nError         : {error}")

    print("=" * 65)


def run_smoke_test() -> bool:
    """Run a coding task against a temporary sample repository."""
    with tempfile.TemporaryDirectory(prefix="codepilot_coder_smoke_") as temporary_directory:
        repository_path = Path(temporary_directory) / "sample-repository"
        repository_path.mkdir()

        create_sample_repository(repository_path)

        calculator_file = repository_path / "src" / "calculator.py"
        original_source = calculator_file.read_text(encoding="utf-8")

        issue = SimpleNamespace(
            number=99999,
            title="Fix calculate_total to multiply price by quantity",
            body=(
                "The calculate_total function currently adds price and "
                "quantity instead of multiplying them. Correct the "
                "implementation. Run the existing tests and provide a "
                "proposed diff. Do not change the tests."
            ),
            labels=["bug"],
            state="open",
            assignee=None,
        )

        print("CodePilot Coder Integration Smoke Test")
        print("-" * 65)
        print(f"Temporary repository: {repository_path}")
        print(f"Task: {issue.title}")
        print("\nCreating the CodePilot Orchestrator...")

        smoke_workspace_root = Path(temporary_directory) / "codepilot-workspaces"

        orchestrator = create_orchestrator(
            workspace_root=smoke_workspace_root,
        )

        print("Invoking the Coder workflow...")
        result = orchestrator.run_coding_task(
            issue,
            repository_path=str(repository_path),
        )

        report_result(result)

        # The original sample repository must remain untouched.
        current_source = calculator_file.read_text(encoding="utf-8")

        if current_source != original_source:
            print("\nFAIL: The original sample repository was modified.")
            return False

        print("\nPASS: The original sample repository remains unchanged.")

        coding_status = get_status(getattr(result, "status", None))

        if coding_status not in {"succeeded", "success"}:
            print("\nFAIL: The Coder workflow did not report success.")
            return False

        test_result = getattr(result, "test_result", None)

        if test_result is None:
            print("\nFAIL: No test result was returned.")
            return False

        test_status = get_status(getattr(test_result, "status", None))

        if test_status != "succeeded":
            print(f"\nFAIL: The final test status was {test_status!r}.")
            return False

        diff_path = getattr(result, "diff_path", None)

        if not diff_path or not Path(diff_path).is_file():
            print("\nFAIL: The expected proposed diff was not generated.")
            return False

        print("\nSUCCESS: All smoke-test acceptance checks passed.")
        return True


def main() -> int:
    """Run the smoke test and return a process exit code."""
    try:
        return 0 if run_smoke_test() else 1
    except Exception as exc:
        print(f"\nSMOKE TEST ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
