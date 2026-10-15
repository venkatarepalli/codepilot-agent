"""Unit tests for the Coder integration smoke-test helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "smoke_test_coder.py"

SPEC = importlib.util.spec_from_file_location(
    "smoke_test_coder",
    SCRIPT_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load smoke-test script: {SCRIPT_PATH}")

SMOKE_TEST_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE_TEST_MODULE)

create_sample_repository = SMOKE_TEST_MODULE.create_sample_repository
get_status = SMOKE_TEST_MODULE.get_status


def test_create_sample_repository_creates_expected_files(
    tmp_path: Path,
) -> None:
    """The sample repository should contain source code and tests."""
    create_sample_repository(tmp_path)

    source_file = tmp_path / "src" / "calculator.py"
    test_file = tmp_path / "tests" / "test_calculator.py"

    assert source_file.is_file()
    assert test_file.is_file()

    source = source_file.read_text(encoding="utf-8")
    tests = test_file.read_text(encoding="utf-8")

    assert "def calculate_total" in source
    assert "return price + quantity" in source
    assert "test_calculate_total_multiplies_price_by_quantity" in tests


def test_sample_repository_starts_with_deliberately_wrong_logic(
    tmp_path: Path,
) -> None:
    """The initial implementation should expose the coding task."""
    create_sample_repository(tmp_path)

    source_file = tmp_path / "src" / "calculator.py"
    source = source_file.read_text(encoding="utf-8")

    assert "return price + quantity" in source
    assert "return price * quantity" not in source


def test_get_status_normalizes_string() -> None:
    assert get_status("PASSED") == "passed"


def test_get_status_normalizes_enum() -> None:
    from enum import Enum

    class Status(Enum):
        PASSED = "passed"

    assert get_status(Status.PASSED) == "passed"


def test_get_status_handles_none() -> None:
    assert get_status(None) == "unknown"
