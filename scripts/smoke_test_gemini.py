"""Make one real Gemini request through CodePilot's DeepAgents Orchestrator.

Run from the project root with the project's virtual environment activated:
    python scripts/smoke_test_gemini.py

This makes a real API request and may incur provider usage or cost.
It does not access GitHub, create files, or execute generated code.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codepilot.config.bootstrap import create_orchestrator  # noqa: E402


def main() -> int:
    """Send a minimal prompt and report whether a model response was received."""
    try:
        orchestrator = create_orchestrator()
        result = orchestrator.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Reply with exactly: CODEPILOT_GEMINI_SMOKE_OK. "
                            "Do not call tools or perform any other action."
                        ),
                    }
                ]
            }
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Gemini smoke test failed: {type(exc).__name__}: {exc}")
        return 1

    messages = result.get("messages", []) if isinstance(result, dict) else []
    content = messages[-1].content if messages else ""
    if isinstance(content, list):
        content = " ".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    print("Gemini smoke test request completed.")
    print(f"Final response: {content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
