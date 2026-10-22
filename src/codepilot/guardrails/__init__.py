"""NeMo Guardrails input rail for CodePilot.

Two guardrail layers exist in CodePilot:

1. **NeMo Guardrails input rail (this module).** Screens untrusted task text
   (GitHub issue title/body, direct TUI tasks) *before* it reaches the
   Orchestrator LLM. Blocks prompt injection, secret exfiltration, destructive
   shell requests, edits to protected files, and attempts to bypass HITL.
   The rail is defined in ``config/config.yml`` + ``config/rails.co`` and runs a
   deterministic Python action, so no LLM call is made.

2. **Sandbox execution policy** (``codepilot.sandbox.policy``). Blocks
   ``rm -rf``, ``curl``, ``wget``, ``pip install``, paths outside the sandbox,
   and sensitive files at ``execute``/``edit_file`` time; routes ``git push``
   to the HITL approval panel.

If ``nemoguardrails`` cannot be imported or raises, the same rules are applied
directly in Python (``engine="python-fallback"``), so the rail fails *closed*.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_DIRECTORY = Path(__file__).resolve().parent / "config"
BLOCKED_MARKER = "CODEPILOT_GUARDRAIL_BLOCKED"
NEMO_ACTION_NAME = "codepilot_input_policy_check"


@dataclass(frozen=True, slots=True)
class GuardrailRule:
    name: str
    pattern: re.Pattern[str]
    reason: str


def _rule(name: str, pattern: str, reason: str) -> GuardrailRule:
    return GuardrailRule(name, re.compile(pattern, re.IGNORECASE | re.DOTALL), reason)


# Text-level rules. Deliberately do NOT block a bare "pip install" mention:
# dependency_update issues legitimately talk about it; the sandbox policy
# blocks the actual command at execution time.
INPUT_RULES: tuple[GuardrailRule, ...] = (
    _rule(
        "prompt_injection",
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|all|your|system)\b"
        r".{0,30}\b(instructions?|rules|prompt|guardrails?|policy)\b"
        r"|\breveal\b.{0,20}\bsystem prompt\b"
        r"|\byou are now\b.{0,40}\b(unrestricted|jailbroken|dan|developer mode)\b"
        r"|\bjailbreak\b"
        # Paraphrased overrides: "disregard the guidance you were given", ...
        r"|\b(ignore|disregard|forget|override)\b.{0,40}"
        r"\b(guidance|instructions?|directions|rules|constraints)\b.{0,25}"
        r"\b(you (were|have been) (given|told)|given to you|above|earlier|before this)\b"
        r"|\b(disregard|forget)\b.{0,30}\bwhat you (were|have been) (told|given)\b"
        # Persona switches: "from now on you are an assistant with no restrictions"
        r"|\b(you are|you're|act as|pretend (to be|you are)|behave (as|like)|from now on)\b"
        r".{0,50}\b(no|without|free (of|from)) (restrictions|rules|limits|limitations"
        r"|filters|guardrails|safety)\b"
        r"|\b(unrestricted|uncensored) (assistant|mode|ai|model)\b",
        "Task text tries to override CodePilot's instructions (prompt injection).",
    ),
    _rule(
        "secret_exfiltration",
        r"\b(print|show|reveal|output|send|post|upload|exfiltrate|leak|echo|cat|dump|email)\b"
        r".{0,40}(GITHUB_TOKEN|OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY|api[_ -]?key"
        r"|access token|secrets?\b|passwords?|private key|(?<![\w.])\.env(?![\w.]))"
        # Indirect exfiltration into an output channel: "add the value of the
        # token from the environment to the PR body". Reading a key from the
        # environment for configuration (no output target) stays allowed.
        r"|(\b(value|contents?) of\b.{0,30}\b(token|api[_ -]?key|secret|password|credential)s?\b"
        r"|\b(token|api[_ -]?key|secret|password|credential)s?\b.{0,30}"
        r"\bfrom (the )?(environment|env\b|os\.environ|config(uration)?)\b)"
        r".{0,60}\b(to|into|in) (the )?(pr|pull request|commit( message)?|comment|issue"
        r"|body|description|logs?|output|readme)\b",
        "Task asks CodePilot to reveal or transmit secrets or credentials.",
    ),
    _rule(
        "destructive_shell",
        r"\brm\s+-[a-z]*(rf|fr)[a-z]*\b"
        # Separate or long flags: "rm -r -f", "rm --recursive --force"
        r"|\brm\b(?=[^\n]{0,40}\s(-[a-z]*r|--recursive))(?=[^\n]{0,40}\s(-[a-z]*f|--force))"
        r"|\b(delete|wipe|erase|destroy)\b.{0,15}\b(everything|the (entire|whole) "
        r"(repo|repository|project|codebase|disk|drive|file ?system))\b"
        r"|\bgit\s+clean\s+-[a-z]*f|\bgit\s+reset\s+--hard\b"
        r"|\b(curl|wget)\b[^\n]{0,200}\|\s*(sudo\s+)?(ba|z)?sh\b"
        r"|\bmkfs(\.\w+)?\b|\bdd\s+if=|:\(\)\s*\{\s*:\|:&\s*\};:"
        r"|\bchmod\s+-R\s+777\s+/",
        "Task requests a destructive or remote-code shell command.",
    ),
    _rule(
        "protected_file_edit",
        r"\b(edit|modify|change|update|overwrite|replace|delete|remove|write|commit|add|create)\b"
        r".{0,40}((?<![\w.])\.env(?![\w.])|\S*\.(pem|key|secret)\b"
        r"|\S*credentials\S*\.(json|ya?ml|txt|ini|cfg|toml)\b|\bid_rsa\b)",
        "Task targets a protected secret file (.env, *.pem, *.key, *.secret, credentials).",
    ),
    _rule(
        "approval_bypass",
        r"\b(disable|bypass|skip|turn off|circumvent|remove)\b.{0,25}"
        r"\b(guardrails?|human approval|approvals?|hitl|human[- ]in[- ]the[- ]loop|review gate)\b"
        r"|\bgit\s+push\s+(-f\b|--force)"
        r"|\b(push|merge|commit)\b.{0,15}\bdirectly (to|into) (main|master)\b",
        "Task tries to bypass human approval, force-push, or write directly to main.",
    ),
)


@dataclass(frozen=True, slots=True)
class GuardrailVerdict:
    """Outcome of screening one piece of task text."""

    allowed: bool
    rule: str | None = None
    reason: str | None = None
    engine: str = "nemo"

    @property
    def blocked(self) -> bool:
        return not self.allowed

    def describe(self) -> str:
        if self.allowed:
            return f"Guardrails ({self.engine}): input allowed."
        return f"Guardrails ({self.engine}) BLOCKED [{self.rule}]: {self.reason}"


def evaluate_input_text(text: str) -> dict[str, Any]:
    """Apply the input rules. Pure function; also the body of the NeMo action."""
    for rule in INPUT_RULES:
        if rule.pattern.search(text or ""):
            return {"blocked": True, "rule": rule.name, "reason": rule.reason}
    return {"blocked": False, "rule": None, "reason": None}


async def _nemo_policy_action(text: str = "") -> dict[str, Any]:
    return evaluate_input_text(text)


class CodePilotGuardrails:
    """NeMo Guardrails-backed input screening with a fail-closed fallback."""

    def __init__(self, config_directory: Path = CONFIG_DIRECTORY) -> None:
        self._config_directory = Path(config_directory)
        self._rails: Any | None = None
        self._init_error: Exception | None = None
        self._lock = Lock()

    def _get_rails(self) -> Any | None:
        if self._rails is not None or self._init_error is not None:
            return self._rails
        try:
            from nemoguardrails import LLMRails, RailsConfig

            config = RailsConfig.from_path(str(self._config_directory))
            rails = LLMRails(config)
            rails.register_action(_nemo_policy_action, name=NEMO_ACTION_NAME)
            self._rails = rails
            logger.info("NeMo Guardrails input rail loaded from %s", self._config_directory)
        except Exception as exc:  # ImportError, config errors, etc.
            self._init_error = exc
            logger.warning(
                "NeMo Guardrails unavailable (%s: %s); using python-fallback rules.",
                type(exc).__name__,
                exc,
            )
        return self._rails

    @property
    def engine(self) -> str:
        return "nemo" if self._get_rails() is not None else "python-fallback"

    def check_input(self, text: str) -> GuardrailVerdict:
        """Screen untrusted task text. Must not be called from a running event loop."""
        with self._lock:
            rails = self._get_rails()
            if rails is not None:
                try:
                    result = rails.generate(
                        messages=[{"role": "user", "content": text or ""}],
                        options={"rails": ["input"], "output_vars": ["verdict"]},
                    )
                    verdict = (result.output_data or {}).get("verdict") or {}
                    content = (result.response or [{}])[0].get("content", "")
                    if verdict.get("blocked") or content == BLOCKED_MARKER:
                        return GuardrailVerdict(
                            allowed=False,
                            rule=verdict.get("rule") or "nemo_input_rail",
                            reason=verdict.get("reason") or "Blocked by NeMo input rail.",
                            engine="nemo",
                        )
                    return GuardrailVerdict(allowed=True, engine="nemo")
                except Exception as exc:
                    logger.warning(
                        "NeMo input rail raised %s; applying python-fallback rules.",
                        type(exc).__name__,
                    )

        fallback = evaluate_input_text(text)
        return GuardrailVerdict(
            allowed=not fallback["blocked"],
            rule=fallback["rule"],
            reason=fallback["reason"],
            engine="python-fallback",
        )


_default_guardrails: CodePilotGuardrails | None = None
_default_lock = Lock()


def get_guardrails() -> CodePilotGuardrails:
    """Return the process-wide guardrails instance."""
    global _default_guardrails
    with _default_lock:
        if _default_guardrails is None:
            _default_guardrails = CodePilotGuardrails()
        return _default_guardrails


__all__ = [
    "BLOCKED_MARKER",
    "CONFIG_DIRECTORY",
    "INPUT_RULES",
    "CodePilotGuardrails",
    "GuardrailVerdict",
    "evaluate_input_text",
    "get_guardrails",
]
