"""Declarative security policy for sandbox operations."""

import shlex
from collections.abc import Sequence
from pathlib import Path

from codepilot.sandbox.models import (
    SecurityCheckResult,
    SecurityDecision,
)


class SandboxPolicy:
    """Validate proposed commands and filesystem operations."""

    BLOCKED_COMMANDS = frozenset({"curl", "wget"})

    SENSITIVE_FILENAMES = frozenset({".env"})

    SENSITIVE_SUFFIXES = frozenset(
        {
            ".secret",
            ".pem",
            ".key",
        }
    )

    APPROVAL_COMMANDS = frozenset({"git push"})

    def __init__(self, sandbox_root: str | Path) -> None:
        """Initialize the policy with the permitted filesystem root."""
        self.sandbox_root = Path(sandbox_root).resolve()

        if not self.sandbox_root.is_dir():
            raise ValueError("The sandbox root must be an existing directory.")

    def check_command(
        self,
        command: str | Sequence[str],
        *,
        working_directory: str | Path,
    ) -> SecurityCheckResult:
        """Validate a command without executing it."""
        try:
            arguments = shlex.split(command) if isinstance(command, str) else list(command)
        except ValueError as exc:
            return SecurityCheckResult(
                SecurityDecision.BLOCK,
                f"Invalid command syntax: {exc}",
            )

        if not arguments or any(not arg.strip() for arg in arguments):
            return SecurityCheckResult(
                SecurityDecision.BLOCK,
                "The command must contain non-empty arguments.",
            )

        executable = Path(arguments[0]).name.lower()

        if executable in self.BLOCKED_COMMANDS:
            return SecurityCheckResult(
                SecurityDecision.BLOCK,
                f"The command '{executable}' is prohibited.",
            )

        # Block package installation, including invocation through Python.
        is_pip_install = (
            executable in {"pip", "pip3"}
            and len(arguments) > 1
            and arguments[1].lower() == "install"
        )

        is_python_module_install = (
            executable in {"python", "python3", "python.exe", "python3.exe"}
            and len(arguments) > 3
            and arguments[1].lower() == "-m"
            and arguments[2].lower() in {"pip", "pip3"}
            and arguments[3].lower() == "install"
        )

        if is_pip_install or is_python_module_install:
            return SecurityCheckResult(
                SecurityDecision.BLOCK,
                "Installing packages in the sandbox is prohibited.",
            )

        if executable in {"rm", "rm.exe"}:
            if self._is_recursive_force_removal(arguments[1:]):
                return SecurityCheckResult(
                    SecurityDecision.BLOCK,
                    "Recursive forced file removal is prohibited.",
                )

        if self._contains_shell_control_operators(arguments):
            return SecurityCheckResult(
                SecurityDecision.BLOCK,
                "Shell operators are prohibited. Submit one argument-list command at a time.",
            )

        directory_result = self.check_path(
            working_directory,
            operation="execute",
        )

        if directory_result.decision != SecurityDecision.ALLOW:
            return directory_result

        if self.is_git_push_command(arguments):
            return SecurityCheckResult(
                SecurityDecision.REQUIRES_APPROVAL,
                "Pushing changes to a Git remote requires human approval.",
            )

        return SecurityCheckResult(
            SecurityDecision.ALLOW,
            "The command passed the sandbox policy checks.",
        )

    @staticmethod
    def is_git_push_command(arguments: Sequence[str]) -> bool:
        """Recognize Git push subcommands, including commands with global options."""
        if not arguments or Path(arguments[0].replace("\\", "/")).name.lower() not in {
            "git",
            "git.exe",
        }:
            return False

        options_with_value = {
            "-C",
            "-c",
            "--git-dir",
            "--work-tree",
            "--namespace",
            "--config-env",
        }
        index = 1
        while index < len(arguments):
            argument = arguments[index]
            if argument == "--":
                return False
            if argument in options_with_value:
                index += 2
                continue
            if argument.startswith(("-C", "-c")) and len(argument) > 2:
                index += 1
                continue
            if argument.startswith(("--git-dir=", "--work-tree=", "--namespace=", "--config-env=")):
                index += 1
                continue
            if argument.startswith("-"):
                index += 1
                continue
            return argument.lower() == "push"
        return False

    def check_path(
        self,
        path: str | Path,
        *,
        operation: str,
    ) -> SecurityCheckResult:
        """Validate a filesystem path and the requested operation."""
        try:
            candidate = Path(path)

            if not candidate.is_absolute():
                candidate = self.sandbox_root / candidate

            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            return SecurityCheckResult(
                SecurityDecision.BLOCK,
                f"Unable to resolve the requested path: {exc}",
            )

        if not self._is_within_sandbox(resolved):
            return SecurityCheckResult(
                SecurityDecision.BLOCK,
                "The requested path is outside the sandbox.",
            )

        if self._is_sensitive_path(resolved):
            return SecurityCheckResult(
                SecurityDecision.REQUIRES_APPROVAL,
                "The requested path matches a sensitive-file rule.",
            )

        if operation not in {"read", "write", "execute", "delete"}:
            return SecurityCheckResult(
                SecurityDecision.BLOCK,
                f"Unsupported filesystem operation: {operation}",
            )

        return SecurityCheckResult(
            SecurityDecision.ALLOW,
            "The path passed the sandbox policy checks.",
        )

    def _is_within_sandbox(self, path: Path) -> bool:
        """Return whether a resolved path is within the sandbox root."""
        try:
            path.relative_to(self.sandbox_root)
        except ValueError:
            return False

        return True

    @classmethod
    def _is_sensitive_path(cls, path: Path) -> bool:
        """Identify sensitive filenames anywhere in a path."""
        for component in path.parts:
            filename = component.lower()

            # Protect environment files, but allow documented examples.
            if filename == ".env" or (
                filename.startswith(".env.") and filename not in {".env.example", ".env.sample"}
            ):
                return True

            if filename in cls.SENSITIVE_FILENAMES:
                return True

            if any(filename.endswith(suffix) for suffix in cls.SENSITIVE_SUFFIXES):
                return True

            if "credentials" in filename:
                return True

        return False

    @staticmethod
    def _is_recursive_force_removal(arguments: Sequence[str]) -> bool:
        """Identify rm arguments combining recursive and force flags."""
        has_recursive = False
        has_force = False

        for argument in arguments:
            if argument == "--":
                break

            if argument.startswith("--"):
                if argument == "--recursive":
                    has_recursive = True
                elif argument == "--force":
                    has_force = True
                continue

            if argument.startswith("-") and argument != "-":
                flags = argument[1:]

                has_recursive |= "r" in flags or "R" in flags
                has_force |= "f" in flags

        return has_recursive and has_force

    @staticmethod
    def _contains_shell_control_operators(
        arguments: Sequence[str],
    ) -> bool:
        """Reject shell operators that could combine commands."""
        operators = {
            ";",
            "&&",
            "||",
            "|",
            ">",
            ">>",
            "<",
            "<<",
            "$(",
            "`",
        }

        return any(
            argument in operators or any(operator in argument for operator in ("$(", "`"))
            for argument in arguments
        )
