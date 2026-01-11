"""CommandReference - Shell command output references."""

import os
import subprocess
from typing import List

from ..render import process_text
from ..utils import count_tokens
from .helpers import remove_ansi


class CommandReference:
    """A reference that captures shell command output."""

    def __init__(
        self,
        command: str,
        format: str = "shell",
        capture_stderr: bool = True,
        shell_executable: str | None = None,
    ):
        """
        Initialize a command reference.

        Args:
            command: The raw command string, e.g. "ls --help"
            format: Output format - "md", "xml", or "shell"
            capture_stderr: Whether to capture stderr as well
            shell_executable: Override the shell executable
        """
        self.command = command
        self.format = format
        self.capture_stderr = capture_stderr
        self.shell_executable = shell_executable

        self.command_output = self._run_command()
        self.output = self._get_contents()

    @property
    def label(self) -> str:
        """Return the command as the label."""
        return self.command

    def read(self) -> str:
        """Read and return the command output."""
        return self.command_output

    def exists(self) -> bool:
        """Check if the command can be executed (always True for commands)."""
        return True

    def token_count(self, encoding: str = "cl100k_base") -> int:
        """Count tokens in the command output."""
        return count_tokens(self.command_output, target=encoding)["count"]

    def _run_command(self) -> str:
        """Execute the command and return combined stdout/stderr."""
        try:
            run_kwargs: dict[str, str | bool] = {
                "shell": True,
                "capture_output": True,
                "text": True,
            }
            if self.shell_executable:
                run_kwargs["executable"] = self.shell_executable
            result = subprocess.run(self.command, **run_kwargs)
            stdout = result.stdout
            stderr = result.stderr if self.capture_stderr else ""
            combined = stdout + ("\n" + stderr if stderr else "")
            return remove_ansi(combined)
        except Exception as e:
            return f"Error running command {self.command}: {str(e)}\n"

    def _get_contents(self) -> str:
        """Format the command output."""
        if self.format == "xml":
            return f'<cmd exec="{self.command}">\n{self.command_output}\n</cmd>'
        else:
            return process_text(
                text=self.command_output,
                clean=False,
                range=None,
                format=self.format,
                label=self.command,
                shell_cmd=self.command if self.format == "shell" else None,
            )

    # Legacy aliases
    def run_command(self) -> str:
        """Legacy method - returns cached command output."""
        return self.command_output

    def get_contents(self) -> str:
        """Legacy method - returns cached formatted output."""
        return self.output


# Alias for backward compatibility
ShellReference = CommandReference


def _normalize_shell_executable(value: str | None) -> str | None:
    """Normalize a shell executable path."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_shell_executable(shell_override: str | None) -> str | None:
    """Resolve the shell executable to use."""
    shell_override = _normalize_shell_executable(shell_override)
    if shell_override:
        return shell_override
    return _normalize_shell_executable(os.environ.get("SHELL"))


def create_command_references(
    commands: List[str],
    format: str = "shell",
    capture_stderr: bool = True,
    shell_executable: str | None = None,
):
    """
    Create CommandReference objects for multiple commands.

    Args:
        commands: List of shell commands to execute
        format: Output format - "md", "xml", or "shell"
        capture_stderr: Whether to capture stderr
        shell_executable: Override the shell executable

    Returns:
        Dict with 'refs' (list of CommandReference) and 'concatenated' (joined output)
    """
    cmd_refs = []
    resolved_shell = _resolve_shell_executable(shell_executable)
    for cmd in commands:
        cmd_ref = CommandReference(
            cmd,
            format=format,
            capture_stderr=capture_stderr,
            shell_executable=resolved_shell,
        )
        cmd_refs.append(cmd_ref)

    concatenated = "\n\n".join(ref.output for ref in cmd_refs)
    return {
        "refs": cmd_refs,
        "concatenated": concatenated,
    }
