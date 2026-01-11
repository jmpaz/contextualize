"""Shell command reference implementation."""

import os
import re
import subprocess

from ..render import process_text
from ..utils import count_tokens


def remove_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def _normalize_shell_executable(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_shell_executable(shell_override: str | None) -> str | None:
    shell_override = _normalize_shell_executable(shell_override)
    if shell_override:
        return shell_override
    return _normalize_shell_executable(os.environ.get("SHELL"))


class ShellReference:
    """Reference to shell command output.

    Also known as CommandReference for backwards compatibility.
    """

    def __init__(
        self,
        command: str,
        format: str = "shell",
        capture_stderr: bool = True,
        shell_executable: str | None = None,
    ):
        """
        :param command: The raw command string, e.g. "ls --help"
        :param format: "md"/"xml"/"shell"
        :param capture_stderr: Whether to capture stderr as well.
        """
        self.command = command
        self.format = format
        self.capture_stderr = capture_stderr
        self.shell_executable = shell_executable

        self._command_output = self._run_command()
        self._output = self._get_contents()

    @property
    def path(self) -> str:
        return self.command

    @property
    def file_content(self) -> str:
        return self._command_output

    @property
    def output(self) -> str:
        return self._output

    @property
    def label(self) -> str:
        return self.command

    def read(self) -> str:
        """Read and return the command output."""
        return self._command_output

    def exists(self) -> bool:
        """Check if the command can be executed."""
        return True  # Commands always "exist"

    def token_count(self, encoding: str = "cl100k_base") -> int:
        """Count tokens in the command output."""
        return count_tokens(self._command_output, target=encoding)["count"]

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
        if self.format == "xml":
            return f'<cmd exec="{self.command}">\n{self._command_output}\n</cmd>'
        else:
            return process_text(
                text=self._command_output,
                clean=False,
                range=None,
                format=self.format,
                label=self.command,
                shell_cmd=self.command if self.format == "shell" else None,
            )


# Alias for backwards compatibility
CommandReference = ShellReference


def create_command_references(
    commands: list[str],
    format: str = "shell",
    capture_stderr: bool = True,
    shell_executable: str | None = None,
) -> dict:
    """
    Runs each command, collects outputs as ShellReference objects,
    and concatenates them similarly to how file references are handled.
    """
    cmd_refs = []
    resolved_shell = _resolve_shell_executable(shell_executable)
    for cmd in commands:
        cmd_ref = ShellReference(
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
