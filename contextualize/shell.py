import re
import subprocess
from typing import List

from .reference import process_text


def remove_ansi(text: str) -> str:
    """
    Remove ANSI escape sequences from text.
    """
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


class CommandReference:
    """
    A wrapper that runs a command, captures its output, and formats it
    with the same approach used by FileReference.
    """

    def __init__(
        self,
        command: str,
        format: str = "shell",
        label: str = "cmd",
        capture_stderr: bool = True,
    ):
        """
        :param command: The raw command string, e.g. "ls --help"
        :param format: "md"/"xml"/"shell"
        :param label: Label style; "cmd" means show the entire command.
        :param capture_stderr: Whether to capture stderr as well.
        """
        self.command = command
        self.format = format
        self.label = label or "cmd"
        self.capture_stderr = capture_stderr

        self.command_output = self.run_command()
        self.output = self.get_contents()

    def run_command(self) -> str:
        """
        Executes the command using shell=True so that shell features like pipes work.
        Captures stdout (and stderr if enabled), and returns a single string with ANSI
        escape sequences removed.
        """
        try:
            result = subprocess.run(
                self.command,
                shell=True,  # allows pipes, redirection, etc.
                capture_output=True,
                text=True,
            )
            stdout = result.stdout
            stderr = result.stderr if self.capture_stderr else ""
            combined = stdout + ("\n" + stderr if stderr else "")
            return remove_ansi(combined)
        except Exception as e:
            return f"Error running command {self.command}: {str(e)}\n"

    def get_contents(self) -> str:
        """
        Wrap the command output. For XML format, use custom tags;
        otherwise, rely on the standard process_text function.
        """
        if self.format == "xml":
            return f'<cmd exec="{self.command}">\n{self.command_output}\n</cmd>'
        else:
            label_str = self.command if self.label == "cmd" else self.label
            return process_text(
                text=self.command_output,
                clean=False,
                range=None,
                format=self.format,
                label=label_str,
                shell_cmd=self.command if self.format == "shell" else None,
            )


def create_command_references(
    commands: List[str],
    format: str = "shell",
    label: str = "cmd",
    capture_stderr: bool = True,
):
    """
    Runs each command, collects outputs as CommandReference objects,
    and concatenates them similarly to how file references are handled.
    """
    cmd_refs = []
    for cmd in commands:
        cmd_ref = CommandReference(
            cmd, format=format, label=label, capture_stderr=capture_stderr
        )
        cmd_refs.append(cmd_ref)

    concatenated = "\n\n".join(ref.output for ref in cmd_refs)
    return {
        "refs": cmd_refs,
        "concatenated": concatenated,
    }
