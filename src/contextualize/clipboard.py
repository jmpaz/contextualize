from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

NATIVE_COMMAND_TIMEOUT_SECONDS = 10
STDERR_EXCERPT_CHARS = 500


class ClipboardError(RuntimeError):
    pass


class ClipboardUnavailable(ClipboardError):
    pass


@dataclass(frozen=True)
class ClipboardDelivery:
    route: str
    confirmed: bool

    def describe(self, subject: str) -> str:
        if self.confirmed:
            return f"Copied {subject} to clipboard"
        return f"Sent {subject} to your terminal via {self.route}"


@dataclass(frozen=True)
class NativeClipboard:
    name: str
    copy_args: tuple[str, ...]
    paste_args: tuple[str, ...]
    env: dict[str, str] | None = None


def copy_to_clipboard(text: str) -> ClipboardDelivery:
    backend = os.environ.get("CONTEXTUALIZE_CLIPBOARD", "auto").strip().lower()
    if backend == "osc52":
        return copy_to_terminal(text)
    if backend in ("native", "pyperclip"):
        return copy_to_native(text)
    if backend != "auto":
        raise ClipboardError(
            "CONTEXTUALIZE_CLIPBOARD must be one of auto, native, or osc52"
        )

    if is_remote_session():
        routes = (copy_to_terminal,)
    elif os.environ.get("TMUX"):
        routes = (copy_to_terminal, copy_to_native)
    else:
        routes = (copy_to_native, copy_to_terminal)

    unavailable: list[str] = []
    for route in routes:
        try:
            return route(text)
        except ClipboardUnavailable as exc:
            unavailable.append(str(exc))
    raise ClipboardError("; ".join(unavailable))


def paste_from_clipboard() -> str:
    try:
        from pyperclip import paste
    except ImportError as exc:
        raise ClipboardError("pyperclip is required for paste") from exc
    return paste()


def is_remote_session() -> bool:
    return bool(os.environ.get("SSH_TTY") or os.environ.get("SSH_CONNECTION"))


def copy_to_native(text: str) -> ClipboardDelivery:
    clipboard = native_clipboard()
    if clipboard is None:
        return copy_with_pyperclip(text)
    return copy_with_command(clipboard, text)


def native_clipboard() -> NativeClipboard | None:
    if sys.platform == "darwin" and commands_available("pbcopy", "pbpaste"):
        env = {name: value for name, value in os.environ.items() if name != "LC_ALL"}
        env["LC_CTYPE"] = "UTF-8"
        return NativeClipboard("pbcopy", ("pbcopy",), ("pbpaste",), env)
    if os.environ.get("WAYLAND_DISPLAY") and commands_available("wl-copy", "wl-paste"):
        return NativeClipboard(
            "wl-copy",
            ("wl-copy", "--type", "text/plain"),
            ("wl-paste", "--no-newline", "--type", "text/plain"),
        )
    return None


def commands_available(*commands: str) -> bool:
    return all(shutil.which(command) for command in commands)


def copy_with_command(clipboard: NativeClipboard, text: str) -> ClipboardDelivery:
    # wl-copy's background child inherits stderr, so a pipe would never reach EOF.
    with tempfile.TemporaryFile() as stderr:
        result = run_native(
            clipboard,
            clipboard.copy_args,
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )
        if result.returncode != 0:
            stderr.seek(0)
            raise ClipboardError(
                f"{clipboard.copy_args[0]} exited with status {result.returncode}"
                f"{stderr_excerpt(stderr.read())}"
            )

    readback = run_native(clipboard, clipboard.paste_args, capture_output=True)
    if readback.returncode != 0:
        raise ClipboardError(
            f"{clipboard.paste_args[0]} could not read the clipboard back "
            f"(status {readback.returncode}){stderr_excerpt(readback.stderr)}"
        )
    verify_readback(
        clipboard.name, text, readback.stdout.decode("utf-8", errors="replace")
    )
    return ClipboardDelivery(clipboard.name, confirmed=True)


def run_native(
    clipboard: NativeClipboard, args: tuple[str, ...], **kwargs
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(args),
            env=clipboard.env,
            timeout=NATIVE_COMMAND_TIMEOUT_SECONDS,
            check=False,
            **kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClipboardError(
            f"{args[0]} timed out after {NATIVE_COMMAND_TIMEOUT_SECONDS}s"
        ) from exc
    except OSError as exc:
        raise ClipboardError(f"{args[0]} could not run: {exc}") from exc


def stderr_excerpt(raw: bytes) -> str:
    message = raw.decode("utf-8", errors="replace").strip()
    return f": {message[-STDERR_EXCERPT_CHARS:]}" if message else ""


def verify_readback(name: str, expected: str, actual: str) -> None:
    if normalize_readback(actual) != normalize_readback(expected):
        raise ClipboardError(
            f"{name} reported success but the clipboard holds different content "
            f"({len(actual)} characters; expected {len(expected)})"
        )


def normalize_readback(text: str) -> str:
    return text.replace("\r\n", "\n").rstrip("\n")


def copy_with_pyperclip(text: str) -> ClipboardDelivery:
    try:
        import pyperclip
    except ImportError as exc:
        raise ClipboardUnavailable(
            "no native clipboard command found and pyperclip is not installed"
        ) from exc
    try:
        pyperclip.copy(text)
        readback = pyperclip.paste()
    except pyperclip.PyperclipException as exc:
        raise ClipboardUnavailable(f"pyperclip: {exc}") from exc
    verify_readback("pyperclip", text, readback)
    return ClipboardDelivery("pyperclip", confirmed=True)


def copy_to_terminal(text: str) -> ClipboardDelivery:
    if not os.environ.get("TMUX"):
        write_terminal_sequence(osc52_sequence(text))
        return ClipboardDelivery("OSC52", confirmed=False)
    if copy_with_tmux(text):
        return ClipboardDelivery("tmux", confirmed=False)
    write_terminal_sequence(tmux_passthrough_sequence(osc52_sequence(text)))
    return ClipboardDelivery("OSC52 (tmux passthrough)", confirmed=False)


def copy_with_tmux(text: str) -> bool:
    if not shutil.which("tmux"):
        return False
    try:
        result = subprocess.run(
            ["tmux", "load-buffer", "-w", "-"],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def osc52_sequence(text: str) -> str:
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"\x1b]52;c;{payload}\x07"


def tmux_passthrough_sequence(sequence: str) -> str:
    escaped = sequence.replace("\x1b", "\x1b\x1b")
    return f"\x1bPtmux;{escaped}\x1b\\"


def write_terminal_sequence(sequence: str) -> None:
    with open_terminal() as terminal:
        try:
            terminal.write(sequence.encode("ascii"))
            terminal.flush()
        except OSError as exc:
            raise ClipboardError(
                f"could not write OSC52 to the terminal: {exc}"
            ) from exc


@contextmanager
def open_terminal():
    try:
        with open("/dev/tty", "wb", buffering=0) as terminal:
            yield terminal
            return
    except OSError as exc:
        stdout_buffer = getattr(sys.stdout, "buffer", None)
        if stdout_buffer is not None and sys.stdout.isatty():
            yield stdout_buffer
            return
        raise ClipboardUnavailable("no terminal available for OSC52 copy") from exc


def unconfirmed_output_path() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME")
    root = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return root / "contextualize" / "clipboard" / "unconfirmed.txt"


def save_unconfirmed_output(text: str) -> Path:
    path = unconfirmed_output_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path
