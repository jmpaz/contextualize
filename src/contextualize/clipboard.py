from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager


class ClipboardError(RuntimeError):
    pass


def copy_to_clipboard(text: str) -> None:
    backend = os.environ.get("CONTEXTUALIZE_CLIPBOARD", "auto").strip().lower()
    if backend == "osc52":
        copy_with_osc52(text)
        return
    if backend == "pyperclip":
        copy_with_pyperclip(text)
        return
    if backend != "auto":
        raise ClipboardError(
            "CONTEXTUALIZE_CLIPBOARD must be one of auto, osc52, or pyperclip"
        )

    copy_functions = (
        (copy_with_osc52, copy_with_pyperclip)
        if should_prefer_osc52()
        else (copy_with_pyperclip, copy_with_osc52)
    )
    errors: list[str] = []
    for copy_function in copy_functions:
        try:
            copy_function(text)
            return
        except Exception as exc:
            errors.append(f"{copy_function.__name__}: {exc}")

    raise ClipboardError("; ".join(errors))


def paste_from_clipboard() -> str:
    try:
        from pyperclip import paste
    except ImportError as exc:
        raise ClipboardError("pyperclip is required for paste") from exc
    return paste()


def should_prefer_osc52() -> bool:
    return bool(
        os.environ.get("SSH_TTY")
        or os.environ.get("SSH_CONNECTION")
        or os.environ.get("TMUX")
    )


def copy_with_pyperclip(text: str) -> None:
    try:
        from pyperclip import copy
    except ImportError as exc:
        raise ClipboardError("pyperclip is required for pyperclip copy") from exc
    copy(text)


def copy_with_osc52(text: str) -> None:
    if os.environ.get("TMUX") and copy_with_tmux(text):
        return

    sequence = osc52_sequence(text)
    if os.environ.get("TMUX"):
        sequence = tmux_passthrough_sequence(sequence)
    write_terminal_sequence(sequence)


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
        terminal.write(sequence.encode("ascii"))
        terminal.flush()


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
        raise ClipboardError("no terminal available for OSC52 copy") from exc
