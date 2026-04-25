from __future__ import annotations

from contextualize import clipboard


def test_osc52_sequence_encodes_clipboard_payload() -> None:
    assert clipboard.osc52_sequence("hello") == "\x1b]52;c;aGVsbG8=\x07"


def test_tmux_passthrough_escapes_embedded_escape_bytes() -> None:
    sequence = clipboard.tmux_passthrough_sequence("\x1b]52;c;aGVsbG8=\x07")

    assert sequence == "\x1bPtmux;\x1b\x1b]52;c;aGVsbG8=\x07\x1b\\"


def test_auto_copy_prefers_osc52_in_ssh(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setenv("SSH_TTY", "/dev/pts/1")
    monkeypatch.delenv("CONTEXTUALIZE_CLIPBOARD", raising=False)
    monkeypatch.setattr(
        clipboard, "copy_with_osc52", lambda text: calls.append(("osc52", text))
    )
    monkeypatch.setattr(
        clipboard,
        "copy_with_pyperclip",
        lambda text: calls.append(("pyperclip", text)),
    )

    clipboard.copy_to_clipboard("remote text")

    assert calls == [("osc52", "remote text")]


def test_auto_copy_falls_back_to_osc52_when_pyperclip_fails(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("CONTEXTUALIZE_CLIPBOARD", raising=False)

    def fail_pyperclip(text: str) -> None:
        calls.append(("pyperclip", text))
        raise RuntimeError("no local clipboard")

    monkeypatch.setattr(clipboard, "copy_with_pyperclip", fail_pyperclip)
    monkeypatch.setattr(
        clipboard, "copy_with_osc52", lambda text: calls.append(("osc52", text))
    )

    clipboard.copy_to_clipboard("fallback text")

    assert calls == [("pyperclip", "fallback text"), ("osc52", "fallback text")]


def test_tmux_osc52_uses_load_buffer_when_available(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Result:
        returncode = 0

    def run_tmux(args, *, input, stdout, stderr, check):
        captured["args"] = args
        captured["input"] = input
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["check"] = check
        return Result()

    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    monkeypatch.setattr(
        clipboard.shutil, "which", lambda command: f"/usr/bin/{command}"
    )
    monkeypatch.setattr(clipboard.subprocess, "run", run_tmux)
    monkeypatch.setattr(
        clipboard,
        "write_terminal_sequence",
        lambda _sequence: (_ for _ in ()).throw(AssertionError("unexpected write")),
    )

    clipboard.copy_with_osc52("tmux text")

    assert captured["args"] == ["tmux", "load-buffer", "-w", "-"]
    assert captured["input"] == b"tmux text"
    assert captured["check"] is False


def test_tmux_osc52_falls_back_to_passthrough_sequence(monkeypatch) -> None:
    captured: dict[str, str] = {}

    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    monkeypatch.setattr(clipboard, "copy_with_tmux", lambda _text: False)
    monkeypatch.setattr(
        clipboard,
        "write_terminal_sequence",
        lambda sequence: captured.setdefault("sequence", sequence),
    )

    clipboard.copy_with_osc52("tmux text")

    assert captured["sequence"] == "\x1bPtmux;\x1b\x1b]52;c;dG11eCB0ZXh0\x07\x1b\\"
