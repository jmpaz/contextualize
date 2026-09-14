from __future__ import annotations

import io
import stat
import subprocess

import pytest

from contextualize import clipboard
from contextualize.clipboard import ClipboardDelivery, ClipboardError

PAYLOAD = "SENTINEL-PAYLOAD héllo ✓"
PREVIOUS = "SENTINEL-PREVIOUS-CLIPBOARD"
TMUX = "/tmp/tmux-1000/default,1,0"


class FakeCommands:
    def __init__(self, monkeypatch, available: set[str]) -> None:
        self.calls: list[dict] = []
        self.outcomes: dict[str, tuple[int, bytes, bytes] | BaseException] = {}
        monkeypatch.setattr(
            clipboard.shutil,
            "which",
            lambda command: f"/usr/bin/{command}" if command in available else None,
        )
        monkeypatch.setattr(clipboard.subprocess, "run", self.run)

    def script(self, command: str, *, returncode=0, stdout=b"", stderr=b"") -> None:
        self.outcomes[command] = (returncode, stdout, stderr)

    def fail_with(self, command: str, exc: BaseException) -> None:
        self.outcomes[command] = exc

    def call(self, command: str) -> dict:
        return next(call for call in self.calls if call["args"][0] == command)

    def run(self, args, **kwargs):
        self.calls.append({"args": list(args), **kwargs})
        outcome = self.outcomes.get(args[0], (0, b"", b""))
        if isinstance(outcome, BaseException):
            raise outcome
        returncode, stdout, stderr = outcome
        if hasattr(kwargs.get("stderr"), "write"):
            kwargs["stderr"].write(stderr)
        if kwargs.get("capture_output"):
            return subprocess.CompletedProcess(args, returncode, stdout, stderr)
        return subprocess.CompletedProcess(args, returncode)


@pytest.fixture(autouse=True)
def local_session(monkeypatch) -> None:
    for name in (
        "SSH_TTY",
        "SSH_CONNECTION",
        "TMUX",
        "WAYLAND_DISPLAY",
        "CONTEXTUALIZE_CLIPBOARD",
        "LC_ALL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def macos(monkeypatch) -> FakeCommands:
    monkeypatch.setattr(clipboard.sys, "platform", "darwin")
    return FakeCommands(monkeypatch, {"pbcopy", "pbpaste"})


@pytest.fixture
def wayland(monkeypatch) -> FakeCommands:
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    return FakeCommands(monkeypatch, {"wl-copy", "wl-paste"})


@pytest.fixture
def terminal_writes(monkeypatch) -> list[str]:
    writes: list[str] = []
    monkeypatch.setattr(clipboard, "write_terminal_sequence", writes.append)
    return writes


def _without_terminal(monkeypatch) -> None:
    def no_tty(*_args, **_kwargs):
        raise OSError("no controlling terminal")

    monkeypatch.setattr(clipboard, "open", no_tty, raising=False)
    monkeypatch.setattr(clipboard.sys, "stdout", io.StringIO())


def test_osc52_sequence_encodes_clipboard_payload() -> None:
    assert clipboard.osc52_sequence("hello") == "\x1b]52;c;aGVsbG8=\x07"


def test_tmux_passthrough_escapes_embedded_escape_bytes() -> None:
    sequence = clipboard.tmux_passthrough_sequence("\x1b]52;c;aGVsbG8=\x07")

    assert sequence == "\x1bPtmux;\x1b\x1b]52;c;aGVsbG8=\x07\x1b\\"


def test_delivery_wording_separates_verified_copies_from_sends() -> None:
    confirmed = ClipboardDelivery("pbcopy", confirmed=True)
    unconfirmed = ClipboardDelivery("OSC52", confirmed=False)

    assert confirmed.describe("12 tokens") == "Copied 12 tokens to clipboard"
    assert (
        unconfirmed.describe("12 tokens") == "Sent 12 tokens to your terminal via OSC52"
    )


def test_macos_copy_is_verified_by_reading_it_back(
    monkeypatch, macos, terminal_writes
) -> None:
    monkeypatch.setenv("LC_ALL", "C")
    macos.script("pbpaste", stdout=PAYLOAD.encode("utf-8"))

    delivery = clipboard.copy_to_clipboard(PAYLOAD)

    assert delivery == ClipboardDelivery("pbcopy", confirmed=True)
    copy_call = macos.call("pbcopy")
    assert copy_call["input"] == PAYLOAD.encode("utf-8")
    assert copy_call["env"]["LC_CTYPE"] == "UTF-8"
    assert "LC_ALL" not in copy_call["env"]
    assert macos.call("pbpaste")["args"] == ["pbpaste"]
    assert terminal_writes == []


def test_macos_pbcopy_failure_fails_without_osc52_fallback(
    macos, terminal_writes
) -> None:
    macos.script("pbcopy", returncode=1, stderr=b"pbcopy: pasteboard unavailable\n")

    with pytest.raises(ClipboardError) as excinfo:
        clipboard.copy_to_clipboard(PAYLOAD)

    assert (
        str(excinfo.value)
        == "pbcopy exited with status 1: pbcopy: pasteboard unavailable"
    )
    assert terminal_writes == []


def test_macos_stale_clipboard_is_a_failure(macos, terminal_writes) -> None:
    macos.script("pbpaste", stdout=PREVIOUS.encode("utf-8"))

    with pytest.raises(ClipboardError, match="holds different content") as excinfo:
        clipboard.copy_to_clipboard(PAYLOAD)

    assert PAYLOAD not in str(excinfo.value)
    assert PREVIOUS not in str(excinfo.value)
    assert terminal_writes == []


def test_native_copy_timeout_is_a_failure(macos, terminal_writes) -> None:
    macos.fail_with("pbcopy", subprocess.TimeoutExpired(["pbcopy"], 10))

    with pytest.raises(ClipboardError, match="pbcopy timed out"):
        clipboard.copy_to_clipboard(PAYLOAD)
    assert terminal_writes == []


def test_wayland_copy_uses_wl_clipboard_and_verifies(wayland, terminal_writes) -> None:
    wayland.script("wl-paste", stdout=PAYLOAD.encode("utf-8"))

    delivery = clipboard.copy_to_clipboard(PAYLOAD)

    assert delivery == ClipboardDelivery("wl-copy", confirmed=True)
    copy_call = wayland.call("wl-copy")
    assert copy_call["args"] == ["wl-copy", "--type", "text/plain"]
    assert copy_call["stdout"] is subprocess.DEVNULL
    assert hasattr(copy_call["stderr"], "write")
    assert "capture_output" not in copy_call
    assert wayland.call("wl-paste")["args"] == [
        "wl-paste",
        "--no-newline",
        "--type",
        "text/plain",
    ]
    assert terminal_writes == []


def test_wayland_connection_failure_fails_without_osc52_fallback(
    wayland, terminal_writes
) -> None:
    wayland.script(
        "wl-copy", returncode=1, stderr=b"Failed to connect to a Wayland server\n"
    )

    with pytest.raises(ClipboardError) as excinfo:
        clipboard.copy_to_clipboard(PAYLOAD)

    assert (
        str(excinfo.value)
        == "wl-copy exited with status 1: Failed to connect to a Wayland server"
    )
    assert terminal_writes == []


def test_readback_tolerates_line_ending_differences(wayland, terminal_writes) -> None:
    wayland.script("wl-paste", stdout=b"first\r\nsecond\n")

    assert clipboard.copy_to_clipboard("first\nsecond").confirmed


def test_pyperclip_copy_that_leaves_the_clipboard_stale_is_a_failure(
    monkeypatch, terminal_writes
) -> None:
    pyperclip = pytest.importorskip("pyperclip")
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    FakeCommands(monkeypatch, set())
    monkeypatch.setattr(pyperclip, "copy", lambda _text: None)
    monkeypatch.setattr(pyperclip, "paste", lambda: PREVIOUS)

    with pytest.raises(ClipboardError, match="pyperclip reported success"):
        clipboard.copy_to_clipboard(PAYLOAD)
    assert terminal_writes == []


def test_local_copy_without_a_native_clipboard_falls_back_to_osc52(
    monkeypatch, terminal_writes
) -> None:
    pyperclip = pytest.importorskip("pyperclip")
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    FakeCommands(monkeypatch, set())

    def no_mechanism(_text: str) -> None:
        raise pyperclip.PyperclipException("could not find a copy/paste mechanism")

    monkeypatch.setattr(pyperclip, "copy", no_mechanism)

    delivery = clipboard.copy_to_clipboard(PAYLOAD)

    assert delivery == ClipboardDelivery("OSC52", confirmed=False)
    assert terminal_writes == [clipboard.osc52_sequence(PAYLOAD)]


def test_ssh_tmux_copy_goes_through_load_buffer(monkeypatch, terminal_writes) -> None:
    monkeypatch.setattr(clipboard.sys, "platform", "darwin")
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.2 51000 10.0.0.1 22")
    monkeypatch.setenv("TMUX", TMUX)
    commands = FakeCommands(monkeypatch, {"tmux", "pbcopy", "pbpaste"})

    delivery = clipboard.copy_to_clipboard(PAYLOAD)

    assert delivery == ClipboardDelivery("tmux", confirmed=False)
    assert [call["args"] for call in commands.calls] == [
        ["tmux", "load-buffer", "-w", "-"]
    ]
    assert commands.call("tmux")["input"] == PAYLOAD.encode("utf-8")
    assert terminal_writes == []


def test_ssh_copy_writes_osc52_to_the_terminal(monkeypatch, terminal_writes) -> None:
    monkeypatch.setattr(clipboard.sys, "platform", "darwin")
    monkeypatch.setenv("SSH_TTY", "/dev/pts/1")
    commands = FakeCommands(monkeypatch, {"pbcopy", "pbpaste"})

    delivery = clipboard.copy_to_clipboard(PAYLOAD)

    assert delivery == ClipboardDelivery("OSC52", confirmed=False)
    assert terminal_writes == [clipboard.osc52_sequence(PAYLOAD)]
    assert commands.calls == []


def test_tmux_falls_back_to_passthrough_when_load_buffer_fails(
    monkeypatch, terminal_writes
) -> None:
    monkeypatch.setenv("SSH_TTY", "/dev/pts/1")
    monkeypatch.setenv("TMUX", TMUX)
    FakeCommands(monkeypatch, {"tmux"}).script("tmux", returncode=1)

    delivery = clipboard.copy_to_clipboard("tmux text")

    assert delivery == ClipboardDelivery("OSC52 (tmux passthrough)", confirmed=False)
    assert terminal_writes == ["\x1bPtmux;\x1b\x1b]52;c;dG11eCB0ZXh0\x07\x1b\\"]


def test_ssh_without_a_terminal_fails_rather_than_using_the_host_clipboard(
    monkeypatch,
) -> None:
    monkeypatch.setattr(clipboard.sys, "platform", "darwin")
    monkeypatch.setenv("SSH_TTY", "/dev/pts/1")
    commands = FakeCommands(monkeypatch, {"pbcopy", "pbpaste"})
    _without_terminal(monkeypatch)

    with pytest.raises(ClipboardError, match="no terminal available for OSC52 copy"):
        clipboard.copy_to_clipboard(PAYLOAD)
    assert commands.calls == []


@pytest.mark.parametrize("backend", ["native", "pyperclip"])
def test_native_override_applies_even_over_ssh(
    monkeypatch, macos, terminal_writes, backend
) -> None:
    monkeypatch.setenv("SSH_TTY", "/dev/pts/1")
    monkeypatch.setenv("CONTEXTUALIZE_CLIPBOARD", backend)
    macos.script("pbpaste", stdout=PAYLOAD.encode("utf-8"))

    delivery = clipboard.copy_to_clipboard(PAYLOAD)

    assert delivery == ClipboardDelivery("pbcopy", confirmed=True)
    assert terminal_writes == []


def test_osc52_override_skips_the_native_clipboard(
    monkeypatch, macos, terminal_writes
) -> None:
    monkeypatch.setenv("CONTEXTUALIZE_CLIPBOARD", "osc52")

    delivery = clipboard.copy_to_clipboard(PAYLOAD)

    assert delivery == ClipboardDelivery("OSC52", confirmed=False)
    assert macos.calls == []


def test_unknown_backend_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXTUALIZE_CLIPBOARD", "xclip")

    with pytest.raises(ClipboardError, match="auto, native, or osc52"):
        clipboard.copy_to_clipboard(PAYLOAD)


def test_unconfirmed_output_is_private_and_replaced(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    clipboard.save_unconfirmed_output("first")
    path = clipboard.save_unconfirmed_output(PAYLOAD)

    assert (
        path == tmp_path / "state" / "contextualize" / "clipboard" / "unconfirmed.txt"
    )
    assert path.read_text(encoding="utf-8") == PAYLOAD
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert list(path.parent.iterdir()) == [path]
