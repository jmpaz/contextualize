from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from contextualize import cli
from contextualize.clipboard import ClipboardDelivery, ClipboardError
from contextualize.plugins import clear_loaded_plugins_cache
from contextualize.plugins import loader as plugin_loader

PAYLOAD = "SENTINEL-PAYLOAD-7f3a"
SEGMENTED_PAYLOAD = (
    f"```\nfirst {PAYLOAD} block\n```\n\n```\nsecond {PAYLOAD} block\n```"
)
PROMPTS = ["-p", "before", "-p", "after"]
CONFIRMED = ClipboardDelivery("pbcopy", confirmed=True)
UNCONFIRMED = ClipboardDelivery("OSC52", confirmed=False)


@pytest.fixture
def saved_output(monkeypatch, tmp_path: Path, request) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(plugin_loader, "_iter_plugin_entrypoints", list)
    clear_loaded_plugins_cache()
    request.addfinalizer(clear_loaded_plugins_cache)
    monkeypatch.setattr(cli, "_stdin_is_capturable", lambda: True)
    return tmp_path / "state" / "contextualize" / "clipboard" / "unconfirmed.txt"


def _clipboard(monkeypatch, *outcomes) -> list[str]:
    copied: list[str] = []
    remaining = list(outcomes)

    def copy(text: str) -> ClipboardDelivery:
        copied.append(text)
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(cli, "copy_to_clipboard", copy)
    return copied


def _enter_pressed(monkeypatch, pressed: bool) -> None:
    monkeypatch.setattr("contextualize.utils.wait_for_enter", lambda: pressed)


def _invoke(args: list[str], payload: str = PAYLOAD):
    return CliRunner().invoke(cli.cli, args, input=payload)


def _printed_output(args: list[str], payload: str = PAYLOAD) -> str:
    result = _invoke(args, payload)
    assert result.exit_code == 0, result.output
    return result.output.removesuffix("\n")


def test_verified_copy_reports_copied_and_keeps_nothing(
    monkeypatch, saved_output
) -> None:
    copied = _clipboard(monkeypatch, CONFIRMED)

    result = _invoke(["--copy"])

    assert result.exit_code == 0, result.output
    assert copied == [_printed_output([])]
    assert "Copied " in result.output
    assert "to clipboard." in result.output
    assert not saved_output.exists()


def test_unconfirmed_copy_says_sent_and_keeps_the_output(
    monkeypatch, saved_output
) -> None:
    _clipboard(monkeypatch, UNCONFIRMED)

    result = _invoke([*PROMPTS, "--copy"])

    assert result.exit_code == 0, result.output
    assert "to your terminal via OSC52." in result.output
    assert "Copied" not in result.output
    assert f"Kept a copy at {saved_output}." in result.output
    assert saved_output.read_text(encoding="utf-8") == _printed_output(PROMPTS)


def test_failed_copy_exits_nonzero_and_saves_the_output(
    monkeypatch, saved_output
) -> None:
    _clipboard(monkeypatch, ClipboardError("pbcopy exited with status 1"))

    result = _invoke([*PROMPTS, "--copy"])

    assert result.exit_code == 1
    assert (
        "Could not copy output to clipboard: pbcopy exited with status 1"
        in result.output
    )
    assert f"Full output saved to {saved_output}" in result.output
    assert PAYLOAD not in result.output
    assert saved_output.read_text(encoding="utf-8") == _printed_output(PROMPTS)


def test_staged_copy_failure_names_the_stage(monkeypatch, saved_output) -> None:
    _enter_pressed(monkeypatch, True)
    copied = _clipboard(
        monkeypatch, CONFIRMED, ClipboardError("wl-copy exited with status 1")
    )

    result = _invoke([*PROMPTS, "--staged-copy", "--copy"])

    assert result.exit_code == 1
    assert copied[0] == "before"
    assert (
        "Could not copy content to clipboard: wl-copy exited with status 1"
        in result.output
    )
    assert saved_output.read_text(encoding="utf-8") == _printed_output(PROMPTS)


def test_segmented_copy_failure_names_the_segment(monkeypatch, saved_output) -> None:
    _enter_pressed(monkeypatch, True)
    _clipboard(monkeypatch, CONFIRMED, ClipboardError("wl-copy exited with status 1"))

    result = _invoke(["--copy-segments", "5"], SEGMENTED_PAYLOAD)

    assert result.exit_code == 1
    assert "Could not copy segment 2/2 to clipboard" in result.output
    assert saved_output.read_text(encoding="utf-8") == _printed_output(
        [], SEGMENTED_PAYLOAD
    )


def test_interrupted_staged_copy_exits_nonzero_and_saves_the_output(
    monkeypatch, saved_output
) -> None:
    _enter_pressed(monkeypatch, False)
    _clipboard(monkeypatch, CONFIRMED)

    result = _invoke([*PROMPTS, "--staged-copy", "--copy"])

    assert result.exit_code == 1
    assert "Copying interrupted before content." in result.output
    assert saved_output.read_text(encoding="utf-8") == _printed_output(PROMPTS)
