from __future__ import annotations

import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from contextualize import cli
from contextualize.plugins import clear_loaded_plugins_cache
from contextualize.plugins import loader as plugin_loader
from contextualize.runtime import get_verbose_logging


@pytest.fixture
def no_plugins(monkeypatch, request) -> None:
    monkeypatch.setattr(plugin_loader, "_iter_plugin_entrypoints", lambda: [])
    clear_loaded_plugins_cache()
    request.addfinalizer(clear_loaded_plugins_cache)


def _invoke_cat(monkeypatch, tmp_path: Path, args: list[str]):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    note_path = tmp_path / "note.txt"
    note_path.write_text("hello", encoding="utf-8")
    captured: dict[str, bool] = {}

    def _create_file_references(*_args, **_kwargs):
        captured.setdefault("verbose", get_verbose_logging())
        return {
            "refs": [
                types.SimpleNamespace(
                    output="hello",
                    file_content="hello",
                    original_file_content="hello",
                    path=str(note_path),
                )
            ],
            "concatenated": "hello",
            "ignored_files": [],
            "ignored_folders": {},
        }

    monkeypatch.setattr(
        "contextualize.references.create_file_references", _create_file_references
    )
    monkeypatch.setattr("contextualize.cli.copy_to_clipboard", lambda _text: None)

    result = CliRunner().invoke(cli.cli, [*args, str(note_path)])
    return result, captured


def test_cat_stdout_keeps_progress_quiet_by_default(
    monkeypatch, tmp_path: Path, no_plugins
) -> None:
    result, captured = _invoke_cat(monkeypatch, tmp_path, ["cat"])

    assert result.exit_code == 0
    assert captured["verbose"] is False
    assert result.output == "hello\n"


def test_cat_copy_enables_progress_by_default(
    monkeypatch, tmp_path: Path, no_plugins
) -> None:
    result, captured = _invoke_cat(monkeypatch, tmp_path, ["cat", "--copy"])

    assert result.exit_code == 0
    assert captured["verbose"] is True
    assert "Copied" in result.output


def test_cat_trace_enables_progress_by_default(
    monkeypatch, tmp_path: Path, no_plugins
) -> None:
    result, captured = _invoke_cat(monkeypatch, tmp_path, ["cat", "--trace"])

    assert result.exit_code == 0
    assert captured["verbose"] is True
    assert "Inputs:" in result.output


def test_cat_quiet_suppresses_copy_progress_default(
    monkeypatch, tmp_path: Path, no_plugins
) -> None:
    result, captured = _invoke_cat(
        monkeypatch, tmp_path, ["cat", "--copy", "--quiet"]
    )

    assert result.exit_code == 0
    assert captured["verbose"] is False


def test_result_message_survives_live_progress(capsys) -> None:
    import click

    from contextualize import cli as cli_module
    from contextualize.progress import (
        live_progress_active,
        reset_progress,
        set_live_progress,
    )

    ctx = click.Context(cli_module.cli)
    ctx.obj = {"verbose_logging": False}

    reset_progress()
    set_live_progress(True, force_terminal=True, transient=False)
    try:
        assert live_progress_active()
        cli_module._echo_result(ctx, "Copied 20488 tokens (cl100k_base) to clipboard.")
        assert "Copied" not in capsys.readouterr().out
        cli_module._finish_run(ctx)
    finally:
        set_live_progress(False)
        reset_progress()

    assert "Copied 20488 tokens (cl100k_base) to clipboard." in capsys.readouterr().out
