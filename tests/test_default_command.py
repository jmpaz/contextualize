from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from contextualize import cli


def _complete(line: str) -> str:
    result = CliRunner().invoke(
        cli.cli,
        [],
        prog_name="contextualize",
        env={
            "_CONTEXTUALIZE_COMPLETE": "fish_complete",
            "COMP_WORDS": line,
            "COMP_CWORD": line.rsplit(" ", 1)[-1],
        },
    )
    assert result.exit_code == 0
    return result.output


def test_implicit_cat_matches_explicit_cat() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("note.txt").write_text("hello world\n", encoding="utf-8")

        implicit = runner.invoke(cli.cli, ["note.txt"])
        explicit = runner.invoke(cli.cli, ["cat", "note.txt"])

        assert implicit.exit_code == 0
        assert explicit.exit_code == 0
        assert implicit.output == explicit.output
        assert "hello world" in implicit.output


def test_bare_invocation_shows_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.cli, [])

    assert result.exit_code == 0
    assert "COMMAND [ARGS]" in result.output
    assert "Contextualize CLI" in result.output


def test_subcommand_wins_over_same_named_path() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("map").write_text("UNIQUE-MAP-FILE-CONTENT\n", encoding="utf-8")

        as_subcommand = runner.invoke(cli.cli, ["map"])
        as_path = runner.invoke(cli.cli, ["./map"])

        assert "UNIQUE-MAP-FILE-CONTENT" not in as_subcommand.output
        assert "UNIQUE-MAP-FILE-CONTENT" in as_path.output


def test_root_option_forwarded_through_implicit_cat() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("note.txt").write_text("hello world\n", encoding="utf-8")

        trailing = runner.invoke(cli.cli, ["note.txt", "--count"])
        leading = runner.invoke(cli.cli, ["--count", "note.txt"])

        assert trailing.exit_code == 0
        assert leading.exit_code == 0
        assert "tokens" in trailing.output
        assert trailing.output == leading.output


def test_short_cluster_forwarded_through_implicit_cat() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("note.txt").write_text("hello world\n", encoding="utf-8")

        clustered = runner.invoke(cli.cli, ["-wp", "PRE", "note.txt"])

        assert clustered.exit_code == 0
        assert "hello world" in clustered.output
        assert "PRE" in clustered.output


def test_subcommand_keeps_options_it_defines(monkeypatch) -> None:
    captured: dict[str, bool] = {}

    def _hydrate_contexts(*_args, **kwargs):
        captured["copy"] = kwargs["overrides"].copy
        return []

    monkeypatch.setattr(
        "contextualize.manifest.contexts.hydrate_contexts", _hydrate_contexts
    )
    runner = CliRunner()

    nested = runner.invoke(cli.cli, ["contexts", "hydrate", "--copy", "demo"])
    paste = runner.invoke(cli.cli, ["paste", "--count", "0"])

    assert nested.exit_code == 0, nested.output
    assert "ignoring global options" not in nested.output
    assert captured["copy"] is True
    assert "--count must be at least 1" in paste.output


def test_prompt_only_mode_preserved_without_path() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["-p", "just a prompt"])

    assert result.exit_code == 0
    assert "just a prompt" in result.output


@pytest.mark.parametrize(
    "line",
    [
        "contextualize --copy cat docs/us",
        "contextualize docs/us",
        "contextualize README.md docs/us",
        "contextualize map docs/us",
        "contextualize hydrate docs/us",
        "contextualize cat docs/{usage,us",
    ],
)
def test_target_arguments_complete_as_file_paths(line: str) -> None:
    incomplete = line.rsplit(" ", 1)[-1]
    assert f"file,{incomplete}" in _complete(line)


def test_root_completion_keeps_subcommands_and_options() -> None:
    assert "plain,cat" in _complete("contextualize ca")

    options = _complete("contextualize --cop")
    assert "plain,--copy" in options
    assert "file," not in options


def test_fish_script_keeps_commas_in_completion_values() -> None:
    result = CliRunner().invoke(
        cli.cli,
        [],
        prog_name="contextualize",
        env={"_CONTEXTUALIZE_COMPLETE": "fish_source"},
    )

    assert result.exit_code == 0
    assert 'string split -m 1 ","' in result.output
