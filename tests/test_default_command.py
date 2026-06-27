from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from contextualize import cli


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


def test_prompt_only_mode_preserved_without_path() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["-p", "just a prompt"])

    assert result.exit_code == 0
    assert "just a prompt" in result.output
