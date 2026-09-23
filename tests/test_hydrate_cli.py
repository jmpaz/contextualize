from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from contextualize import cli

PLAIN_TARGET = "store:notes/plain.md"
OTHER_TARGET = "store:notes/other.md"


def _hydrate(*args: str):
    return CliRunner().invoke(cli.cli, ["hydrate", *args, "--quiet"])


def _answer_confirmations(monkeypatch, answer: bool | None) -> list[str]:
    prompts: list[str] = []

    def _confirm(prompt: str) -> bool | None:
        prompts.append(prompt)
        return answer

    monkeypatch.setattr(cli, "_confirm_on_tty", _confirm)
    return prompts


@pytest.fixture(autouse=True)
def prompts(monkeypatch) -> list[str]:
    return _answer_confirmations(monkeypatch, None)


@pytest.fixture
def store_dir(fake_store, tmp_path: Path) -> Path:
    fake_store.entries["notes/other.md"] = "other prose"
    context_dir = tmp_path / "out"
    for target in (PLAIN_TARGET, OTHER_TARGET):
        result = _hydrate(target, "--dir", str(context_dir))
        assert result.exit_code == 0, result.output
    return context_dir


@pytest.fixture
def local_note(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    note = tmp_path / "notes/plain.md"
    note.parent.mkdir()
    note.write_text("local prose", encoding="utf-8")
    return note


def test_successive_hydrations_share_a_directory(
    fake_store, tmp_path: Path, prompts: list[str]
) -> None:
    fake_store.entries["notes/other.md"] = "other prose"
    context_dir = tmp_path / "out"

    first = _hydrate(PLAIN_TARGET, "--dir", str(context_dir))
    second = _hydrate(OTHER_TARGET, "--dir", str(context_dir))

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert prompts == []
    assert (context_dir / "notes/plain.md").read_text() == "just prose, no timing"
    assert (context_dir / "notes/other.md").read_text() == "other prose"


def test_rehydrating_a_target_asks_to_replace_only_its_file(
    store_dir: Path, fake_store, monkeypatch
) -> None:
    fake_store.entries["notes/plain.md"] = "revised prose"
    prompts = _answer_confirmations(monkeypatch, True)

    result = _hydrate(PLAIN_TARGET, "--dir", str(store_dir))

    assert result.exit_code == 0, result.output
    assert len(prompts) == 1
    assert f"  {store_dir / 'notes/plain.md'}\n" in result.stderr
    assert f"  {store_dir / 'notes'}\n" not in result.stderr
    assert (store_dir / "notes/plain.md").read_text() == "revised prose"
    assert (store_dir / "notes/other.md").read_text() == "other prose"


def test_declining_replacement_leaves_files_untouched(
    store_dir: Path, fake_store, monkeypatch
) -> None:
    fake_store.entries["notes/plain.md"] = "revised prose"
    _answer_confirmations(monkeypatch, False)

    result = _hydrate(PLAIN_TARGET, "--dir", str(store_dir))

    assert result.exit_code == 1
    assert (store_dir / "notes/plain.md").read_text() == "just prose, no timing"


def test_overwrite_replaces_only_its_file_without_asking(
    store_dir: Path, fake_store, prompts: list[str]
) -> None:
    fake_store.entries["notes/plain.md"] = "revised prose"

    result = _hydrate(PLAIN_TARGET, "--dir", str(store_dir), "--overwrite")

    assert result.exit_code == 0, result.output
    assert prompts == []
    assert (store_dir / "notes/plain.md").read_text() == "revised prose"
    assert (store_dir / "notes/other.md").read_text() == "other prose"


def test_replacing_a_hydrated_symlink_leaves_its_source_alone(
    fake_store, local_note: Path
) -> None:
    linked = _hydrate("notes/plain.md", "--dir", "out")
    replaced = _hydrate(PLAIN_TARGET, "--dir", "out", "--overwrite")

    assert linked.exit_code == 0, linked.output
    assert replaced.exit_code == 0, replaced.output
    hydrated = local_note.parent.parent / "out/notes/plain.md"
    assert not hydrated.is_symlink()
    assert hydrated.read_text() == "just prose, no timing"
    assert local_note.read_text() == "local prose"


def test_hydrate_refuses_to_replace_a_directory(local_note: Path) -> None:
    kept = local_note.parent.parent / "out/notes/plain.md/kept.md"
    kept.parent.mkdir(parents=True)
    kept.write_text("kept", encoding="utf-8")

    result = _hydrate("notes/plain.md", "--dir", "out", "--overwrite")

    assert result.exit_code == 1
    assert "a directory exists there" in result.output
    assert kept.read_text() == "kept"


def test_hydrate_copy_option_copies_instead_of_linking(local_note: Path) -> None:
    result = _hydrate("--copy", "notes/plain.md", "--dir", "out")

    assert result.exit_code == 0, result.output
    assert "ignoring global options" not in result.output
    copied = local_note.parent.parent / "out/notes/plain.md"
    assert not copied.is_symlink()
    assert copied.read_text() == "local prose"


def test_noninteractive_hydrate_requires_overwrite_to_replace(
    tmp_path: Path, local_note: Path
) -> None:
    hydrated = tmp_path / "out/notes/plain.md"
    hydrated.parent.mkdir(parents=True)
    hydrated.write_text("stale", encoding="utf-8")
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg-config"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
    }

    def run(*extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "contextualize.cli", "hydrate", "notes/plain.md"]
            + ["--dir", "out", "--quiet", *extra],
            cwd=tmp_path,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            start_new_session=True,
            timeout=60,
        )

    refused = run()

    assert refused.returncode == 1
    assert "Use --overwrite" in refused.stderr
    assert hydrated.read_text() == "stale"

    replaced = run("--overwrite")

    assert replaced.returncode == 0, replaced.stderr
    assert hydrated.resolve() == local_note.resolve()
