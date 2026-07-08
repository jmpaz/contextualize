from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from contextualize import cli
from contextualize.serve.core import show as query_show


def _isolated_env(tmp_path: Path) -> dict:
    return {"XDG_CONFIG_HOME": str(tmp_path / "xdg-config")}


def _write_manifest(drive: Path) -> None:
    drive.mkdir(parents=True, exist_ok=True)
    (drive / "manifest.yaml").write_text(
        "config:\n"
        "  name: demo\n"
        "  context:\n"
        "    dir: .context/demo\n"
        "    include-meta: true\n"
        "components:\n"
        "  - name: alpha\n"
        "    files:\n"
        "      - a.md  # first note\n"
        "      - b.md\n"
        "  - name: refs\n"
        "    files:\n"
        "      - a.md\n"
        "      # - d.md  # excluded\n",
        encoding="utf-8",
    )
    (drive / "a.md").write_text("alpha content\n", encoding="utf-8")
    (drive / "b.md").write_text("beta content\n", encoding="utf-8")


def test_show_cli_text_and_json_match_core(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    _write_manifest(drive)
    runner = CliRunner()

    text_result = runner.invoke(
        cli.cli, ["show", str(drive / "manifest.yaml")], env=_isolated_env(tmp_path)
    )
    assert text_result.exit_code == 0
    assert "alpha" in text_result.output
    assert "first note" in text_result.output

    json_result = runner.invoke(
        cli.cli, ["show", str(drive / "manifest.yaml"), "--json"], env=_isolated_env(tmp_path)
    )
    assert json_result.exit_code == 0
    payload = json.loads(json_result.output)
    expected = query_show(str(drive / "manifest.yaml"), cwd=str(drive))
    assert payload == expected


def test_show_cli_exits_nonzero_for_not_found(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["show", "nowhere-registered:x", "--registry", str(_empty_registry(tmp_path))],
        env=_isolated_env(tmp_path),
    )
    assert result.exit_code == 1
    payload_result = runner.invoke(
        cli.cli,
        ["show", "nowhere-registered:x", "--registry", str(_empty_registry(tmp_path)), "--json"],
        env=_isolated_env(tmp_path),
    )
    assert payload_result.exit_code == 1
    assert json.loads(payload_result.output)["state"] == "not-found"


def test_show_cli_exits_zero_for_unhydrated_and_disabled(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    _write_manifest(drive)
    runner = CliRunner()

    root = runner.invoke(cli.cli, ["show", str(drive / "manifest.yaml")], env=_isolated_env(tmp_path))
    assert root.exit_code == 0
    assert "unhydrated" in root.output

    disabled = runner.invoke(
        cli.cli, ["show", f"{drive / 'manifest.yaml'}:refs.d"], env=_isolated_env(tmp_path)
    )
    assert disabled.exit_code == 0
    assert "disabled" in disabled.output


def _empty_registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"version": 1, "contexts": {}}), encoding="utf-8")
    return path


def test_cat_resolves_context_selector_and_around(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    _write_manifest(drive)
    runner = CliRunner()

    result = runner.invoke(
        cli.cli,
        ["cat", f"{drive / 'manifest.yaml'}:alpha.a", "--around", "1"],
        env=_isolated_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "alpha content" in result.output
    assert "authored context" in result.output
    assert "first note" in result.output


def test_cat_selector_honors_registry_flag(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    _write_manifest(drive)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {"version": 1, "contexts": {"demo": {"targetDir": str(drive), "manifest": {"source": "manifest.yaml"}}}}
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        cli.cli,
        ["cat", "demo:alpha.a", "--registry", str(registry_path)],
        env=_isolated_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "alpha content" in result.output


def test_cat_json_carries_selector_structure(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    _write_manifest(drive)
    runner = CliRunner()

    result = runner.invoke(
        cli.cli,
        ["cat", f"{drive / 'manifest.yaml'}:alpha", "--json"],
        env=_isolated_env(tmp_path),
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["selectors"][0]["state"] == "ok"
    assert payload["selectors"][0]["specs"] == [str(drive / "a.md"), str(drive / "b.md")]


def test_cat_selector_failure_is_a_clean_error(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    _write_manifest(drive)
    runner = CliRunner()

    result = runner.invoke(
        cli.cli, ["cat", f"{drive / 'manifest.yaml'}:nosuch"], env=_isolated_env(tmp_path)
    )
    assert result.exit_code != 0
    assert "nosuch" in str(result.output) + str(result.exception)


def test_cat_leaves_non_selector_colon_args_untouched(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    _write_manifest(drive)
    (drive / "plain.md").write_text("hello: world\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        cli.cli, ["cat", str(drive / "plain.md")], env=_isolated_env(tmp_path)
    )
    assert result.exit_code == 0
    assert "hello: world" in result.output


def test_links_cli_reports_registry_coverage(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    _write_manifest(drive)
    runner = CliRunner()

    result = runner.invoke(
        cli.cli,
        ["links", str(drive / "manifest.yaml"), "--registry", str(_empty_registry(tmp_path))],
        env=_isolated_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "coverage" in result.output


def test_status_cli_registry_wide_and_per_context(tmp_path: Path) -> None:
    one = tmp_path / "one"
    one.mkdir()
    (one / "manifest.yaml").write_text(
        "config:\n  name: one\n  context:\n    include-meta: true\n"
        "components:\n  - name: m\n    text: hi\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {"version": 1, "contexts": {"one": {"targetDir": str(one), "manifest": {"source": "manifest.yaml"}}}}
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    registry_wide = runner.invoke(
        cli.cli, ["status", "--registry", str(registry_path)], env=_isolated_env(tmp_path)
    )
    assert registry_wide.exit_code == 0
    assert "registry:" in registry_wide.output
    assert "one" in registry_wide.output

    per_context = runner.invoke(
        cli.cli, ["status", "one", "--registry", str(registry_path)], env=_isolated_env(tmp_path)
    )
    assert per_context.exit_code == 0
    assert "unhydrated" in per_context.output
