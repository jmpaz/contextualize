from __future__ import annotations

import types
from pathlib import Path

import pytest

click_testing = pytest.importorskip("click.testing")
pytest.importorskip("pyperclip")
CliRunner = click_testing.CliRunner


def _plugin_root(tmp_path: Path) -> Path:
    return tmp_path / "home" / ".config" / "contextualize" / "plugins"


def _write_plugin(root: Path, name: str, *, register_auth: bool) -> None:
    repo = root / name
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "plugin.yaml").write_text(
        "\n".join(
            [
                f"name: {name}",
                "module: plugin",
                "api_version: '1'",
                "priority: 500",
                "enabled: true",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        "import click",
        "PLUGIN_API_VERSION = '1'",
        f"PLUGIN_NAME = '{name}'",
        "PLUGIN_PRIORITY = 500",
        "def can_resolve(target, context):",
        "    return False",
        "def resolve(target, context):",
        "    return []",
    ]
    if register_auth:
        lines.extend(
            [
                "def register_auth_command(group):",
                "    @group.command('demo')",
                "    def _auth_demo():",
                "        click.echo('demo auth ok')",
            ]
        )
    (repo / "plugin.py").write_text("\n".join(lines), encoding="utf-8")


def test_auth_placeholder_without_auth_plugins(monkeypatch, tmp_path: Path) -> None:
    from contextualize import cli
    from contextualize.plugins import clear_loaded_plugins_cache

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    clear_loaded_plugins_cache()

    runner = CliRunner()
    help_result = runner.invoke(cli.cli, ["--help"])
    assert help_result.exit_code == 0
    assert " auth " in help_result.output

    auth_result = runner.invoke(cli.cli, ["auth"])
    assert auth_result.exit_code == 0
    assert "No loaded plugins expose authentication handlers." in auth_result.output
    assert "Installed plugins: none" in auth_result.output


def test_auth_placeholder_lists_installed_plugins_and_sources(
    monkeypatch, tmp_path: Path
) -> None:
    from contextualize import cli
    from contextualize.plugins import clear_loaded_plugins_cache

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_plugin(_plugin_root(tmp_path), "arena", register_auth=False)
    clear_loaded_plugins_cache()

    runner = CliRunner()
    auth_result = runner.invoke(cli.cli, ["auth"])
    assert auth_result.exit_code == 0
    assert "No loaded plugins expose authentication handlers." in auth_result.output
    assert "Installed plugins:" in auth_result.output
    assert "- arena: path (" in auth_result.output

    missing_result = runner.invoke(cli.cli, ["auth", "arena"])
    assert missing_result.exit_code != 0
    assert (
        "Plugin 'arena' is installed but does not expose an authentication handler."
        in missing_result.output
    )


def test_external_plugin_registers_auth_command(monkeypatch, tmp_path: Path) -> None:
    from contextualize import cli
    from contextualize.plugins import clear_loaded_plugins_cache

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_plugin(_plugin_root(tmp_path), "demo-auth", register_auth=True)
    clear_loaded_plugins_cache()

    runner = CliRunner()
    auth_result = runner.invoke(cli.cli, ["auth"])
    assert auth_result.exit_code == 0
    assert "Commands:" in auth_result.output
    assert "demo" in auth_result.output

    auth_help = runner.invoke(cli.cli, ["auth", "--help"])
    assert auth_help.exit_code == 0
    assert "demo" in auth_help.output

    auth_command = runner.invoke(cli.cli, ["auth", "demo"])
    assert auth_command.exit_code == 0
    assert "demo auth ok" in auth_command.output


def test_auth_placeholder_renders_package_plugin_source(
    monkeypatch, tmp_path: Path
) -> None:
    from contextualize import cli
    from contextualize.plugins import clear_loaded_plugins_cache
    from contextualize.plugins import loader as plugin_loader

    class _FakeEntrypoint:
        name = "pkg-arena"
        value = "contextualize_plugins.arena:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "pkg-arena"
            plugin.PLUGIN_PRIORITY = 300
            plugin.can_resolve = lambda _target, _context: False
            plugin.resolve = lambda _target, _context: []
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_FakeEntrypoint()]
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    clear_loaded_plugins_cache()

    runner = CliRunner()
    auth_result = runner.invoke(cli.cli, ["auth"])
    assert auth_result.exit_code == 0
    assert (
        "- pkg-arena: package (contextualize_plugins.arena:plugin)"
        in auth_result.output
    )
