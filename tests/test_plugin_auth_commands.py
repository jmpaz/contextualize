from __future__ import annotations

import types
from pathlib import Path

import pytest

click_testing = pytest.importorskip("click.testing")
pytest.importorskip("pyperclip")
CliRunner = click_testing.CliRunner


def _entrypoint(
    *,
    entrypoint_name: str,
    entrypoint_value: str,
    plugin_name: str,
    register_auth: bool,
) -> object:
    class _FakeEntrypoint:
        name = entrypoint_name
        value = entrypoint_value

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = plugin_name
            plugin.PLUGIN_PRIORITY = 300
            plugin.can_resolve = lambda _target, _context: False
            plugin.resolve = lambda _target, _context: []
            if register_auth:
                import click

                def register_auth_command(group):
                    @group.command("demo")
                    def _auth_demo() -> None:
                        click.echo("demo auth ok")

                plugin.register_auth_command = register_auth_command
            return plugin

    return _FakeEntrypoint()


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
    from contextualize.plugins import loader as plugin_loader

    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [
            _entrypoint(
                entrypoint_name="pkg-arena",
                entrypoint_value="contextualize_plugins.arena:plugin",
                plugin_name="arena",
                register_auth=False,
            )
        ],
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    clear_loaded_plugins_cache()

    runner = CliRunner()
    auth_result = runner.invoke(cli.cli, ["auth"])
    assert auth_result.exit_code == 0
    assert "No loaded plugins expose authentication handlers." in auth_result.output
    assert "Installed plugins:" in auth_result.output
    assert "- arena  contextualize_plugins.arena:plugin" in auth_result.output

    missing_result = runner.invoke(cli.cli, ["auth", "arena"])
    assert missing_result.exit_code != 0
    assert (
        "Plugin 'arena' is installed but does not expose an authentication handler."
        in missing_result.output
    )


def test_external_plugin_registers_auth_command(monkeypatch, tmp_path: Path) -> None:
    from contextualize import cli
    from contextualize.plugins import clear_loaded_plugins_cache
    from contextualize.plugins import loader as plugin_loader

    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [
            _entrypoint(
                entrypoint_name="demo-auth",
                entrypoint_value="contextualize_plugins.demo_auth:plugin",
                plugin_name="demo-auth",
                register_auth=True,
            )
        ],
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
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


def test_auth_placeholder_renders_plugin_source(monkeypatch, tmp_path: Path) -> None:
    from contextualize import cli
    from contextualize.plugins import clear_loaded_plugins_cache
    from contextualize.plugins import loader as plugin_loader

    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [
            _entrypoint(
                entrypoint_name="pkg-arena",
                entrypoint_value="contextualize_plugins.arena:plugin",
                plugin_name="pkg-arena",
                register_auth=False,
            )
        ],
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    clear_loaded_plugins_cache()

    runner = CliRunner()
    auth_result = runner.invoke(cli.cli, ["auth"])
    assert auth_result.exit_code == 0
    assert "- pkg-arena  contextualize_plugins.arena:plugin" in auth_result.output


def test_plugins_command_without_plugins(monkeypatch, tmp_path: Path) -> None:
    from contextualize import cli
    from contextualize.plugins import clear_loaded_plugins_cache

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    clear_loaded_plugins_cache()

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["plugins"])
    assert result.exit_code == 0
    assert "Installed plugins: none" in result.output


def test_plugins_command_lists_installed_plugins_and_sources(
    monkeypatch, tmp_path: Path
) -> None:
    from contextualize import cli
    from contextualize.plugins import clear_loaded_plugins_cache
    from contextualize.plugins import loader as plugin_loader

    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [
            _entrypoint(
                entrypoint_name="pkg-arena",
                entrypoint_value="contextualize_plugins.arena:plugin",
                plugin_name="arena",
                register_auth=False,
            ),
            _entrypoint(
                entrypoint_name="pkg-youtube",
                entrypoint_value="contextualize_plugins.youtube:plugin",
                plugin_name="youtube",
                register_auth=False,
            ),
        ],
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    clear_loaded_plugins_cache()

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["plugins"])
    assert result.exit_code == 0
    assert "Installed plugins:" in result.output
    assert "- arena    contextualize_plugins.arena:plugin" in result.output
    assert "- youtube  contextualize_plugins.youtube:plugin" in result.output


def test_plugins_command_missing_plugin_lists_installed_plugins(
    monkeypatch, tmp_path: Path
) -> None:
    from contextualize import cli
    from contextualize.plugins import clear_loaded_plugins_cache
    from contextualize.plugins import loader as plugin_loader

    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [
            _entrypoint(
                entrypoint_name="pkg-arena",
                entrypoint_value="contextualize_plugins.arena:plugin",
                plugin_name="arena",
                register_auth=False,
            )
        ],
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    clear_loaded_plugins_cache()

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["plugins", "missing"])
    assert result.exit_code != 0
    assert "Plugin 'missing' is not installed." in result.output
    assert "Installed plugins:" in result.output
    assert "- arena  contextualize_plugins.arena:plugin" in result.output


def test_plugins_command_renders_single_plugin(monkeypatch, tmp_path: Path) -> None:
    from contextualize import cli
    from contextualize.plugins import clear_loaded_plugins_cache
    from contextualize.plugins import loader as plugin_loader

    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [
            _entrypoint(
                entrypoint_name="pkg-arena",
                entrypoint_value="contextualize_plugins.arena:plugin",
                plugin_name="arena",
                register_auth=False,
            )
        ],
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    clear_loaded_plugins_cache()

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["plugins", "arena"])
    assert result.exit_code == 0
    assert result.output.strip() == "- arena  contextualize_plugins.arena:plugin"
