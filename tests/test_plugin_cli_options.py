from __future__ import annotations

import types
from pathlib import Path

import click
from click.testing import CliRunner

from contextualize import cli
from contextualize.plugins import clear_loaded_plugins_cache
from contextualize.plugins import loader as plugin_loader


def test_cat_passes_plugin_cli_overrides_into_file_reference_creation(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    class _DemoEntrypoint:
        name = "demo-cli"
        value = "contextualize_plugins.demo_cli:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "demo-cli"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda _target, _context: False
            plugin.resolve = lambda _target, _context: []

            def register_cli_options(command_name, command):
                if command_name != "cat":
                    return
                command.params.append(
                    click.Option(
                        ["--demo-text"],
                        default=None,
                        help="demo plugin option",
                    )
                )

            def collect_cli_overrides(command_name, params):
                if command_name != "cat":
                    return None
                value = params.get("demo_text")
                if not value:
                    return None
                return {"value": value}

            plugin.register_cli_options = register_cli_options
            plugin.collect_cli_overrides = collect_cli_overrides
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_DemoEntrypoint()]
    )
    clear_loaded_plugins_cache()

    note_path = tmp_path / "note.txt"
    note_path.write_text("hello", encoding="utf-8")

    captured: dict[str, object] = {}

    def _create_file_references(*args, **kwargs):
        captured["plugin_overrides"] = kwargs.get("plugin_overrides")
        return {
            "refs": [],
            "concatenated": "",
            "ignored_files": [],
            "ignored_folders": {},
        }

    monkeypatch.setattr(
        "contextualize.references.create_file_references", _create_file_references
    )

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cat", str(note_path), "--demo-text", "hello"])

    assert result.exit_code == 0
    assert captured["plugin_overrides"] == {"demo-cli": {"value": "hello"}}


def test_cat_help_groups_plugin_options_into_plugin_sections(monkeypatch) -> None:
    class _DemoEntrypoint:
        name = "demo-cli"
        value = "contextualize_plugins.demo_cli:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "demo-cli"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda _target, _context: False
            plugin.resolve = lambda _target, _context: []

            def register_cli_options(command_name, command):
                if command_name != "cat":
                    return
                command.params.append(
                    click.Option(
                        ["--demo-text"],
                        default=None,
                        help="demo plugin option",
                    )
                )

            plugin.register_cli_options = register_cli_options
            plugin.collect_cli_overrides = lambda *_args, **_kwargs: None
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_DemoEntrypoint()]
    )
    clear_loaded_plugins_cache()

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cat", "--help"])

    assert result.exit_code == 0
    assert "Options:\n" in result.output
    assert "plugin: demo-cli\n" in result.output
    assert "--demo-text" in result.output


def test_payload_passes_plugin_cli_overrides_into_manifest_render(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    class _DemoEntrypoint:
        name = "demo-cli"
        value = "contextualize_plugins.demo_cli:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "demo-cli"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda _target, _context: False
            plugin.resolve = lambda _target, _context: []

            def register_cli_options(command_name, command):
                if command_name != "payload":
                    return
                command.params.append(
                    click.Option(
                        ["--demo-text"],
                        default=None,
                        help="demo plugin option",
                    )
                )

            def collect_cli_overrides(command_name, params):
                if command_name != "payload":
                    return None
                value = params.get("demo_text")
                if not value:
                    return None
                return {"value": value}

            plugin.register_cli_options = register_cli_options
            plugin.collect_cli_overrides = collect_cli_overrides
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_DemoEntrypoint()]
    )
    clear_loaded_plugins_cache()

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("config: {}\ncomponents: []\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def _render_manifest(*args, **kwargs):
        captured["plugin_overrides"] = kwargs.get("plugin_overrides")
        return types.SimpleNamespace(
            payload="payload",
            input_refs=[],
            trace_items=[],
            base_dir=str(tmp_path),
            skipped_paths=[],
            skip_impact={},
        )

    monkeypatch.setattr(
        "contextualize.manifest.payload.render_manifest", _render_manifest
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.cli, ["payload", str(manifest_path), "--demo-text", "hello"]
    )

    assert result.exit_code == 0
    assert result.output == "payload\n"
    assert captured["plugin_overrides"] == {"demo-cli": {"value": "hello"}}
