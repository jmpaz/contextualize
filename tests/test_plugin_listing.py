from __future__ import annotations

import types
from pathlib import Path

from click.testing import CliRunner

from contextualize import cli
from contextualize.plugins import clear_loaded_plugins_cache
from contextualize.plugins import loader as plugin_loader


def _entrypoint(*, include_list_targets: bool) -> object:
    class _DemoEntrypoint:
        name = "demo-list"
        value = "contextualize_plugins.demo_list:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "demo-list"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith("demo://")
            plugin.resolve = lambda _target, _context: [
                {
                    "source": "demo://resolved",
                    "label": "demo/resolved.md",
                    "content": "resolved",
                }
            ]
            if include_list_targets:
                plugin.list_targets = lambda _target, _context: [
                    {"target": "demo://root/a.md", "label": "a.md"},
                    {"target": "demo://root/b.md", "label": "b.md"},
                ]
            return plugin

    return _DemoEntrypoint()


def test_cat_list_uses_plugin_list_targets(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [_entrypoint(include_list_targets=True)],
    )
    clear_loaded_plugins_cache()

    result = CliRunner().invoke(cli.cli, ["cat", "--list", "demo://root"])

    assert result.exit_code == 0
    assert result.output == "- `demo://root/a.md`\n- `demo://root/b.md`\n"


def test_cat_list_reports_plugins_without_listing_hook(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [_entrypoint(include_list_targets=False)],
    )
    clear_loaded_plugins_cache()

    result = CliRunner().invoke(cli.cli, ["cat", "--list", "demo://root"])

    assert result.exit_code != 0
    assert (
        "Plugin 'demo-list' does not support listing targets: demo://root"
        in result.output
    )
