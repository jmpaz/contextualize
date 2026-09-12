from __future__ import annotations

import json
import types
from pathlib import Path

from click.testing import CliRunner

from contextualize import cli
from contextualize.plugins import clear_loaded_plugins_cache
from contextualize.plugins import loader as plugin_loader
from contextualize.plugins.resolve import classify_plugin_target, list_plugin_targets


def _entrypoint(*, include_list_targets: bool, plain_list: bool = False) -> object:
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
                items = [
                    {"target": "demo://root/a.md", "label": "a.md"},
                    {
                        "target": "demo://root/b.md",
                        "label": "b.md",
                        "kind": "channel",
                        "traverse": False,
                    },
                ]
                if plain_list:
                    plugin.list_targets = lambda _target, _context: items
                else:
                    plugin.list_targets = lambda _target, _context: {
                        "targets": items,
                        "summary": {"title": "Demo root"},
                        "pagination": {"returned": 2, "hasMore": False},
                        "metadata": {"shape": "envelope"},
                        "capabilities": {"materialize": False},
                    }
            return plugin

    return _DemoEntrypoint()


def _failing_entrypoint() -> object:
    class _FailingEntrypoint:
        name = "demo-list"
        value = "contextualize_plugins.demo_list:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "demo-list"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith("demo://")
            plugin.resolve = lambda _target, _context: []

            def list_targets(_target, _context):
                raise RuntimeError("auth missing")

            plugin.list_targets = list_targets
            return plugin

    return _FailingEntrypoint()


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


def test_plugin_list_targets_requires_envelope(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [_entrypoint(include_list_targets=True, plain_list=True)],
    )
    clear_loaded_plugins_cache()

    result = list_plugin_targets("demo://root")

    assert result.matched is True
    assert result.supported is False
    assert result.items == ()


def test_plugin_list_targets_accepts_envelope(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [_entrypoint(include_list_targets=True)],
    )
    clear_loaded_plugins_cache()

    result = list_plugin_targets("demo://root")

    assert [item["target"] for item in result.items] == [
        "demo://root/a.md",
        "demo://root/b.md",
    ]
    assert result.items[1]["kind"] == "channel"
    assert result.items[1]["traverse"] is False
    assert result.summary == {"title": "Demo root"}
    assert result.pagination == {"returned": 2, "hasMore": False}
    assert result.metadata == {"shape": "envelope"}
    assert result.capabilities == {"materialize": False}


def test_plugin_list_targets_pages_envelope(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    class _DemoEntrypoint:
        name = "demo-list"
        value = "contextualize_plugins.demo_list:plugin"

        def load(self):
            items = [
                {"target": f"demo://root/{index}", "label": f"{index}.md"}
                for index in range(5)
            ]
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "demo-list"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith("demo://")
            plugin.resolve = lambda _target, _context: []
            plugin.list_targets = lambda _target, _context: {
                "targets": items,
                "summary": {"sampledItems": items},
                "pagination": {"returned": 5, "totalCount": 5, "hasMore": False},
            }
            return plugin

    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [_DemoEntrypoint()],
    )
    clear_loaded_plugins_cache()

    result = list_plugin_targets(
        "demo://root",
        context_options={"list_limit": 2, "list_offset": 2},
    )

    assert [item["target"] for item in result.items] == [
        "demo://root/2",
        "demo://root/3",
    ]
    assert result.pagination == {
        "returned": 2,
        "totalCount": 5,
        "hasMore": True,
        "offset": 2,
        "limit": 2,
        "nextOffset": 4,
    }
    assert result.summary == {"sampledItems": list(result.items)}


def test_plugin_list_targets_preserves_failure_reason(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [_failing_entrypoint()],
    )
    clear_loaded_plugins_cache()

    result = list_plugin_targets("demo://root")

    assert result.matched is True
    assert result.supported is False
    assert result.plugin_name == "demo-list"
    assert result.error == "RuntimeError: auth missing"


def test_classify_plugin_target_passes_inspection_context(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    captured = {}

    class _DemoEntrypoint:
        name = "demo-list"
        value = "contextualize_plugins.demo_list:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "demo-list"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith("demo://")
            plugin.resolve = lambda _target, _context: []

            def classify_target(_target, context):
                captured.update(context)
                return {
                    "provider": "demo-list",
                    "kind": "root",
                    "relations": [
                        {
                            "target": "demo://parent",
                            "label": "Parent",
                            "kind": "container",
                            "metadata": {"relation": "contained_by"},
                        }
                    ],
                    "metadata": {"shape": "descriptor"},
                    "capabilities": {"listTargets": True},
                }

            plugin.classify_target = classify_target
            return plugin

    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [_DemoEntrypoint()],
    )
    clear_loaded_plugins_cache()

    result = classify_plugin_target(
        "demo://root",
        overrides={"demo-list": {"mode": "inspect"}},
        use_cache=False,
        refresh_cache=True,
    )

    assert result is not None
    assert result["relations"][0]["target"] == "demo://parent"
    assert result["metadata"] == {"shape": "descriptor"}
    assert result["capabilities"] == {"listTargets": True}
    assert captured["use_cache"] is False
    assert captured["refresh_cache"] is True
    assert captured["overrides"] == {"demo-list": {"mode": "inspect"}}


def test_cat_list_json_preserves_each_provider_envelope(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [_entrypoint(include_list_targets=True)],
    )
    clear_loaded_plugins_cache()
    result = CliRunner().invoke(
        cli.cli, ["cat", "--list", "--json", "demo://root", "demo://second"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["content"] == "- `demo://root/a.md`\n- `demo://root/b.md`"
    assert [item["selector"] for item in payload["selectors"]] == ["demo://root", "demo://second"]
    assert [listing["source"] for listing in payload["listings"]] == [
        "demo://root",
        "demo://second",
    ]
    for listing in payload["listings"]:
        assert listing["provider"] == "demo-list"
        assert listing["targets"][1]["traverse"] is False
        assert listing["targets"][1]["kind"] == "channel"
        assert listing["summary"] == {"title": "Demo root"}
        assert listing["pagination"] == {"returned": 2, "hasMore": False}
        assert listing["metadata"] == {"shape": "envelope"}
        assert listing["capabilities"] == {"materialize": False}


def test_cat_list_json_preserves_empty_page_continuation(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    class EntryPoint:
        name = "empty-page"
        value = "empty_page:plugin"

        def load(self):
            return types.SimpleNamespace(
                PLUGIN_API_VERSION="1",
                PLUGIN_NAME="empty-page",
                PLUGIN_PRIORITY=500,
                can_resolve=lambda target, context: target.startswith("empty://"),
                resolve=lambda target, context: [],
                list_targets=lambda target, context: {
                    "targets": [],
                    "summary": {"scanned": 25, "next_target": "empty://next"},
                    "pagination": {
                        "returned": 0,
                        "hasMore": True,
                        "nextTarget": "empty://next",
                    },
                    "metadata": {"scan_limit_reached": True},
                    "capabilities": {"listTargets": True},
                },
            )

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [EntryPoint()]
    )
    clear_loaded_plugins_cache()
    result = CliRunner().invoke(cli.cli, ["cat", "--list", "--json", "empty://root"])
    assert result.exit_code == 0, result.output
    listing = json.loads(result.output)["listings"][0]
    assert listing["targets"] == []
    assert listing["pagination"]["hasMore"] is True
    assert listing["summary"]["next_target"] == "empty://next"
    assert listing["metadata"]["scan_limit_reached"] is True
