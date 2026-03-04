from __future__ import annotations

import types
from pathlib import Path

import pytest

from contextualize.manifest.hydrate import HydrateOverrides, build_hydration_plan_data
from contextualize.plugins import clear_loaded_plugins_cache
from contextualize.plugins import loader as plugin_loader


def test_hydrate_manifest_uses_custom_plugin_scheme(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    class _DemoEntrypoint:
        name = "demo"
        value = "contextualize_plugins.demo:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "demo"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith("demo://")
            plugin.resolve = lambda target, _context: [
                {
                    "source": target,
                    "label": "demo/doc.md",
                    "content": "hello from hydrate plugin",
                    "metadata": {
                        "context_subpath": "demo/doc.md",
                        "source_ref": "demo",
                        "source_path": "doc",
                    },
                }
            ]
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_DemoEntrypoint()]
    )
    clear_loaded_plugins_cache()

    context_dir = tmp_path / "ctx"
    plan = build_hydration_plan_data(
        {
            "config": {"context": {"dir": str(context_dir), "include-meta": False}},
            "components": [{"name": "main", "files": ["demo://abc"]}],
        },
        manifest_cwd=str(tmp_path),
        overrides=HydrateOverrides(),
        cwd=str(tmp_path),
    )

    rel_paths = {
        path.relative_to(context_dir).as_posix() for path, _ in plan.files_to_write
    }
    assert "demo/doc.md" in rel_paths
    assert any(
        content == "hello from hydrate plugin" for _, content in plan.files_to_write
    )


def test_hydrate_manifest_fails_for_unresolved_external_scheme(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    clear_loaded_plugins_cache()

    with pytest.raises(ValueError, match="No plugin could resolve external target"):
        build_hydration_plan_data(
            {
                "config": {
                    "context": {"dir": str(tmp_path / "ctx"), "include-meta": False}
                },
                "components": [{"name": "main", "files": ["unknown://abc"]}],
            },
            manifest_cwd=str(tmp_path),
            overrides=HydrateOverrides(),
            cwd=str(tmp_path),
        )
