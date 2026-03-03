from __future__ import annotations

from pathlib import Path

import pytest

from contextualize.manifest.hydrate import HydrateOverrides, build_hydration_plan_data
from contextualize.plugins import clear_loaded_plugins_cache


def _write_plugin(
    root: Path,
    name: str,
    *,
    can_expr: str,
    resolve_body: str,
) -> None:
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
    (repo / "plugin.py").write_text(
        "\n".join(
            [
                "PLUGIN_API_VERSION = '1'",
                f"PLUGIN_NAME = '{name}'",
                "PLUGIN_PRIORITY = 500",
                "def can_resolve(target, context):",
                f"    return {can_expr}",
                "def resolve(target, context):",
                *[f"    {line}" for line in resolve_body.splitlines()],
            ]
        ),
        encoding="utf-8",
    )


def test_hydrate_manifest_uses_custom_plugin_scheme(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plugins_root = tmp_path / "plugins"
    monkeypatch.setenv("CONTEXTUALIZE_PLUGIN_DIRS", str(plugins_root))
    _write_plugin(
        plugins_root,
        "demo",
        can_expr="target.startswith('demo://')",
        resolve_body=(
            "return [{\n"
            "    'source': target,\n"
            "    'label': 'demo/doc.md',\n"
            "    'content': 'hello from hydrate plugin',\n"
            "    'metadata': {\n"
            "        'context_subpath': 'demo/doc.md',\n"
            "        'source_ref': 'demo',\n"
            "        'source_path': 'doc',\n"
            "    },\n"
            "}]"
        ),
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
    monkeypatch.delenv("CONTEXTUALIZE_PLUGIN_DIRS", raising=False)
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
