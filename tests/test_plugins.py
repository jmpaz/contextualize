from __future__ import annotations

import types
from pathlib import Path

import pytest

from contextualize.plugins import clear_loaded_plugins_cache
from contextualize.plugins import loader as plugin_loader
from contextualize.references import create_file_references


def _config_plugin_root(tmp_path: Path) -> Path:
    return tmp_path / "home" / ".config" / "contextualize" / "plugins"


def _write_plugin(
    root: Path,
    repo_name: str,
    *,
    plugin_name: str | None = None,
    priority: int,
    can_expr: str,
    resolve_body: str,
) -> None:
    name = plugin_name or repo_name
    repo = root / repo_name
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "plugin.yaml").write_text(
        "\n".join(
            [
                f"name: {name}",
                "module: plugin",
                "api_version: '1'",
                f"priority: {priority}",
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
                f"PLUGIN_PRIORITY = {priority}",
                "def can_resolve(target, context):",
                f"    return {can_expr}",
                "def resolve(target, context):",
                *[f"    {line}" for line in resolve_body.splitlines()],
            ]
        ),
        encoding="utf-8",
    )


def _reset_plugin_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CONTEXTUALIZE_PLUGIN_DIRS", raising=False)
    clear_loaded_plugins_cache()


def test_plugins_load_from_env_plugin_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    env_root = tmp_path / "plugins"
    _write_plugin(
        env_root,
        "demo",
        priority=500,
        can_expr="target.startswith('demo://')",
        resolve_body=(
            "return [{\n"
            "    'source': target,\n"
            "    'label': 'demo/path.txt',\n"
            "    'content': 'hello from plugin',\n"
            "}]"
        ),
    )
    monkeypatch.setenv("CONTEXTUALIZE_PLUGIN_DIRS", str(env_root))

    clear_loaded_plugins_cache()
    result = create_file_references(["demo://abc"], format="raw")
    assert result["concatenated"] == "hello from plugin"


def test_plugins_load_from_default_config_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    config_root = _config_plugin_root(tmp_path)
    _write_plugin(
        config_root,
        "config-demo",
        priority=500,
        can_expr="target.startswith('config://')",
        resolve_body=(
            "return [{\n"
            "    'source': target,\n"
            "    'label': 'config/path.txt',\n"
            "    'content': 'hello from config plugin',\n"
            "}]"
        ),
    )

    clear_loaded_plugins_cache()
    result = create_file_references(["config://abc"], format="raw")
    assert result["concatenated"] == "hello from config plugin"


def test_plugins_load_from_default_and_env_plugin_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    config_root = _config_plugin_root(tmp_path)
    env_root = tmp_path / "plugins"
    _write_plugin(
        config_root,
        "config-demo",
        priority=500,
        can_expr="target.startswith('config://')",
        resolve_body=(
            "return [{\n"
            "    'source': target,\n"
            "    'label': 'config/path.txt',\n"
            "    'content': 'hello from config plugin',\n"
            "}]"
        ),
    )
    _write_plugin(
        env_root,
        "env-demo",
        priority=500,
        can_expr="target.startswith('env://')",
        resolve_body=(
            "return [{\n"
            "    'source': target,\n"
            "    'label': 'env/path.txt',\n"
            "    'content': 'hello from env plugin',\n"
            "}]"
        ),
    )
    monkeypatch.setenv("CONTEXTUALIZE_PLUGIN_DIRS", str(env_root))

    clear_loaded_plugins_cache()
    config_result = create_file_references(["config://abc"], format="raw")
    assert config_result["concatenated"] == "hello from config plugin"

    env_result = create_file_references(["env://abc"], format="raw")
    assert env_result["concatenated"] == "hello from env plugin"


def test_unresolved_external_scheme_fails_without_plugin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    clear_loaded_plugins_cache()
    with pytest.raises(ValueError, match="No plugin could resolve external target"):
        create_file_references(["unknown://abc"], format="raw")


def test_plugin_priority_prefers_higher_priority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    env_root = tmp_path / "plugins"
    monkeypatch.setenv("CONTEXTUALIZE_PLUGIN_DIRS", str(env_root))
    _write_plugin(
        env_root,
        "high-priority",
        priority=900,
        can_expr="target.startswith('prio://')",
        resolve_body=(
            "return [{\n"
            "    'source': target,\n"
            "    'label': 'high/item.md',\n"
            "    'content': 'high',\n"
            "}]"
        ),
    )
    _write_plugin(
        env_root,
        "low-priority",
        priority=100,
        can_expr="target.startswith('prio://')",
        resolve_body=(
            "return [{\n"
            "    'source': target,\n"
            "    'label': 'low/item.md',\n"
            "    'content': 'low',\n"
            "}]"
        ),
    )

    clear_loaded_plugins_cache()
    result = create_file_references(["prio://x"], format="raw")
    assert result["concatenated"] == "high"


def test_duplicate_plugin_name_uses_higher_priority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    config_root = _config_plugin_root(tmp_path)
    _write_plugin(
        config_root,
        "same",
        priority=100,
        can_expr="target.startswith('same://')",
        resolve_body=(
            "return [{\n"
            "    'source': target,\n"
            "    'label': 'root/item.md',\n"
            "    'content': 'root',\n"
            "}]"
        ),
    )
    _write_plugin(
        config_root,
        "same-high",
        plugin_name="same",
        priority=900,
        can_expr="target.startswith('same://')",
        resolve_body=(
            "return [{\n"
            "    'source': target,\n"
            "    'label': 'nested/item.md',\n"
            "    'content': 'nested',\n"
            "}]"
        ),
    )

    clear_loaded_plugins_cache()
    loaded = plugin_loader.get_loaded_plugins()
    same = [plugin for plugin in loaded if plugin.name == "same"]
    assert len(same) == 1
    assert "same-high/plugin.yaml" in same[0].origin

    result = create_file_references(["same://abc"], format="raw")
    assert result["concatenated"] == "nested"


def test_entrypoint_plugin_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    class _FakeEntrypoint:
        name = "entry-demo"
        value = "contextualize_plugins.entry_demo:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "entry-demo"
            plugin.PLUGIN_PRIORITY = 300

            def can_resolve(target: str, _context: dict[str, object]) -> bool:
                return target.startswith("entry://")

            def resolve(
                target: str, _context: dict[str, object]
            ) -> list[dict[str, str]]:
                return [
                    {
                        "source": target,
                        "label": "entry/item.md",
                        "content": "entry source",
                    }
                ]

            plugin.can_resolve = can_resolve
            plugin.resolve = resolve
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_FakeEntrypoint()]
    )

    clear_loaded_plugins_cache()
    result = create_file_references(["entry://abc"], format="raw")
    assert result["concatenated"] == "entry source"


def test_package_style_plugin_supports_relative_imports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    env_root = tmp_path / "plugins"
    repo = env_root / "package-demo"
    package_root = repo / "pkg"
    package_root.mkdir(parents=True, exist_ok=True)
    (repo / "plugin.yaml").write_text(
        "\n".join(
            [
                "name: package-demo",
                "module: pkg.plugin",
                "api_version: '1'",
                "priority: 500",
                "enabled: true",
            ]
        ),
        encoding="utf-8",
    )
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "helpers.py").write_text(
        "\n".join(
            [
                "def build_documents(target):",
                "    return [{",
                "        'source': target,",
                "        'label': 'pkg/path.txt',",
                "        'content': 'hello from package plugin',",
                "    }]",
            ]
        ),
        encoding="utf-8",
    )
    (package_root / "plugin.py").write_text(
        "\n".join(
            [
                "from .helpers import build_documents",
                "",
                "PLUGIN_API_VERSION = '1'",
                "PLUGIN_NAME = 'package-demo'",
                "PLUGIN_PRIORITY = 500",
                "",
                "def can_resolve(target, context):",
                "    return target.startswith('pkg://')",
                "",
                "def resolve(target, context):",
                "    return build_documents(target)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONTEXTUALIZE_PLUGIN_DIRS", str(env_root))

    clear_loaded_plugins_cache()
    result = create_file_references(["pkg://abc"], format="raw")
    assert result["concatenated"] == "hello from package plugin"


def test_path_plugins_prefer_over_entrypoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    config_root = _config_plugin_root(tmp_path)
    _write_plugin(
        config_root,
        "same-provider",
        priority=100,
        can_expr="target.startswith('same://')",
        resolve_body=(
            "return [{\n"
            "    'source': target,\n"
            "    'label': 'config/item.md',\n"
            "    'content': 'config source',\n"
            "}]"
        ),
    )

    class _SameEntrypoint:
        name = "same-provider"
        value = "contextualize_plugins.same:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "same-provider"
            plugin.PLUGIN_PRIORITY = 999

            def can_resolve(target: str, _context: dict[str, object]) -> bool:
                return target.startswith("same://")

            def resolve(
                target: str, _context: dict[str, object]
            ) -> list[dict[str, str]]:
                return [
                    {
                        "source": target,
                        "label": "entry/item.md",
                        "content": "entry source",
                    }
                ]

            plugin.can_resolve = can_resolve
            plugin.resolve = resolve
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_SameEntrypoint()]
    )

    clear_loaded_plugins_cache()
    loaded = plugin_loader.get_loaded_plugins()
    same = [plugin for plugin in loaded if plugin.name == "same-provider"]
    assert len(same) == 1
    assert "same-provider/plugin.yaml" in same[0].origin

    result = create_file_references(["same://abc"], format="raw")
    assert result["concatenated"] == "config source"
