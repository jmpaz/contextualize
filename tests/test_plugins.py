from __future__ import annotations

import types
from pathlib import Path

import pytest

from contextualize.plugins import clear_loaded_plugins_cache
from contextualize.plugins import loader as plugin_loader
from contextualize.references import create_file_references


def _reset_plugin_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    clear_loaded_plugins_cache()


def _entrypoint(
    *,
    entrypoint_name: str,
    entrypoint_value: str,
    plugin_name: str,
    priority: int,
    target_prefix: str,
    content: str,
) -> object:
    class _FakeEntrypoint:
        name = entrypoint_name
        value = entrypoint_value

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = plugin_name
            plugin.PLUGIN_PRIORITY = priority

            def can_resolve(target: str, _context: dict[str, object]) -> bool:
                return target.startswith(target_prefix)

            def resolve(
                target: str, _context: dict[str, object]
            ) -> list[dict[str, str]]:
                return [
                    {
                        "source": target,
                        "label": f"{plugin_name}/item.md",
                        "content": content,
                    }
                ]

            plugin.can_resolve = can_resolve
            plugin.resolve = resolve
            return plugin

    return _FakeEntrypoint()


def test_unresolved_external_scheme_fails_without_plugin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    clear_loaded_plugins_cache()
    with pytest.raises(ValueError, match="No plugin could resolve external target"):
        create_file_references(["unknown://abc"], format="raw")


def test_entrypoint_plugin_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [
            _entrypoint(
                entrypoint_name="entry-demo",
                entrypoint_value="contextualize_plugins.entry_demo:plugin",
                plugin_name="entry-demo",
                priority=300,
                target_prefix="entry://",
                content="entry source",
            )
        ],
    )

    clear_loaded_plugins_cache()
    result = create_file_references(["entry://abc"], format="raw")
    assert result["concatenated"] == "entry source"


def test_plugin_priority_prefers_higher_priority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [
            _entrypoint(
                entrypoint_name="low",
                entrypoint_value="contextualize_plugins.low:plugin",
                plugin_name="low-priority",
                priority=100,
                target_prefix="prio://",
                content="low",
            ),
            _entrypoint(
                entrypoint_name="high",
                entrypoint_value="contextualize_plugins.high:plugin",
                plugin_name="high-priority",
                priority=900,
                target_prefix="prio://",
                content="high",
            ),
        ],
    )

    clear_loaded_plugins_cache()
    result = create_file_references(["prio://x"], format="raw")
    assert result["concatenated"] == "high"


def test_duplicate_plugin_name_uses_higher_priority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [
            _entrypoint(
                entrypoint_name="same-low",
                entrypoint_value="contextualize_plugins.same_low:plugin",
                plugin_name="same",
                priority=100,
                target_prefix="same://",
                content="low",
            ),
            _entrypoint(
                entrypoint_name="same-high",
                entrypoint_value="contextualize_plugins.same_high:plugin",
                plugin_name="same",
                priority=900,
                target_prefix="same://",
                content="high",
            ),
        ],
    )

    clear_loaded_plugins_cache()
    loaded = plugin_loader.get_loaded_plugins()
    same = [plugin for plugin in loaded if plugin.name == "same"]
    assert len(same) == 1
    assert same[0].origin == "entrypoint:contextualize_plugins.same_high:plugin"

    result = create_file_references(["same://abc"], format="raw")
    assert result["concatenated"] == "high"


def test_duplicate_plugin_name_tie_uses_lexicographic_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [
            _entrypoint(
                entrypoint_name="same-b",
                entrypoint_value="contextualize_plugins.same_b:plugin",
                plugin_name="same",
                priority=500,
                target_prefix="same://",
                content="from b",
            ),
            _entrypoint(
                entrypoint_name="same-a",
                entrypoint_value="contextualize_plugins.same_a:plugin",
                plugin_name="same",
                priority=500,
                target_prefix="same://",
                content="from a",
            ),
        ],
    )

    clear_loaded_plugins_cache()
    loaded = plugin_loader.get_loaded_plugins()
    same = [plugin for plugin in loaded if plugin.name == "same"]
    assert len(same) == 1
    assert same[0].origin == "entrypoint:contextualize_plugins.same_a:plugin"

    result = create_file_references(["same://abc"], format="raw")
    assert result["concatenated"] == "from a"


def test_injected_http_target_uses_plugin_and_inherits_cache_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    class _DemoEntrypoint:
        name = "demo-http"
        value = "contextualize_plugins.demo_http:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "demo-http"
            plugin.PLUGIN_PRIORITY = 500

            def can_resolve(target: str, _context: dict[str, object]) -> bool:
                return target.startswith("https://demo.test/")

            def resolve(
                target: str, context: dict[str, object]
            ) -> list[dict[str, str]]:
                refresh = bool(context.get("refresh_cache"))
                use_cache = bool(context.get("use_cache", True))
                return [
                    {
                        "source": target,
                        "label": "demo-http/item.md",
                        "content": (
                            f"resolved via plugin refresh={refresh} use_cache={use_cache}"
                        ),
                    }
                ]

            plugin.can_resolve = can_resolve
            plugin.resolve = resolve
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_DemoEntrypoint()]
    )
    clear_loaded_plugins_cache()

    note_path = tmp_path / "note.md"
    note_path.write_text(
        "{cx::wrap=xml::https://demo.test/thread}",
        encoding="utf-8",
    )

    result = create_file_references(
        [str(note_path)],
        format="raw",
        inject=True,
        use_cache=False,
        refresh_cache=True,
    )

    assert result["concatenated"] == (
        "<paste>\nresolved via plugin refresh=True use_cache=False\n</paste>"
    )
