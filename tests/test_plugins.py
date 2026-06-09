from __future__ import annotations

import types
from pathlib import Path

import pytest

from contextualize.plugins import clear_loaded_plugins_cache
from contextualize.plugins import loader as plugin_loader
from contextualize.references import create_file_references
from contextualize.references import factory as reference_factory


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


def test_embedded_target_materialization_uses_normal_file_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    class _EmbeddedEntrypoint:
        name = "embedded"
        value = "contextualize_plugins.embedded:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "embedded"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith(
                ("root://", "asset://")
            )

            def resolve(target: str, _context: dict[str, object]) -> list[dict]:
                if target.startswith("root://"):
                    raise AssertionError("parent should not be resolved")
                return []

            plugin.resolve = resolve
            plugin.list_targets = lambda _target, _context: {
                "targets": [
                    {
                        "target": "asset://child",
                        "label": "child.md",
                        "kind": "attachment:text",
                    }
                ]
            }
            plugin.materialize = lambda _target, _context: [
                {
                    "source": "asset://child",
                    "label": "child.md",
                    "filename": "child.md",
                    "content": b"child content",
                    "content_type": "text/markdown",
                }
            ]
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_EmbeddedEntrypoint()]
    )
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["root://thread"],
        format="raw",
        target_depth=1,
        include_parent=False,
    )

    assert result["concatenated"] == "child content"


def test_embedded_text_only_materialized_audio_is_transcribed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    transcribed: list[str] = []

    class _EmbeddedEntrypoint:
        name = "embedded"
        value = "contextualize_plugins.embedded:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "embedded"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith(
                ("root://", "asset://")
            )
            plugin.resolve = lambda _target, _context: []
            plugin.list_targets = lambda _target, _context: {
                "targets": [{"target": "asset://child", "label": "child.mp3"}]
            }
            plugin.materialize = lambda _target, _context: [
                {
                    "source": "asset://child",
                    "label": "child.mp3",
                    "filename": "child.mp3",
                    "content": b"\xffaudio",
                    "content_type": "audio/mpeg",
                }
            ]
            return plugin

    def _transcribe(path: str | Path, **_kwargs) -> str:
        transcribed.append(Path(path).name)
        return "audio transcript"

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_EmbeddedEntrypoint()]
    )
    monkeypatch.setattr(
        "contextualize.references.file.transcribe_media_file", _transcribe
    )
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["root://thread"],
        format="raw",
        target_depth=1,
        include_parent=False,
        text_only=True,
    )

    captured = capsys.readouterr()
    assert result["concatenated"] == "audio transcript"
    assert transcribed == ["child.mp3"]
    assert captured.err == ""


def test_embedded_text_only_materialized_unknown_binary_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    class _EmbeddedEntrypoint:
        name = "embedded"
        value = "contextualize_plugins.embedded:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "embedded"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith(
                ("root://", "asset://")
            )
            plugin.resolve = lambda _target, _context: []
            plugin.list_targets = lambda _target, _context: {
                "targets": [{"target": "asset://child", "label": "child.bin"}]
            }
            plugin.materialize = lambda _target, _context: [
                {
                    "source": "asset://child",
                    "label": "child.bin",
                    "filename": "child.bin",
                    "content": b"\xff\x00binary",
                    "content_type": "application/octet-stream",
                }
            ]
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_EmbeddedEntrypoint()]
    )
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["root://thread"],
        format="raw",
        target_depth=1,
        include_parent=False,
        text_only=True,
    )

    captured = capsys.readouterr()
    assert result["refs"] == []
    assert captured.err == ""


def test_embedded_target_resolves_plugin_claimed_child_without_materialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    class _EmbeddedEntrypoint:
        name = "embedded"
        value = "contextualize_plugins.embedded:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "embedded"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith(
                ("root://", "child://")
            )

            def resolve(target: str, _context: dict[str, object]) -> list[dict]:
                if target.startswith("root://"):
                    raise AssertionError("parent should not be resolved")
                return [
                    {
                        "source": target,
                        "label": "child.md",
                        "content": "child content",
                    }
                ]

            plugin.resolve = resolve
            plugin.list_targets = lambda _target, _context: {
                "targets": [{"target": "child://one", "label": "child.md"}]
            }
            plugin.materialize = lambda _target, _context: pytest.fail(
                "resolved plugin children should not be materialized"
            )
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_EmbeddedEntrypoint()]
    )
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["root://thread"],
        format="raw",
        target_depth=1,
        include_parent=False,
    )

    assert result["concatenated"] == "child content"


def test_embedded_target_depth_skips_non_traversable_items(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    class _EmbeddedEntrypoint:
        name = "embedded"
        value = "contextualize_plugins.embedded:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "embedded"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith(
                ("root://", "block://", "channel://")
            )

            def resolve(target: str, _context: dict[str, object]) -> list[dict]:
                if target.startswith("root://") or target.startswith("channel://"):
                    raise AssertionError(f"unexpected target resolution: {target}")
                return [
                    {
                        "source": target,
                        "label": target.removeprefix("block://") + ".md",
                        "content": target.removeprefix("block://") + " content",
                    }
                ]

            def list_targets(target: str, _context: dict[str, object]) -> dict:
                if target == "root://thread":
                    return {
                        "targets": [
                            {"target": "block://child", "label": "child.md"},
                            {
                                "target": "channel://nested",
                                "label": "nested",
                                "kind": "channel",
                                "traverse": False,
                            },
                        ]
                    }
                if target == "block://child":
                    return {
                        "targets": [
                            {"target": "block://grandchild", "label": "grandchild.md"}
                        ]
                    }
                return {"targets": []}

            plugin.resolve = resolve
            plugin.list_targets = list_targets
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_EmbeddedEntrypoint()]
    )
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["root://thread"],
        format="raw",
        target_depth=2,
        include_parent=False,
    )

    assert result["concatenated"] == "child content\n\ngrandchild content"


def test_embedded_target_skips_unclaimed_plain_http_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    class _EmbeddedEntrypoint:
        name = "embedded"
        value = "contextualize_plugins.embedded:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "embedded"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith("root://")
            plugin.resolve = lambda _target, _context: []
            plugin.list_targets = lambda _target, _context: {
                "targets": [
                    {
                        "target": "https://x.com/example/status/1?s=20",
                        "label": "tweet",
                    }
                ]
            }
            return plugin

    def _url_reference(*_args, **_kwargs):
        raise AssertionError("plain embedded links should not be fetched directly")

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_EmbeddedEntrypoint()]
    )
    monkeypatch.setattr(reference_factory, "URLReference", _url_reference)
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["root://thread"],
        format="raw",
        target_depth=1,
        include_parent=False,
    )

    assert result["concatenated"] == ""


def test_embedded_target_allows_file_like_http_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    class _EmbeddedEntrypoint:
        name = "embedded"
        value = "contextualize_plugins.embedded:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "embedded"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith("root://")
            plugin.resolve = lambda _target, _context: []
            plugin.list_targets = lambda _target, _context: {
                "targets": [
                    {
                        "target": "https://files.example/paper.pdf?download=1",
                        "label": "paper",
                    }
                ]
            }
            return plugin

    class _URLReference:
        def __init__(self, url: str, **_kwargs) -> None:
            assert url == "https://files.example/paper.pdf?download=1"
            self.output = "pdf content"

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_EmbeddedEntrypoint()]
    )
    monkeypatch.setattr(reference_factory, "URLReference", _URLReference)
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["root://thread"],
        format="raw",
        target_depth=1,
        include_parent=False,
    )

    assert result["concatenated"] == "pdf content"
