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



def test_embedded_resolution_materialization_uses_normal_file_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    class _ArenaEntrypoint:
        name = "arena"
        value = "contextualize_plugins.arena:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "arena"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith(
                ("root://", "asset://", "https://www.are.na/block/")
            )

            def resolve(target: str, _context: dict[str, object]) -> list[dict]:
                if target == "root://channel":
                    return [
                        {
                            "source": target,
                            "label": "entry.md",
                            "content": "entry content",
                            "metadata": {
                                "provider": "arena",
                                "block_id": 42,
                                "block_type": "Link",
                                "channel_path": "channels/root",
                                "context_subpath": "channels/root/42.md",
                            },
                        }
                    ]
                return []

            def list_targets(target: str, _context: dict[str, object]) -> dict:
                if target != "https://www.are.na/block/42":
                    return {"targets": []}
                return {
                    "targets": [
                        {
                            "target": "asset://child",
                            "label": "child.md",
                            "kind": "attachment:text",
                        }
                    ]
                }

            plugin.resolve = resolve
            plugin.list_targets = list_targets
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
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_ArenaEntrypoint()]
    )
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["root://channel"],
        format="raw",
        embedded_resolution=True,
    )

    assert result["concatenated"] == "entry content\n\nchild content"


def test_embedded_resolution_lists_direct_plugin_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    listed: list[str] = []

    class _DirectEntrypoint:
        name = "direct"
        value = "contextualize_plugins.direct:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "direct"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith(
                ("direct://", "asset://")
            )

            def resolve(target: str, _context: dict[str, object]) -> list[dict]:
                if target != "direct://message":
                    return []
                return [
                    {
                        "source": target,
                        "label": "message.md",
                        "content": "message content",
                        "metadata": {
                            "provider": "direct",
                            "context_subpath": "messages/message.md",
                        },
                    }
                ]

            def list_targets(target: str, _context: dict[str, object]) -> dict:
                listed.append(target)
                return {
                    "targets": [
                        {
                            "target": "asset://attachment",
                            "label": "attachment.md",
                        }
                    ]
                }

            plugin.resolve = resolve
            plugin.list_targets = list_targets
            plugin.materialize = lambda _target, _context: [
                {
                    "source": "asset://attachment",
                    "label": "attachment.md",
                    "filename": "attachment.md",
                    "content": b"attachment content",
                    "content_type": "text/markdown",
                }
            ]
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_DirectEntrypoint()]
    )
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["direct://message"],
        format="raw",
        embedded_resolution=True,
    )

    assert listed == ["direct://message"]
    assert result["concatenated"] == "message content\n\nattachment content"


def test_embedded_resolution_text_only_materialized_audio_is_transcribed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)
    transcribed: list[str] = []

    class _ArenaEntrypoint:
        name = "arena"
        value = "contextualize_plugins.arena:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "arena"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith(
                ("root://", "asset://", "https://www.are.na/block/")
            )

            def resolve(target: str, _context: dict[str, object]) -> list[dict]:
                if target != "root://channel":
                    return []
                return [
                    {
                        "source": target,
                        "label": "entry.md",
                        "content": "entry content",
                        "metadata": {
                            "provider": "arena",
                            "block_id": 42,
                            "block_type": "Attachment",
                            "channel_path": "channels/root",
                            "context_subpath": "channels/root/42.md",
                        },
                    }
                ]

            plugin.resolve = resolve
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
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_ArenaEntrypoint()]
    )
    monkeypatch.setattr(
        "contextualize.references.file.transcribe_media_file", _transcribe
    )
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["root://channel"],
        format="raw",
        embedded_resolution=True,
        text_only=True,
    )

    captured = capsys.readouterr()
    assert result["concatenated"] == "entry content\n\naudio transcript"
    assert transcribed == ["child.mp3"]
    assert captured.err == ""


def test_embedded_resolution_materialized_unknown_binary_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    class _ArenaEntrypoint:
        name = "arena"
        value = "contextualize_plugins.arena:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "arena"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith(
                ("root://", "asset://", "https://www.are.na/block/")
            )

            def resolve(target: str, _context: dict[str, object]) -> list[dict]:
                if target != "root://channel":
                    return []
                return [
                    {
                        "source": target,
                        "label": "entry.md",
                        "content": "entry content",
                        "metadata": {
                            "provider": "arena",
                            "block_id": 42,
                            "block_type": "Attachment",
                            "channel_path": "channels/root",
                            "context_subpath": "channels/root/42.md",
                        },
                    }
                ]

            plugin.resolve = resolve
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
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_ArenaEntrypoint()]
    )
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["root://channel"],
        format="raw",
        embedded_resolution=True,
        text_only=True,
    )

    captured = capsys.readouterr()
    assert result["concatenated"] == "entry content"
    assert captured.err == ""


def test_embedded_resolution_resolves_plugin_claimed_child_without_materialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    class _ArenaEntrypoint:
        name = "arena"
        value = "contextualize_plugins.arena:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "arena"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith(
                ("root://", "https://www.are.na/block/")
            )

            def resolve(target: str, _context: dict[str, object]) -> list[dict]:
                if target != "root://channel":
                    return []
                return [
                    {
                        "source": target,
                        "label": "entry.md",
                        "content": "entry content",
                        "metadata": {
                            "provider": "arena",
                            "block_id": 42,
                            "block_type": "Link",
                            "channel_path": "channels/root",
                            "context_subpath": "channels/root/42.md",
                        },
                    }
                ]

            plugin.resolve = resolve
            plugin.list_targets = lambda _target, _context: {
                "targets": [{"target": "child://one", "label": "child.md"}]
            }
            return plugin

    class _ChildEntrypoint:
        name = "child"
        value = "contextualize_plugins.child:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "child"
            plugin.PLUGIN_PRIORITY = 600
            plugin.can_resolve = lambda target, _context: target.startswith("child://")
            plugin.resolve = lambda target, _context: [
                {
                    "source": target,
                    "label": "child.md",
                    "content": "child content",
                }
            ]
            plugin.materialize = lambda _target, _context: pytest.fail(
                "resolved plugin children should not be materialized"
            )
            return plugin

    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [_ArenaEntrypoint(), _ChildEntrypoint()],
    )
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["root://channel"],
        format="raw",
        embedded_resolution=True,
    )

    assert result["concatenated"] == "entry content\n\nchild content"


def test_embedded_resolution_does_not_recurse_child_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    class _ArenaEntrypoint:
        name = "arena"
        value = "contextualize_plugins.arena:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "arena"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith(
                ("root://", "https://www.are.na/block/")
            )

            def resolve(target: str, _context: dict[str, object]) -> list[dict]:
                if target != "root://channel":
                    return []
                return [
                    {
                        "source": target,
                        "label": "entry.md",
                        "content": "entry content",
                        "metadata": {
                            "provider": "arena",
                            "block_id": 42,
                            "block_type": "Link",
                            "channel_path": "channels/root",
                            "context_subpath": "channels/root/42.md",
                        },
                    }
                ]

            plugin.resolve = resolve
            plugin.list_targets = lambda _target, _context: {
                "targets": [{"target": "child://one", "label": "child.md"}]
            }
            return plugin

    class _ChildEntrypoint:
        name = "child"
        value = "contextualize_plugins.child:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "child"
            plugin.PLUGIN_PRIORITY = 600
            plugin.can_resolve = lambda target, _context: target.startswith("child://")
            plugin.resolve = lambda target, _context: [
                {
                    "source": target,
                    "label": "child.md",
                    "content": "child content",
                }
            ]
            plugin.list_targets = lambda _target, _context: {
                "targets": [{"target": "grandchild://one", "label": "grandchild.md"}]
            }
            return plugin

    class _GrandchildEntrypoint:
        name = "grandchild"
        value = "contextualize_plugins.grandchild:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "grandchild"
            plugin.PLUGIN_PRIORITY = 700
            plugin.can_resolve = lambda target, _context: target.startswith(
                "grandchild://"
            )
            plugin.resolve = lambda _target, _context: [
                {
                    "source": "grandchild://one",
                    "label": "grandchild.md",
                    "content": "grandchild content",
                }
            ]
            return plugin

    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [_ArenaEntrypoint(), _ChildEntrypoint(), _GrandchildEntrypoint()],
    )
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["root://channel"],
        format="raw",
        embedded_resolution=True,
    )

    assert result["concatenated"] == "entry content\n\nchild content"


def test_embedded_resolution_skips_unclaimed_http_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_plugin_env(monkeypatch, tmp_path)

    class _ArenaEntrypoint:
        name = "arena"
        value = "contextualize_plugins.arena:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "arena"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith(
                ("root://", "https://www.are.na/block/")
            )

            def resolve(target: str, _context: dict[str, object]) -> list[dict]:
                if target != "root://channel":
                    return []
                return [
                    {
                        "source": target,
                        "label": "entry.md",
                        "content": "entry content",
                        "metadata": {
                            "provider": "arena",
                            "block_id": 42,
                            "block_type": "Link",
                            "channel_path": "channels/root",
                            "context_subpath": "channels/root/42.md",
                        },
                    }
                ]

            plugin.resolve = resolve
            plugin.list_targets = lambda _target, _context: {
                "targets": [
                    {
                        "target": "https://files.example/paper.pdf?download=1",
                        "label": "paper",
                    }
                ]
            }
            return plugin

    def _url_reference(*_args, **_kwargs):
        raise AssertionError("embedded resolution should not fetch unclaimed links")

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_ArenaEntrypoint()]
    )
    monkeypatch.setattr(reference_factory, "URLReference", _url_reference)
    clear_loaded_plugins_cache()

    result = create_file_references(
        ["root://channel"],
        format="raw",
        embedded_resolution=True,
    )

    assert result["concatenated"] == "entry content"
