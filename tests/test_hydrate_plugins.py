from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from contextualize.manifest.hydrate import (
    HydrateOverrides,
    apply_hydration_plan,
    build_hydration_plan_data,
    _resolve_external_items_via_refs,
)
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

            def classify_target(_target, _context):
                raise AssertionError(
                    "hydrate should not classify targets during planning"
                )

            plugin.classify_target = classify_target
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


def test_hydrate_manifest_preserves_colon_plugin_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    class _ColonEntrypoint:
        name = "colon"
        value = "contextualize_plugins.colon:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "colon"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith("note:")
            plugin.resolve = lambda target, _context: [
                {
                    "source": target,
                    "label": "note/doc.md",
                    "content": "hello from colon target",
                    "metadata": {"context_subpath": "note/doc.md"},
                }
            ]
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_ColonEntrypoint()]
    )
    clear_loaded_plugins_cache()

    context_dir = tmp_path / "ctx"
    plan = build_hydration_plan_data(
        {
            "config": {"context": {"dir": str(context_dir), "include-meta": False}},
            "components": [{"name": "main", "files": ["note:voice/abc"]}],
        },
        manifest_cwd=str(tmp_path),
        overrides=HydrateOverrides(),
        cwd=str(tmp_path),
    )

    rel_paths = {
        path.relative_to(context_dir).as_posix() for path, _ in plan.files_to_write
    }
    assert "note/doc.md" in rel_paths


def test_hydrate_plugin_dedupe_can_skip_noncanonical_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    class _DedupeEntrypoint:
        name = "dedupe"
        value = "contextualize_plugins.dedupe:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "dedupe"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target == "dedupe://root"
            plugin.resolve = lambda target, _context: [
                {
                    "source": target,
                    "label": "channel/a.md",
                    "content": "canonical",
                    "metadata": {
                        "context_subpath": "channel/a.md",
                        "source_ref": "dedupe",
                        "source_path": "channel/a",
                        "hydrate_dedupe": {
                            "mode": "canonical_symlink",
                            "key": "block:1",
                            "rank": 0,
                        },
                    },
                },
                {
                    "source": target,
                    "label": "channel/b.md",
                    "content": "canonical",
                    "metadata": {
                        "context_subpath": "channel/b.md",
                        "source_ref": "dedupe",
                        "source_path": "channel/b",
                        "hydrate_dedupe": {
                            "mode": "canonical_symlink",
                            "key": "block:1",
                            "rank": 1,
                        },
                    },
                },
                {
                    "source": target,
                    "label": "flat.md",
                    "content": "canonical",
                    "metadata": {
                        "context_subpath": "flat.md",
                        "source_ref": "dedupe",
                        "source_path": "flat",
                        "hydrate_dedupe": {
                            "mode": "canonical_symlink",
                            "key": "block:1",
                            "rank": 10_000,
                            "link": False,
                        },
                    },
                },
            ]
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_DedupeEntrypoint()]
    )
    clear_loaded_plugins_cache()

    context_dir = tmp_path / "ctx"
    plan = build_hydration_plan_data(
        {
            "config": {"context": {"dir": str(context_dir), "include-meta": True}},
            "components": [{"name": "main", "files": ["dedupe://root"]}],
        },
        manifest_cwd=str(tmp_path),
        overrides=HydrateOverrides(),
        cwd=str(tmp_path),
    )

    written_paths = {
        path.relative_to(context_dir).as_posix() for path, _ in plan.files_to_write
    }
    symlink_paths = {
        path.relative_to(context_dir).as_posix()
        for path, _ in plan.files_to_symlink
    }
    index_text = next(
        content
        for path, content in plan.files_to_write
        if path.relative_to(context_dir).as_posix() == "index.json"
    )
    index_paths = {
        entry["context_path"]
        for entry in json.loads(index_text)["components"]["main"]
    }

    assert "channel/a.md" in written_paths
    assert "channel/b.md" in symlink_paths
    assert "flat.md" not in written_paths
    assert "flat.md" not in symlink_paths
    assert "flat.md" not in index_paths



def test_hydrate_embedded_resolution_targets_inherit_parent_context_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

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
                        "label": "entry-42.md",
                        "content": "entry 42",
                        "metadata": {
                            "provider": "arena",
                            "block_id": 42,
                            "block_type": "Link",
                            "channel_path": "channels/root",
                            "context_subpath": "channels/root/42.md",
                            "source_ref": "are.na",
                            "source_path": "channels/root/42",
                        },
                    },
                    {
                        "source": target,
                        "label": "entry-43.md",
                        "content": "entry 43",
                        "metadata": {
                            "provider": "arena",
                            "block_id": 43,
                            "block_type": "Link",
                            "channel_path": "channels/root",
                            "context_subpath": "channels/root/43.md",
                            "source_ref": "are.na",
                            "source_path": "channels/root/43",
                        },
                    },
                ]

            def list_targets(target: str, _context: dict[str, object]) -> dict:
                if target == "https://www.are.na/block/42":
                    return {"targets": [{"target": "asset://child", "label": "Child"}]}
                if target == "https://www.are.na/block/43":
                    return {"targets": [{"target": "asset://child", "label": "Child again"}]}
                return {"targets": []}

            plugin.resolve = resolve
            plugin.list_targets = list_targets
            return plugin

    class _AssetEntrypoint:
        name = "asset"
        value = "contextualize_plugins.asset:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "asset"
            plugin.PLUGIN_PRIORITY = 600
            plugin.can_resolve = lambda target, _context: target == "asset://child"
            plugin.resolve = lambda target, _context: [
                {
                    "source": target,
                    "label": "asset.md",
                    "content": "child content",
                    "metadata": {
                        "context_subpath": "child.md",
                        "source_ref": "asset",
                        "source_path": "child",
                    },
                }
            ]
            return plugin

    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [_ArenaEntrypoint(), _AssetEntrypoint()],
    )
    clear_loaded_plugins_cache()

    context_dir = tmp_path / "ctx"
    plan = build_hydration_plan_data(
        {
            "config": {"context": {"dir": str(context_dir), "include-meta": False}},
            "components": [
                {
                    "name": "main",
                    "embedded-resolution": True,
                    "files": ["root://channel"],
                }
            ],
        },
        manifest_cwd=str(tmp_path),
        overrides=HydrateOverrides(),
        cwd=str(tmp_path),
    )

    assert [
        (path.relative_to(context_dir).as_posix(), content)
        for path, content in plan.files_to_write
    ] == [
        ("channels/root/42.md", "entry 42"),
        ("channels/root/43.md", "entry 43"),
        ("channels/root/42.asset-child.md", "child content"),
        ("channels/root/43.asset-child.md", "child content"),
    ]


def test_hydrate_embedded_resolution_media_targets_use_parent_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

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
            plugin.resolve = lambda target, _context: [
                {
                    "source": target,
                    "label": "entry.md",
                    "content": "entry content",
                    "metadata": {
                        "provider": "arena",
                        "block_id": 42,
                        "block_type": "Embed",
                        "channel_path": "channels/root",
                        "context_subpath": "channels/root/42.md",
                        "source_ref": "are.na",
                        "source_path": "channels/root/42",
                    },
                }
            ] if target == "root://channel" else []
            plugin.list_targets = lambda target, _context: {
                "targets": [{"target": "media://child", "label": "Media child"}]
            } if target == "https://www.are.na/block/42" else {"targets": []}
            return plugin

    class _YtdlpEntrypoint:
        name = "ytdlp"
        value = "contextualize_plugins.ytdlp:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "ytdlp"
            plugin.PLUGIN_PRIORITY = 600
            plugin.can_resolve = lambda target, _context: target == "media://child"
            plugin.resolve = lambda target, _context: [
                {
                    "source": target,
                    "label": "Video",
                    "content": "media transcript",
                    "metadata": {
                        "context_subpath": "ytdlp-youtube-abc123.md",
                        "source_ref": "www.youtube.com",
                        "source_path": "youtube:abc123",
                    },
                }
            ]
            return plugin

    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [_ArenaEntrypoint(), _YtdlpEntrypoint()],
    )
    clear_loaded_plugins_cache()

    context_dir = tmp_path / "ctx"
    plan = build_hydration_plan_data(
        {
            "config": {"context": {"dir": str(context_dir), "include-meta": False}},
            "components": [
                {
                    "name": "main",
                    "embedded-resolution": True,
                    "files": ["root://channel"],
                }
            ],
        },
        manifest_cwd=str(tmp_path),
        overrides=HydrateOverrides(),
        cwd=str(tmp_path),
    )

    assert [
        (path.relative_to(context_dir).as_posix(), content)
        for path, content in plan.files_to_write
    ] == [
        ("channels/root/42.md", "entry content"),
        ("channels/root/42.ytdlp-youtube-abc123.md", "media transcript"),
    ]


def test_hydrate_embedded_resolution_arena_attachment_uses_dot_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    class _ArenaEntrypoint:
        name = "arena"
        value = "contextualize_plugins.arena:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "arena"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = lambda target, _context: target.startswith(
                ("root://", "attachment://", "https://www.are.na/block/")
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
                                "block_type": "Attachment",
                                "channel_path": "channels/root",
                                "context_subpath": "channels/root/42.md",
                                "source_ref": "are.na",
                                "source_path": "channels/root/42",
                            },
                        }
                    ]
                if target == "attachment://child":
                    return [
                        {
                            "source": target,
                            "label": "attachment-video.mp4.md",
                            "content": "attachment content",
                            "metadata": {
                                "context_subpath": (
                                    "arena-block-42/attachment-video.mp4.mp4.md"
                                ),
                                "source_ref": "are.na",
                                "source_path": "42/attachment/video.mp4",
                            },
                        }
                    ]
                return []

            plugin.resolve = resolve
            plugin.list_targets = lambda target, _context: {
                "targets": [{"target": "attachment://child", "label": "Attachment"}]
            } if target == "https://www.are.na/block/42" else {"targets": []}
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_ArenaEntrypoint()]
    )
    clear_loaded_plugins_cache()

    context_dir = tmp_path / "ctx"
    plan = build_hydration_plan_data(
        {
            "config": {"context": {"dir": str(context_dir), "include-meta": False}},
            "components": [
                {
                    "name": "main",
                    "embedded-resolution": True,
                    "files": ["root://channel"],
                }
            ],
        },
        manifest_cwd=str(tmp_path),
        overrides=HydrateOverrides(),
        cwd=str(tmp_path),
    )

    assert [
        (path.relative_to(context_dir).as_posix(), content)
        for path, content in plan.files_to_write
    ] == [
        ("channels/root/42.md", "entry content"),
        ("channels/root/42.attachment-video.mp4.mp4.md", "attachment content"),
    ]


def test_hydrate_embedded_resolution_http_refs_use_ref_path_and_dot_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Ref:
        path = "https://example.com/assets/doc.txt"
        file_content = "doc content"
        _contextualize_context_prefix = "channels/root/42"
        _contextualize_context_sidecar_stem = "channels/root/42"

    monkeypatch.setattr(
        "contextualize.manifest.hydrate.create_file_references",
        lambda *_args, **_kwargs: {"refs": [_Ref()]},
    )

    items = _resolve_external_items_via_refs(
        "root://channel",
        alias=None,
        embedded_resolution=True,
    )

    assert len(items) == 1
    assert items[0].source_type == "http"
    assert items[0].source_ref == "https://example.com"
    assert items[0].source_path == "assets/doc.txt"
    assert items[0].context_subpath == "channels/root/42.doc.txt.md"
    assert items[0].manifest_spec == "https://example.com/assets/doc.txt"


def test_hydrate_manifest_embedded_resolution_materializes_embedded_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

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
                if target == "root://thread":
                    return [
                        {
                            "source": target,
                            "label": "entry.md",
                            "content": "entry content",
                            "metadata": {
                                "provider": "arena",
                                "block_id": 42,
                                "block_type": "Link",
                                "channel_path": "thread",
                                "context_subpath": "thread/42.md",
                                "source_ref": "are.na",
                                "source_path": "thread/42",
                            },
                        }
                    ]
                return []

            plugin.resolve = resolve
            plugin.list_targets = lambda target, _context: {
                "targets": [
                    {"target": "asset://child", "label": "child.txt"},
                    {
                        "target": "asset://nested-channel",
                        "label": "Nested channel",
                        "kind": "channel",
                        "traverse": False,
                    },
                ]
            } if target == "https://www.are.na/block/42" else {"targets": []}

            def materialize(target: str, _context: dict[str, object]) -> list[dict]:
                if target != "asset://child":
                    raise AssertionError(f"unexpected target materialization: {target}")
                return [
                    {
                        "source": "asset://child",
                        "label": "child.txt",
                        "filename": "child.txt",
                        "content": b"child content",
                        "content_type": "text/plain",
                    }
                ]

            plugin.materialize = materialize
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_ArenaEntrypoint()]
    )
    clear_loaded_plugins_cache()

    context_dir = tmp_path / "ctx"
    plan = build_hydration_plan_data(
        {
            "config": {
                "context": {
                    "dir": str(context_dir),
                    "include-meta": False,
                    "path-strategy": "by-component",
                }
            },
            "components": [
                {
                    "name": "main",
                    "embedded-resolution": True,
                    "files": ["root://thread"],
                }
            ],
        },
        manifest_cwd=str(tmp_path),
        overrides=HydrateOverrides(),
        cwd=str(tmp_path),
    )

    assert [
        (path.relative_to(context_dir).as_posix(), content)
        for path, content in plan.files_to_write
    ] == [
        ("main/thread/42.md", "entry content"),
        ("main/thread/42.materialized-child.txt.md", "child content"),
    ]

def test_hydrate_git_target_copies_binary_files_without_resolving_references(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    asset = repo_dir / "src" / "image.png"
    text = repo_dir / "src" / "readme.txt"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"\x89PNG\r\n\x1a\nbinary")
    text.write_text("hello", encoding="utf-8")

    def fail_create_file_references(*_args, **_kwargs):
        raise AssertionError("hydrate git should not resolve files as text refs")

    monkeypatch.setattr(
        "contextualize.manifest.hydrate.ensure_repo", lambda _target: str(repo_dir)
    )
    monkeypatch.setattr(
        "contextualize.manifest.hydrate.create_file_references",
        fail_create_file_references,
    )

    context_dir = tmp_path / "ctx"
    plan = build_hydration_plan_data(
        {
            "config": {"context": {"dir": str(context_dir), "include-meta": False}},
            "components": [
                {"name": "main", "files": ["https://github.com/org/repo:src"]}
            ],
        },
        manifest_cwd=str(tmp_path),
        overrides=HydrateOverrides(),
        cwd=str(tmp_path),
    )

    copied = {
        dest.relative_to(context_dir).as_posix(): source
        for dest, source in plan.files_to_copy
    }
    assert copied["repo/src/image.png"] == asset
    assert copied["repo/src/readme.txt"] == text

    result = apply_hydration_plan(plan)

    assert result.file_count == 2
    assert (context_dir / "repo/src/image.png").read_bytes() == asset.read_bytes()
    assert (context_dir / "repo/src/readme.txt").read_text(encoding="utf-8") == "hello"


def test_hydrate_local_file_with_late_invalid_utf8_is_materialized(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.dat"
    source.write_bytes((b"a" * 5000) + b"\xd8")
    context_dir = tmp_path / "ctx"

    plan = build_hydration_plan_data(
        {
            "config": {"context": {"dir": str(context_dir), "include-meta": False}},
            "components": [{"name": "main", "files": [str(source)]}],
        },
        manifest_cwd=str(tmp_path),
        overrides=HydrateOverrides(copy=True),
        cwd=str(tmp_path),
    )

    copied = {
        dest.relative_to(context_dir).as_posix(): copy_source
        for dest, copy_source in plan.files_to_copy
    }
    assert copied == {"asset.dat": source}

    apply_hydration_plan(plan)

    assert (context_dir / "asset.dat").read_bytes() == source.read_bytes()


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


def test_hydrate_manifest_fails_for_unloaded_colon_plugin_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(plugin_loader, "_iter_plugin_entrypoints", lambda: [])
    clear_loaded_plugins_cache()

    with pytest.raises(
        ValueError,
        match="No plugin could resolve external target: note:voice/abc.m4a",
    ):
        build_hydration_plan_data(
            {
                "config": {
                    "context": {"dir": str(tmp_path / "ctx"), "include-meta": False}
                },
                "components": [{"name": "main", "files": ["note:voice/abc.m4a"]}],
            },
            manifest_cwd=str(tmp_path),
            overrides=HydrateOverrides(),
            cwd=str(tmp_path),
        )
