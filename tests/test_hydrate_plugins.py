from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from contextualize.manifest.hydrate import (
    HydrateOverrides,
    apply_hydration_plan,
    build_hydration_plan_data,
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


def test_hydrate_manifest_follows_component_embedded_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

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
                "targets": [
                    {"target": "asset://child", "label": "child.txt"},
                    {
                        "target": "asset://nested-channel",
                        "label": "Nested channel",
                        "kind": "channel",
                        "traverse": False,
                    },
                ]
            }

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
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_EmbeddedEntrypoint()]
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
                    "target-depth": 1,
                    "include-parent": False,
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
    ] == [("main/child.txt", "child content")]


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
