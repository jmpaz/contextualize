from __future__ import annotations

from pathlib import Path

from contextualize.manifest.build import _resolve_spec_to_seed_refs
from contextualize.plugins import clear_loaded_plugins_cache


def _write_plugin(root: Path, name: str) -> None:
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
                "    return target.startswith('demo://')",
                "def resolve(target, context):",
                "    return [{'source': target, 'label': 'demo/item.md', 'content': 'demo body'}]",
            ]
        ),
        encoding="utf-8",
    )


def test_manifest_build_routes_custom_scheme_through_plugins(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plugins_root = tmp_path / "plugins"
    monkeypatch.setenv("CONTEXTUALIZE_PLUGIN_DIRS", str(plugins_root))
    _write_plugin(plugins_root, "demo")
    clear_loaded_plugins_cache()

    seed_refs, trace_inputs, trace_items = _resolve_spec_to_seed_refs(
        "demo://abc",
        {},
        str(tmp_path),
        inject=False,
        depth=5,
        component_name="main",
        label_suffix=None,
    )

    assert len(seed_refs) == 1
    assert len(trace_inputs) == 1
    assert trace_items == []
    assert "demo body" in seed_refs[0].output
    assert trace_inputs[0].path == "demo://abc"


def test_manifest_build_counts_git_url_inputs_in_trace(
    monkeypatch, tmp_path: Path
) -> None:
    local_file = tmp_path / "repo-file.txt"
    local_file.write_text("from git clone", encoding="utf-8")

    class _FakeRef:
        def __init__(self, path: str) -> None:
            self.path = path
            self.output = "wrapped"
            self.file_content = "from git clone"

    def _ensure_repo(_target):
        return str(tmp_path)

    def _expand_git_paths(_repo_dir: str, _pattern: str):
        return [str(local_file)]

    def _create_refs(*_args, **_kwargs):
        return {"refs": [_FakeRef(str(local_file))]}

    monkeypatch.setattr("contextualize.manifest.build.ensure_repo", _ensure_repo)
    monkeypatch.setattr(
        "contextualize.manifest.build.expand_git_paths", _expand_git_paths
    )
    monkeypatch.setattr(
        "contextualize.manifest.build.create_file_references",
        _create_refs,
    )

    seed_refs, trace_inputs, trace_items = _resolve_spec_to_seed_refs(
        "https://github.com/org/repo:path/in/repo.txt",
        {},
        str(tmp_path),
        inject=False,
        depth=5,
        component_name="main",
        label_suffix=None,
    )

    assert len(seed_refs) == 1
    assert len(trace_inputs) == 1
    assert trace_items == []
    assert trace_inputs[0].path == str(local_file)
