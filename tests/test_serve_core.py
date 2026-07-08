from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from contextualize.manifest import HydrateOverrides, hydrate_contexts
from contextualize.manifest.hydrate import apply_hydration_plan, build_hydration_plan
from contextualize.serve import cat_selector, links, shelf_for_cwd, show, status
from contextualize.manifest.contexts import load_context_registry


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))


def _registry(path: Path, contexts: dict) -> str:
    path.write_text(json.dumps({"version": 1, "contexts": contexts}), encoding="utf-8")
    return str(path)


def _hydrate(manifest_path: Path, cwd: Path) -> None:
    plan = build_hydration_plan(str(manifest_path), overrides=HydrateOverrides(), cwd=str(cwd))
    apply_hydration_plan(plan)


GRAMMAR_MANIFEST = """config:
  name: demo
  context:
    dir: .context/demo
    include-meta: true

components:
  # primary voice notes
  - group: content
    components:
      - name: alpha
        files:
          - a.md  # first note
          - b.md
      - name: beta
        files:
          - c.md

  - name: refs
    files:
      - a.md
      # - d.md  # excluded: superseded by b.md

  - set: fused
    files:
      - a.md
      - b.md

  - group: empty
    components: []
"""

ALL_DISABLED_MANIFEST = """config:
  name: alloff
components:
  - name: allOff
    files:
      # - a.md  # disabled
      # - b.md  # disabled
"""


@pytest.fixture
def grammar_dir(tmp_path: Path) -> Path:
    drive = tmp_path / "drive"
    drive.mkdir()
    (drive / "manifest.yaml").write_text(GRAMMAR_MANIFEST, encoding="utf-8")
    (drive / "a.md").write_text("alpha content\n", encoding="utf-8")
    (drive / "b.md").write_text("beta content\n", encoding="utf-8")
    (drive / "c.md").write_text("gamma content\n", encoding="utf-8")
    return drive


@pytest.fixture
def empty_registry(tmp_path: Path) -> str:
    return _registry(tmp_path / "registry.json", {})


@pytest.fixture
def all_disabled_dir(tmp_path: Path) -> Path:
    drive = tmp_path / "alloff-drive"
    drive.mkdir()
    (drive / "manifest.yaml").write_text(ALL_DISABLED_MANIFEST, encoding="utf-8")
    return drive


def test_show_renders_order_comments_and_disabled(monkeypatch, grammar_dir, empty_registry):
    _isolate(monkeypatch, grammar_dir)
    result = show(str(grammar_dir / "manifest.yaml"), registry_path=empty_registry)

    assert result["state"] == "unhydrated"
    children = result["node"]["children"]
    assert [child["name"] for child in children] == ["content", "refs", "fused", "empty"]

    content = children[0]
    assert content["kind"] == "group"
    assert content["comment"] == {"text": "primary voice notes", "line_start": 8, "line_end": 8}
    alpha = content["children"][0]
    assert alpha["members"]["files"][0]["inline_comment"] == "first note"

    refs = children[1]
    disabled_member = refs["members"]["files"][1]
    assert disabled_member["disabled"] is True
    assert "d.md" in disabled_member["raw"]

    fused = children[2]
    assert fused["kind"] == "set"


def test_show_narrows_to_component_and_member(monkeypatch, grammar_dir, empty_registry):
    _isolate(monkeypatch, grammar_dir)
    node = show(f"{grammar_dir / 'manifest.yaml'}:content.alpha", registry_path=empty_registry)["node"]
    assert node["path"] == "content.alpha"
    assert [m["spec"] for m in node["members"]["files"]] == ["a.md", "b.md"]

    member = show(f"{grammar_dir / 'manifest.yaml'}:content.alpha.a", registry_path=empty_registry)
    assert member["state"] == "ok"
    assert member["node"] == {
        "kind": "member",
        "key": "files",
        "spec": "a.md",
        "alias": None,
        "order": 0,
        "disabled": False,
        "comment": None,
        "inline_comment": "first note",
        "state": "ok",
        "detail": None,
    }


def test_show_depth_collapses_nested_groups(monkeypatch, grammar_dir, empty_registry):
    _isolate(monkeypatch, grammar_dir)
    result = show(str(grammar_dir / "manifest.yaml"), depth=0, registry_path=empty_registry)
    content = result["node"]["children"][0]
    assert content["collapsed"] is True
    assert content["child_count"] == 2
    assert "children" not in content


def test_cat_gathers_group_and_member_with_adjacency(monkeypatch, grammar_dir, empty_registry):
    _isolate(monkeypatch, grammar_dir)
    origin = str(grammar_dir / "manifest.yaml")

    whole = cat_selector(f"{origin}:content.alpha")
    assert whole["state"] == "ok"
    assert whole["specs"] == [str(grammar_dir / "a.md"), str(grammar_dir / "b.md")]
    assert whole["payload"] == "live-source"

    group = cat_selector(f"{origin}:content")
    assert sorted(group["specs"]) == sorted(
        str(grammar_dir / name) for name in ("a.md", "b.md", "c.md")
    )

    member = cat_selector(f"{origin}:content.alpha.a", around=1)
    assert member["specs"] == [str(grammar_dir / "a.md")]
    assert "a.md" in member["adjacency"]["text"]
    assert member["adjacency"]["line_start"] <= member["adjacency"]["line_end"]


def test_cat_prefers_resolved_copy_once_hydrated(monkeypatch, grammar_dir, empty_registry):
    _isolate(monkeypatch, grammar_dir)
    _hydrate(grammar_dir / "manifest.yaml", grammar_dir)
    origin = str(grammar_dir / "manifest.yaml")

    result = cat_selector(f"{origin}:content.alpha", cwd=str(grammar_dir))
    assert result["payload"] == "resolved-copy"
    assert all(spec.startswith(str(grammar_dir / ".context")) for spec in result["specs"])

    fused = cat_selector(f"{origin}:fused", cwd=str(grammar_dir))
    assert fused["payload"] == "resolved-copy"
    text = Path(fused["specs"][0]).read_text(encoding="utf-8")
    assert "alpha content" in text and "beta content" in text


def test_cat_disabled_member_reports_disabled_not_missing(monkeypatch, grammar_dir, empty_registry):
    _isolate(monkeypatch, grammar_dir)
    origin = str(grammar_dir / "manifest.yaml")
    result = cat_selector(f"{origin}:refs.d")
    assert result["state"] == "disabled"
    assert result["specs"] == []


def test_cat_all_disabled_component_is_disabled_only(monkeypatch, all_disabled_dir, empty_registry):
    _isolate(monkeypatch, all_disabled_dir)
    origin = str(all_disabled_dir / "manifest.yaml")
    result = cat_selector(f"{origin}:allOff")
    assert result["state"] == "disabled-only"
    assert result["specs"] == []


def test_show_group_with_only_disabled_children_is_disabled_only(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "components:\n"
        "  - group: g\n"
        "    components:\n"
        "      # - name: a\n"
        "      #   text: hi\n"
        "  - name: other\n"
        "    text: fine\n",
        encoding="utf-8",
    )
    result = show(f"{manifest}:g", registry_path=empty_registry)
    assert result["state"] == "disabled-only"
    assert result["node"]["children"][0]["disabled"] is True


def test_shelf_for_cwd_scopes_to_target_dir(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    inside = tmp_path / "proj"
    inside.mkdir()
    (inside / "manifest.yaml").write_text(
        "config:\n  name: proj\n  context:\n    include-meta: false\ncomponents:\n  - name: m\n    text: hi\n",
        encoding="utf-8",
    )
    registry = load_context_registry(
        _registry(tmp_path / "registry.json", {"proj": {"targetDir": str(inside), "manifest": {"source": "manifest.yaml"}}})
    )

    assert [entry["name"] for entry in shelf_for_cwd(str(inside), registry)] == ["proj"]
    assert shelf_for_cwd(str(tmp_path), registry) == []
    (inside / "sub").mkdir()
    assert [entry["name"] for entry in shelf_for_cwd(str(inside / "sub"), registry)] == ["proj"]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert shelf_for_cwd(str(elsewhere), registry) == []


def test_links_join_out_and_in_across_registered_contexts(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    parent_dir = tmp_path / "parent"
    child_dir = tmp_path / "child"
    parent_dir.mkdir()
    child_dir.mkdir()

    (child_dir / "note.md").write_text(
        "---\ntags:\n  - ctx/manifest\n---\n\n```yaml\nconfig:\n  name: child\n  context:\n"
        "    dir: .context/child\n    include-meta: false\ncomponents:\n  - name: main\n    text: hi\n```\n",
        encoding="utf-8",
    )
    (parent_dir / "manifest.yaml").write_text(
        "config:\n  name: parent\n  context:\n    dir: .context/parent\n    include-meta: true\n"
        "components:\n  - name: refs\n    files:\n      - ../child/note.md\n",
        encoding="utf-8",
    )

    registry_path = _registry(
        tmp_path / "registry.json",
        {
            "parent": {"targetDir": str(parent_dir), "manifest": {"source": "manifest.yaml"}},
            "child": {"targetDir": str(child_dir), "manifest": {"source": "note.md"}},
        },
    )

    statuses = hydrate_contexts(["parent"], registry_path=registry_path, status_path=str(tmp_path / "status.json"))
    assert statuses[0].result == "hydrated"

    out = links("parent", registry_path=registry_path)
    assert out["state"] == "ok"
    assert len(out["out"]) == 1
    assert out["out"][0]["form"] == "recognized"

    inbound = links("child", registry_path=registry_path, direction="in")
    assert inbound["state"] == "ok"
    assert len(inbound["in"]) == 1
    assert inbound["in"][0]["source_context"] == "parent"
    assert inbound["in"][0]["target_context"] == "child"
    assert inbound["coverage"]["scanned"] == 1


def test_status_detects_drift(monkeypatch, grammar_dir, empty_registry):
    _isolate(monkeypatch, grammar_dir)
    origin = str(grammar_dir / "manifest.yaml")
    _hydrate(grammar_dir / "manifest.yaml", grammar_dir)

    fresh = status(origin, registry_path=empty_registry, cwd=str(grammar_dir))
    assert fresh["state"] == "ok"
    assert fresh["drift"]["any"] is False

    time.sleep(1.1)
    (grammar_dir / "a.md").write_text("alpha content CHANGED\n", encoding="utf-8")

    drifted = status(origin, registry_path=empty_registry, cwd=str(grammar_dir))
    assert drifted["state"] == "stale-cache"
    assert drifted["drift"]["any"] is True
    assert any(item["path"].endswith("a.md") for item in drifted["drift"]["sources_changed"])


def test_status_registry_wide_summarizes_all_contexts(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "manifest.yaml").write_text(
        "config:\n  name: one\n  context:\n    include-meta: true\ncomponents:\n  - name: m\n    text: hi\n",
        encoding="utf-8",
    )
    (two / "manifest.yaml").write_text(
        "config:\n  name: two\n  context:\n    include-meta: false\ncomponents:\n  - name: m\n    text: hi\n",
        encoding="utf-8",
    )
    registry_path = _registry(
        tmp_path / "registry.json",
        {
            "one": {"targetDir": str(one), "manifest": {"source": "manifest.yaml"}},
            "two": {"targetDir": str(two), "manifest": {"source": "manifest.yaml"}},
        },
    )
    hydrate_contexts(["one"], registry_path=registry_path, status_path=str(tmp_path / "status.json"))

    result = status(None, registry_path=registry_path)
    assert result["registry"]["total"] == 2
    names_states = {c["name"]: c["state"] for c in result["contexts"]}
    assert names_states == {"one": "ok", "two": "unhydrated"}


def test_designed_state_not_found_for_unknown_origin(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    result = show("nowhere-at-all:comp", registry_path=empty_registry)
    assert result["state"] == "not-found"
    assert result["next_steps"]


def test_designed_state_path_not_found_for_bad_selector(monkeypatch, grammar_dir, empty_registry):
    _isolate(monkeypatch, grammar_dir)
    result = show(f"{grammar_dir / 'manifest.yaml'}:nosuch", registry_path=empty_registry)
    assert result["state"] == "not-found"
    assert "nosuch" in result["detail"]


def test_designed_state_empty_group(monkeypatch, grammar_dir, empty_registry):
    _isolate(monkeypatch, grammar_dir)
    result = show(f"{grammar_dir / 'manifest.yaml'}:empty", registry_path=empty_registry)
    assert result["state"] == "empty-group"


def test_designed_state_unresolvable_source(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    manifest = tmp_path / "gone.yaml"
    manifest.write_text("config:\n  name: gone\ncomponents:\n  - name: m\n    text: hi\n", encoding="utf-8")
    registry_path = _registry(
        tmp_path / "registry.json",
        {"gone": {"targetDir": str(tmp_path), "manifest": {"source": "gone.yaml"}}},
    )
    manifest.unlink()

    result = show("gone", registry_path=registry_path)
    assert result["state"] == "unresolvable-source"
