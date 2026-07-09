"""Every designed state named in the manifest-grammar/serving-surface spec
(dsi5.2, dsi7), proven directly: each returns a legible state and at least
one next step rather than a raw error or a silent gap. One test per state
per verb where that verb can produce it."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from contextualize.manifest import HydrateOverrides, hydrate_contexts
from contextualize.manifest.hydrate import apply_hydration_plan, build_hydration_plan
from contextualize.serve import cat_selector, links, show, status


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))


def _registry(path: Path, contexts: dict) -> str:
    path.write_text(json.dumps({"version": 1, "contexts": contexts}), encoding="utf-8")
    return str(path)


def _hydrate(manifest_path: Path, cwd: Path) -> None:
    plan = build_hydration_plan(str(manifest_path), overrides=HydrateOverrides(), cwd=str(cwd))
    apply_hydration_plan(plan)


@pytest.fixture
def empty_registry(tmp_path: Path) -> str:
    return _registry(tmp_path / "registry.json", {})


ALL_STATES = {
    "not-found",
    "unhydrated",
    "disabled",
    "disabled-only",
    "empty-group",
    "empty-manifest",
    "unresolvable-source",
    "stale-cache",
    "dangling-reference",
    "hydrate-failed",
    "mark-quote-requires-range",
    "mark-invalid",
    "mark-invalid-time",
    "mark-at-and-span",
    "mark-missing-time",
    "mark-beyond-duration",
    "marks-on-untimed-target",
    "marks-require-single-document",
    "mark-params-unsupported",
    "transcript-drift",
    "ok",
}


def _assert_legible(result: dict) -> None:
    assert result["state"] in ALL_STATES
    if result["state"] != "ok":
        assert result.get("detail"), "non-ok state must name the condition"
        assert result.get("next_steps"), "non-ok state must name a path out"


def test_not_found_across_all_four_verbs(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    for result in (
        show("nowhere-at-all:x", registry_path=empty_registry),
        cat_selector("nowhere-at-all:x", registry_path=empty_registry),
        links("nowhere-at-all:x", registry_path=empty_registry),
        status("nowhere-at-all", registry_path=empty_registry),
    ):
        assert result["state"] == "not-found"
        _assert_legible(result)


def test_unhydrated_is_root_level_show_status_and_links_out(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    manifest = tmp_path / "manifest.yaml"
    (tmp_path / "a.md").write_text("a\n", encoding="utf-8")
    manifest.write_text(
        "components:\n  - name: m\n    files:\n      - a.md\n", encoding="utf-8"
    )
    origin = str(manifest)

    shown = show(origin, registry_path=empty_registry)
    _assert_legible(shown)
    assert shown["state"] == "unhydrated"

    out_links = links(origin, registry_path=empty_registry, direction="out")
    _assert_legible(out_links)
    assert out_links["state"] == "unhydrated"

    statused = status(origin, registry_path=empty_registry, cwd=str(tmp_path))
    _assert_legible(statused)
    assert statused["state"] == "unhydrated"

    cat_whole = cat_selector(f"{origin}:m", cwd=str(tmp_path))
    assert cat_whole["state"] == "ok", "cat draws live substance without hydration"
    assert cat_whole["payload"] == "pointer"


def test_empty_group(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("components:\n  - group: g\n    components: []\n", encoding="utf-8")
    result = show(f"{manifest}:g", registry_path=empty_registry)
    _assert_legible(result)
    assert result["state"] == "empty-group"


def test_failed_hydrate_outranks_green_status(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    drive = tmp_path / "drive"
    drive.mkdir()
    (drive / "a.md").write_text("a\n", encoding="utf-8")
    (drive / "manifest.yaml").write_text(
        "config:\n  name: demo\n  context:\n    dir: .context/demo\ncomponents:\n"
        "  - name: m\n    files:\n      - a.md\n",
        encoding="utf-8",
    )
    registry_path = _registry(
        tmp_path / "registry.json",
        {"demo": {"targetDir": str(drive), "manifest": {"source": "manifest.yaml"}}},
    )
    hydrate_contexts(["demo"], registry_path=registry_path, status_path=str(tmp_path / "status.json"))

    failed_status = tmp_path / "failed-status.json"
    failed_status.write_text(
        json.dumps(
            {
                "contexts": {
                    "demo": {"result": "failed", "reason": "source repo unreachable"}
                }
            }
        ),
        encoding="utf-8",
    )

    result = status("demo", registry_path=registry_path, status_path=str(failed_status))
    _assert_legible(result)
    assert result["state"] == "hydrate-failed"
    assert "source repo unreachable" in result["detail"]
    assert result["hydrated"] is True


def test_components_less_manifest_is_empty_manifest_not_empty_group(
    monkeypatch, tmp_path, empty_registry
):
    _isolate(monkeypatch, tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "config:\n  context:\n    dir: .context/empty\ncomponents: []\n", encoding="utf-8"
    )
    _hydrate(manifest, tmp_path)
    result = show(str(manifest), registry_path=empty_registry)
    _assert_legible(result)
    assert result["state"] == "empty-manifest"


def test_disabled_component_and_disabled_member(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "components:\n"
        "  - name: refs\n"
        "    files:\n"
        "      - a.md\n"
        "      # - b.md  # not yet ready\n"
        "  # - name: skipped\n"
        "  #   text: parked\n",
        encoding="utf-8",
    )
    disabled_component = show(f"{manifest}:skipped", registry_path=empty_registry)
    _assert_legible(disabled_component)
    assert disabled_component["state"] == "disabled"

    disabled_member = show(f"{manifest}:refs.b", registry_path=empty_registry)
    _assert_legible(disabled_member)
    assert disabled_member["state"] == "disabled"

    cat_result = cat_selector(f"{manifest}:refs.b")
    _assert_legible(cat_result)
    assert cat_result["state"] == "disabled"


def test_disabled_only_component_and_group(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "components:\n"
        "  - name: allOff\n"
        "    files:\n"
        "      # - a.md\n"
        "      # - b.md\n"
        "  - group: g\n"
        "    components:\n"
        "      # - name: a\n"
        "      #   text: hi\n",
        encoding="utf-8",
    )
    component_result = cat_selector(f"{manifest}:allOff")
    _assert_legible(component_result)
    assert component_result["state"] == "disabled-only"

    group_result = show(f"{manifest}:g", registry_path=empty_registry)
    _assert_legible(group_result)
    assert group_result["state"] == "disabled-only"


def test_unresolvable_source(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    manifest = tmp_path / "gone.yaml"
    manifest.write_text("components:\n  - name: m\n    text: hi\n", encoding="utf-8")
    registry_path = _registry(
        tmp_path / "registry.json",
        {"gone": {"targetDir": str(tmp_path), "manifest": {"source": "gone.yaml"}}},
    )
    manifest.unlink()

    for result in (
        show("gone", registry_path=registry_path),
        cat_selector("gone:m", registry_path=registry_path),
        links("gone", registry_path=registry_path),
        status("gone", registry_path=registry_path),
    ):
        _assert_legible(result)
        assert result["state"] == "unresolvable-source"


def test_stale_cache_from_hydration_older_than_manifest(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    drive = tmp_path / "drive"
    drive.mkdir()
    manifest = drive / "manifest.yaml"
    manifest.write_text(
        "config:\n  name: demo\n  context:\n    include-meta: true\n"
        "components:\n  - name: m\n    text: hi\n",
        encoding="utf-8",
    )
    _hydrate(manifest, drive)

    time.sleep(1.1)
    manifest.write_text(
        "config:\n  name: demo\n  context:\n    include-meta: true\n"
        "components:\n  - name: m\n    text: hi there\n",
        encoding="utf-8",
    )

    result = status(str(manifest), registry_path=empty_registry, cwd=str(drive))
    _assert_legible(result)
    assert result["state"] == "stale-cache"
    assert result["drift"]["hydration_stale"] is True
    assert result["drift"]["sources_changed"] == []


def test_dangling_reference(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    parent_dir = tmp_path / "parent"
    child_dir = tmp_path / "child"
    parent_dir.mkdir()
    child_dir.mkdir()

    child_note = child_dir / "note.md"
    child_note.write_text(
        "---\ntags:\n  - ctx/manifest\n---\n\n```yaml\ncomponents:\n  - name: main\n    text: hi\n```\n",
        encoding="utf-8",
    )
    (parent_dir / "manifest.yaml").write_text(
        "config:\n  name: parent\n  context:\n    dir: .context/parent\n    include-meta: true\n"
        "components:\n  - name: refs\n    files:\n      - ../child/note.md\n",
        encoding="utf-8",
    )
    registry_path = _registry(
        tmp_path / "registry.json",
        {"parent": {"targetDir": str(parent_dir), "manifest": {"source": "manifest.yaml"}}},
    )
    statuses = hydrate_contexts(["parent"], registry_path=registry_path, status_path=str(tmp_path / "s.json"))
    assert statuses[0].result == "hydrated"

    child_note.unlink()

    result = status("parent", registry_path=registry_path)
    _assert_legible(result)
    assert result["state"] == "dangling-reference"
    assert len(result["drift"]["references_gone"]) == 1

    out = links("parent", registry_path=registry_path, direction="out")
    assert Path(out["out"][0]["target_path"]).resolve() == child_note.resolve()


def test_cat_cannot_narrow_a_group_with_a_member_token(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "components:\n  - group: g\n    components:\n      - name: a\n        text: hi\n",
        encoding="utf-8",
    )
    result = cat_selector(f"{manifest}:g.nonexistent-member")
    _assert_legible(result)
    assert result["state"] == "not-found"


def test_cat_on_text_only_component_is_honest_not_disabled(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("components:\n  - name: m\n    text: inline content\n", encoding="utf-8")
    result = cat_selector(f"{manifest}:m")
    _assert_legible(result)
    assert result["state"] == "not-found"
    assert "text/prefix/suffix" in result["detail"]


def test_cat_bare_selector_needs_a_component(monkeypatch, tmp_path, empty_registry):
    _isolate(monkeypatch, tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("components:\n  - name: a\n    text: hi\n", encoding="utf-8")
    result = cat_selector(str(manifest))
    _assert_legible(result)
    assert result["state"] == "not-found"


def test_status_reports_bad_registry_path_legibly(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    result = status(None, registry_path=str(tmp_path / "does-not-exist.json"))
    assert result["state"] != "ok"
    assert result.get("detail")
