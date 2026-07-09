"""Serving marks: show renders them under members, cat draws them, links
aggregates them, status counts their drift (marks spec §4.3/4.5). Store
resolution comes from the conftest fake plugin; no live context-reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextualize.manifest import HydrateOverrides
from contextualize.manifest.hydrate import apply_hydration_plan, build_hydration_plan
from contextualize.serve import cat_selector, draw_substance, links, show, status

from conftest import STORE_TARGET, UNTIMED_TARGET

MARKED_MANIFEST = """config:
  context:
    dir: {ctx}
    include-meta: true

components:
  - name: reckoning
    files:
      # - skipped.md
      - path: "{target}"
        marks:
          # the case-study charge
          - at: 0:04            # opening
          - at: 0:25-1:00
            quote: |
              ...treat this as a case study.
            refs:
              - notes/op9f.md
          # - at: 0:09
"""


def _registry(path: Path, contexts: dict) -> str:
    path.write_text(json.dumps({"version": 1, "contexts": contexts}), encoding="utf-8")
    return str(path)


def _write_marked(tmp_path: Path, target: str = STORE_TARGET) -> Path:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        MARKED_MANIFEST.format(ctx=tmp_path / "ctx", target=target), encoding="utf-8"
    )
    return manifest


def _write_mini(tmp_path: Path, member_lines: str) -> Path:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "config:\n"
        f"  context:\n    dir: {tmp_path / 'ctx'}\n    include-meta: true\n"
        "components:\n"
        "  - name: reckoning\n"
        "    files:\n"
        f"{member_lines}",
        encoding="utf-8",
    )
    return manifest


def _hydrate(manifest: Path) -> None:
    plan = build_hydration_plan(
        str(manifest), overrides=HydrateOverrides(), cwd=str(manifest.parent)
    )
    apply_hydration_plan(plan)


@pytest.fixture
def empty_registry(tmp_path: Path) -> str:
    return _registry(tmp_path / "registry.json", {})


def _member(result: dict) -> dict:
    return result["node"]["members"]["files"][1]


def test_show_renders_marks_under_member(fake_store, tmp_path, empty_registry):
    manifest = _write_marked(tmp_path)
    result = show(f"{manifest}:reckoning", registry_path=empty_registry)
    assert result["state"] == "ok"

    members = result["node"]["members"]["files"]
    assert members[0]["disabled"] is True

    marks = members[1]["marks"]
    assert [m.get("disabled") for m in marks] == [False, False, True]

    point = marks[0]
    assert point["at"] == "0:04"
    assert point["comment"]["text"] == "the case-study charge"
    assert point["inline_comment"] == "opening"
    assert point["quote"] is False
    assert point["state"] == "ok"
    assert point["address"] == f"{STORE_TARGET}@0:04"

    ranged = marks[1]
    assert ranged["at"] == "0:25-1:00"
    assert ranged["quote"] is True
    assert ranged["refs"] == ["notes/op9f.md"]

    disabled = marks[2]
    assert disabled["raw"] == "- at: 0:09"


def test_show_member_carries_marks(fake_store, tmp_path, empty_registry):
    manifest = _write_marked(tmp_path)
    result = show(f"{manifest}:reckoning.1", registry_path=empty_registry)
    assert result["state"] == "ok"
    assert result["node"]["kind"] == "member"
    assert [m["at"] for m in result["node"]["marks"] if not m["disabled"]] == [
        "0:04",
        "0:25-1:00",
    ]


def test_show_accepts_mapping_form_marks(fake_store, tmp_path, empty_registry):
    manifest = _write_mini(
        tmp_path,
        f'      - path: "{STORE_TARGET}"\n'
        '        marks: {at: "0:04"}\n',
    )
    result = show(f"{manifest}:reckoning", registry_path=empty_registry)
    assert result["state"] == "ok"
    marks = result["node"]["members"]["files"][0]["marks"]
    assert [m["at"] for m in marks] == ["0:04"]
    assert marks[0]["state"] == "ok"
    assert marks[0]["address"] == f"{STORE_TARGET}@0:04"

    member = show(f"{manifest}:reckoning.1", registry_path=empty_registry)
    assert member["node"]["marks"][0]["at"] == "0:04"

    marked = show(f"{manifest}:reckoning.1.1", registry_path=empty_registry)
    assert marked["state"] == "ok"
    assert marked["node"]["kind"] == "mark"

    _hydrate(manifest)
    hydrated = show(f"{manifest}:reckoning", registry_path=empty_registry)
    assert hydrated["node"]["members"]["files"][0]["marks"][0]["state"] == "ok"


def test_show_mark_state_reflects_index_records(fake_store, tmp_path, empty_registry):
    manifest = _write_mini(
        tmp_path,
        f'      - path: "{STORE_TARGET}"\n'
        "        marks:\n"
        "          - at: 0:04\n"
        "          - at: 50:00\n",
    )
    _hydrate(manifest)
    result = show(f"{manifest}:reckoning", registry_path=empty_registry)
    marks = result["node"]["members"]["files"][0]["marks"]
    assert marks[0]["state"] == "ok"
    assert "detail" not in marks[0]
    assert marks[1]["state"] == "mark-beyond-duration"
    assert "duration 1:21" in marks[1]["detail"]


def test_show_addresses_one_mark(fake_store, tmp_path, empty_registry):
    manifest = _write_marked(tmp_path)
    result = show(f"{manifest}:reckoning.1.2", registry_path=empty_registry)
    assert result["state"] == "ok"
    node = result["node"]
    assert node["kind"] == "mark"
    assert node["at"] == "0:25-1:00"
    assert node["address"] == f"{STORE_TARGET}@0:25-1:00"
    assert node["quote"].startswith("...treat this as a case study.")
    assert node["refs"] == ["notes/op9f.md"]

    by_time = show(f"{manifest}:reckoning.1.0:04", registry_path=empty_registry)
    assert by_time["state"] == "ok"
    assert by_time["node"]["at"] == "0:04"
    assert by_time["node"]["inline_comment"] == "opening"


def test_mark_token_without_marks_is_legible(fake_store, tmp_path, empty_registry):
    manifest = _write_mini(tmp_path, f'      - path: "{STORE_TARGET}"\n')
    result = show(f"{manifest}:reckoning.1.1", registry_path=empty_registry)
    assert result["state"] == "not-found"
    assert "No marks on" in result["detail"]
    assert result["next_steps"]


def test_unknown_mark_token_names_the_valid_forms(fake_store, tmp_path, empty_registry):
    manifest = _write_marked(tmp_path)
    result = show(f"{manifest}:reckoning.1.9", registry_path=empty_registry)
    assert result["state"] == "not-found"
    assert "authored time or ordinal 1-2" in result["detail"]


def test_ambiguous_mark_time_resolves_first_and_says_so(
    fake_store, tmp_path, empty_registry
):
    manifest = _write_mini(
        tmp_path,
        f'      - path: "{STORE_TARGET}"\n'
        "        marks:\n"
        "          - at: 0:04\n"
        "          - at: 0:04\n",
    )
    result = show(f"{manifest}:reckoning.1.0:04", registry_path=empty_registry)
    assert result["state"] == "ok"
    assert "matches 2 marks" in result["detail"]


def test_cat_draws_pinned_mark_when_hydrated(fake_store, tmp_path, empty_registry):
    manifest = _write_marked(tmp_path)
    _hydrate(manifest)
    result = cat_selector(
        f"{manifest}:reckoning.1.2", registry_path=empty_registry, cwd=str(tmp_path)
    )
    assert result["state"] == "ok"
    assert result["payload"] == "copy"
    assert result["specs"] == [f"{STORE_TARGET}@0:25-1:00"]

    drawn = draw_substance(result)
    content = drawn["content"]
    assert f"--- {STORE_TARGET}@0:25-1:00 ---" in content
    assert "asr:\ntreat this as a case study for the work itself" in content
    assert "quote:\n...treat this as a case study." in content
    assert "refs:\n- notes/op9f.md" in content


def test_cat_draws_mark_live_when_unhydrated(fake_store, tmp_path, empty_registry):
    manifest = _write_marked(tmp_path)
    result = cat_selector(
        f"{manifest}:reckoning.1.1", registry_path=empty_registry, cwd=str(tmp_path)
    )
    assert result["state"] == "ok"
    assert result["payload"] == "pointer"

    drawn = draw_substance(result)
    assert "asr:\nso the reckoning note starts here" in drawn["content"]
    assert "opening" in drawn["content"]
    assert (STORE_TARGET, "0:04") in fake_store.resolve_calls


def test_cat_mark_adjacency_covers_the_authored_lines(
    fake_store, tmp_path, empty_registry
):
    manifest = _write_marked(tmp_path)
    result = cat_selector(
        f"{manifest}:reckoning.1.2",
        around=1,
        registry_path=empty_registry,
        cwd=str(tmp_path),
    )
    assert result["state"] == "ok"
    assert "0:25-1:00" in result["adjacency"]["text"]


def _assert_legible(result: dict) -> None:
    assert result.get("detail"), "designed state must name the condition"
    assert result.get("next_steps"), "designed state must name a path out"


def test_cat_quote_without_range_suggests_the_segment_boundary(
    fake_store, tmp_path, empty_registry
):
    manifest = _write_mini(
        tmp_path,
        f'      - path: "{STORE_TARGET}"\n'
        "        marks:\n"
        "          - at: 0:04\n"
        "            quote: |\n"
        "              solo\n",
    )
    _hydrate(manifest)
    result = cat_selector(
        f"{manifest}:reckoning.1.1", registry_path=empty_registry, cwd=str(tmp_path)
    )
    assert result["state"] == "mark-quote-requires-range"
    _assert_legible(result)
    assert "The containing segment ends at 0:20." in result["detail"]
    assert "content" not in draw_substance(result)

    shown = show(f"{manifest}:reckoning.1.1", registry_path=empty_registry)
    assert shown["state"] == "mark-quote-requires-range"
    _assert_legible(shown)


def test_cat_mark_beyond_duration_from_index_and_live(
    fake_store, tmp_path, empty_registry
):
    manifest = _write_mini(
        tmp_path,
        f'      - path: "{STORE_TARGET}"\n'
        "        marks:\n"
        "          - at: 50:00\n",
    )
    live = draw_substance(
        cat_selector(
            f"{manifest}:reckoning.1.1", registry_path=empty_registry, cwd=str(tmp_path)
        )
    )
    assert live["state"] == "mark-beyond-duration"
    _assert_legible(live)
    assert "content" not in live

    _hydrate(manifest)
    pinned = cat_selector(
        f"{manifest}:reckoning.1.1", registry_path=empty_registry, cwd=str(tmp_path)
    )
    assert pinned["state"] == "mark-beyond-duration"
    _assert_legible(pinned)

    shown = show(f"{manifest}:reckoning.1.1", registry_path=empty_registry)
    assert shown["state"] == "mark-beyond-duration"
    _assert_legible(shown)


def test_cat_marks_on_untimed_targets(fake_store, tmp_path, empty_registry):
    (tmp_path / "plain.md").write_text("prose\n", encoding="utf-8")
    for member_lines in (
        '      - path: plain.md\n        marks:\n          - at: 0:04\n',
        f'      - path: "{UNTIMED_TARGET}"\n        marks:\n          - at: 0:04\n',
    ):
        manifest = _write_mini(tmp_path, member_lines)
        result = draw_substance(
            cat_selector(
                f"{manifest}:reckoning.1.1",
                registry_path=empty_registry,
                cwd=str(tmp_path),
            )
        )
        assert result["state"] == "marks-on-untimed-target"
        _assert_legible(result)
        assert "content" not in result

        _hydrate(manifest)
        shown = show(f"{manifest}:reckoning.1.1", registry_path=empty_registry)
        assert shown["state"] == "marks-on-untimed-target"
        _assert_legible(shown)


def test_cat_marks_require_single_document(fake_store, tmp_path, empty_registry):
    manifest = _write_mini(
        tmp_path,
        '      - path: "store:all"\n        marks:\n          - at: 0:04\n',
    )
    result = draw_substance(
        cat_selector(
            f"{manifest}:reckoning.1.1", registry_path=empty_registry, cwd=str(tmp_path)
        )
    )
    assert result["state"] == "marks-require-single-document"
    _assert_legible(result)
    assert "content" not in result

    _hydrate(manifest)
    shown = show(f"{manifest}:reckoning.1.1", registry_path=empty_registry)
    assert shown["state"] == "marks-require-single-document"
    _assert_legible(shown)


def test_mark_params_unsupported_across_cat_index_and_show(
    fake_store, tmp_path, empty_registry
):
    manifest = _write_mini(
        tmp_path,
        f'      - path: "{STORE_TARGET}?after=2w"\n'
        "        marks:\n"
        "          - at: 0:04\n",
    )
    live = draw_substance(
        cat_selector(
            f"{manifest}:reckoning.1.1", registry_path=empty_registry, cwd=str(tmp_path)
        )
    )
    assert live["state"] == "mark-params-unsupported"
    _assert_legible(live)
    assert "content" not in live

    _hydrate(manifest)
    index = json.loads((tmp_path / "ctx" / "index.json").read_text(encoding="utf-8"))
    marks_entries = [
        e
        for e in index["components"]["reckoning"]
        if isinstance(e, dict) and e.get("kind") == "marks"
    ]
    assert marks_entries[0]["marks"][0]["state"] == "mark-params-unsupported"

    shown = show(f"{manifest}:reckoning.1.1", registry_path=empty_registry)
    assert shown["state"] == "mark-params-unsupported"
    _assert_legible(shown)


def test_cat_mark_invalid_time_is_legible(fake_store, tmp_path, empty_registry):
    manifest = _write_mini(
        tmp_path,
        f'      - path: "{STORE_TARGET}"\n'
        "        marks:\n"
        "          - at: nonsense\n",
    )
    result = cat_selector(
        f"{manifest}:reckoning.1.1", registry_path=empty_registry, cwd=str(tmp_path)
    )
    assert result["state"] == "mark-invalid-time"
    _assert_legible(result)


def test_mark_invalid_item_is_legible(fake_store, tmp_path, empty_registry):
    manifest = _write_mini(
        tmp_path,
        f'      - path: "{STORE_TARGET}"\n'
        "        marks:\n"
        "          - 4:12\n",
    )
    result = show(f"{manifest}:reckoning.1.1", registry_path=empty_registry)
    assert result["state"] == "mark-invalid"
    _assert_legible(result)
    assert "mapping" in result["detail"]
    assert "mapping" in result["next_steps"][0]

    catted = cat_selector(
        f"{manifest}:reckoning.1.1", registry_path=empty_registry, cwd=str(tmp_path)
    )
    assert catted["state"] == "mark-invalid"
    _assert_legible(catted)
    assert "content" not in draw_substance(catted)

    _hydrate(manifest)
    shown = show(f"{manifest}:reckoning.1.1", registry_path=empty_registry)
    assert shown["state"] == "mark-invalid"
    _assert_legible(shown)


def test_scalar_marks_value_degrades_to_one_invalid_mark(
    fake_store, tmp_path, empty_registry
):
    manifest = _write_mini(
        tmp_path,
        f'      - path: "{STORE_TARGET}"\n'
        '        marks: "0:04"\n',
    )
    result = show(f"{manifest}:reckoning", registry_path=empty_registry)
    marks = result["node"]["members"]["files"][0]["marks"]
    assert [m["state"] for m in marks] == ["mark-invalid"]
    assert marks[0]["at"] == "0:04"
    assert marks[0]["detail"]


def test_cat_accepts_bare_target_addresses(fake_store, tmp_path, empty_registry):
    address = f"{STORE_TARGET}@0:25-1:00"
    result = cat_selector(address, registry_path=empty_registry, cwd=str(tmp_path))
    assert result["state"] == "ok"
    assert result["origin"]["kind"] == "target"
    assert result["specs"] == [address]

    drawn = draw_substance(result)
    assert "treat this as a case study for the work itself" in drawn["content"]
    assert "mood kit day begins" not in drawn["content"]


def test_cli_cat_draws_a_mark_selector(fake_store, tmp_path):
    from click.testing import CliRunner

    from contextualize import cli

    manifest = _write_marked(tmp_path)
    _hydrate(manifest)
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cat", f"{manifest}:reckoning.1.2"])
    assert result.exit_code == 0, result.output
    assert "quote:\n...treat this as a case study." in result.output
    assert "refs:\n- notes/op9f.md" in result.output


def test_cli_cat_mark_state_exits_with_the_detail(fake_store, tmp_path):
    from click.testing import CliRunner

    from contextualize import cli

    manifest = _write_mini(
        tmp_path,
        f'      - path: "{STORE_TARGET}"\n'
        "        marks:\n"
        "          - at: 50:00\n",
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cat", f"{manifest}:reckoning.1.1"])
    assert result.exit_code != 0
    assert "beyond" in result.output


def _registered_marked_context(tmp_path: Path) -> tuple[str, Path]:
    drive = tmp_path / "drive"
    drive.mkdir()
    manifest = drive / "manifest.yaml"
    manifest.write_text(
        MARKED_MANIFEST.format(ctx=drive / ".context" / "annot", target=STORE_TARGET),
        encoding="utf-8",
    )
    _hydrate(manifest)
    registry = _registry(
        tmp_path / "registry.json",
        {"annot": {"targetDir": str(drive), "manifest": {"source": "manifest.yaml"}}},
    )
    return registry, manifest


def test_links_target_aggregates_registered_marks(fake_store, tmp_path):
    registry, _manifest = _registered_marked_context(tmp_path)
    result = links(STORE_TARGET, registry_path=registry, cwd=str(tmp_path))

    assert result["state"] == "ok"
    assert result["origin"]["kind"] == "target"
    assert result["target"]["base"] == STORE_TARGET
    assert result["out"] is None and result["shared"] is None

    edges = result["in"]
    assert [e["authored"] for e in edges] == ["0:04", "0:25-1:00"]
    assert all(e["source_context"] == "annot" for e in edges)
    assert all(e["component"] == "reckoning" for e in edges)
    assert edges[0]["address"] == f"{STORE_TARGET}@0:04"
    assert edges[0]["inline_comment"] == "opening"

    scope = result["coverage"]["tag_scope"]
    assert scope["tags"] == ["ctx/manifest"]
    assert scope["skipped"] == "no links.discovery root configured"


def test_links_target_span_query_filters_overlap(fake_store, tmp_path):
    registry, _manifest = _registered_marked_context(tmp_path)
    result = links(f"{STORE_TARGET}@0:20-0:30", registry_path=registry, cwd=str(tmp_path))
    assert result["target"]["start_s"] == 20.0
    assert [e["authored"] for e in result["in"]] == ["0:25-1:00"]


def test_links_target_names_the_empty_answer(fake_store, tmp_path, empty_registry):
    result = links(UNTIMED_TARGET, registry_path=empty_registry, cwd=str(tmp_path))
    assert result["state"] == "ok"
    assert result["in"] == []
    assert result["detail"] == "No marks address this target."


def _write_discovery_config(tmp_path: Path, root: Path) -> None:
    config_dir = tmp_path / "xdg-config" / "contextualize"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(
        f"links:\n  discovery:\n    root: {root}\n    tags:\n      - ctx/manifest\n",
        encoding="utf-8",
    )


def _mock_zk(monkeypatch, notes: list[Path]) -> None:
    import subprocess

    def _run(args, **kwargs):
        assert args == ["zk", "list", "--tag", "ctx/manifest*", "--format", "json", "--quiet"]
        payload = [{"absPath": str(path), "path": path.name} for path in notes]
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("contextualize.manifest.contexts.subprocess.run", _run)


def test_links_tag_scope_discovers_marked_notes(fake_store, monkeypatch, tmp_path):
    registry, registered_manifest = _registered_marked_context(tmp_path)
    notes_root = tmp_path / "notes"
    notes_root.mkdir()

    note_sub = notes_root / "annotated.md"
    note_sub.write_text(
        "---\ntags:\n  - ctx/manifest/voice\n---\n\n"
        "```yaml\ncomponents:\n  - name: pulls\n    files:\n"
        f'      - path: "{STORE_TARGET}"\n'
        "        marks:\n"
        "          - at: 0:30-0:45\n```\n",
        encoding="utf-8",
    )
    note_bare = notes_root / "bare.md"
    note_bare.write_text(
        "---\ntags:\n  - ctx/manifest\n---\n\nprose only\n", encoding="utf-8"
    )

    _write_discovery_config(tmp_path, notes_root)
    _mock_zk(monkeypatch, [note_sub, note_bare, registered_manifest])

    result = links(STORE_TARGET, registry_path=registry, cwd=str(tmp_path))

    note_edges = [e for e in result["in"] if e.get("source_note")]
    assert len(note_edges) == 1
    assert note_edges[0]["source_note"] == str(note_sub)
    assert note_edges[0]["component"] == "pulls"
    assert note_edges[0]["authored"] == "0:30-0:45"

    registered_edges = [e for e in result["in"] if e.get("source_context") == "annot"]
    assert len(registered_edges) == 2

    scope = result["coverage"]["tag_scope"]
    assert scope["root"] == str(notes_root)
    assert scope["notes_scanned"] == 2
    assert scope["notes_with_manifest"] == 1
    assert "skipped" not in scope


def test_links_tag_scope_degrades_without_zk(fake_store, monkeypatch, tmp_path):
    registry, _manifest = _registered_marked_context(tmp_path)
    notes_root = tmp_path / "notes"
    notes_root.mkdir()
    _write_discovery_config(tmp_path, notes_root)

    def _run(args, **kwargs):
        raise FileNotFoundError("zk")

    monkeypatch.setattr("contextualize.manifest.contexts.subprocess.run", _run)

    result = links(STORE_TARGET, registry_path=registry, cwd=str(tmp_path))
    assert result["state"] == "ok"
    assert "zk" in result["coverage"]["tag_scope"]["skipped"]
    assert [e["source_context"] for e in result["in"]] == ["annot", "annot"]


def test_manifest_tag_matches_subtags_once():
    from contextualize.manifest.source import frontmatter_has_manifest_tag

    sub = "---\ntags:\n  - ctx/manifest/voice\n---\nbody\n"
    assert frontmatter_has_manifest_tag(sub) is True
    assert frontmatter_has_manifest_tag(sub, tag=["ctx/manifest"]) is True
    festo = "---\ntags:\n  - ctx/manifesto\n---\nbody\n"
    assert frontmatter_has_manifest_tag(festo) is False


def test_links_out_carries_mark_ref_edges(fake_store, tmp_path):
    registry, _manifest = _registered_marked_context(tmp_path)
    result = links("annot", registry_path=registry, direction="out", cwd=str(tmp_path))
    mark_edges = [e for e in result["out"] if e.get("form") == "mark"]
    assert [e["spec"] for e in mark_edges] == ["notes/op9f.md"]
    assert mark_edges[0]["mark_address"] == f"{STORE_TARGET}@0:25-1:00"


CURRENT_CAPTURE = {
    "id": 3,
    "model": "fake-asr-1",
    "capturedAt": "2026-07-07T12:40:00Z",
    "active": True,
}
SUPERSEDED_CAPTURE = {
    "id": 9,
    "model": "fake-asr-2",
    "capturedAt": "2026-07-08T09:00:00Z",
    "active": True,
}


def _stub_reader(monkeypatch, tmp_path: Path, captures: list[dict]) -> Path:
    payload = tmp_path / "captures.json"
    payload.write_text(json.dumps(captures), encoding="utf-8")
    script = tmp_path / "context-reader-stub"
    script.write_text(f"#!/bin/sh\ncat '{payload}'\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("CONTEXTUALIZE_READER_COMMAND", str(script))
    return payload


def test_status_counts_mark_drift(fake_store, monkeypatch, tmp_path):
    registry, _manifest = _registered_marked_context(tmp_path)
    payload = _stub_reader(monkeypatch, tmp_path, [CURRENT_CAPTURE])

    fresh = status("annot", registry_path=registry, cwd=str(tmp_path))
    assert fresh["state"] == "ok"
    assert fresh["drift"]["marks_drifted"] == []
    assert fresh["drift"]["marks_unchecked"] is None

    payload.write_text(json.dumps([SUPERSEDED_CAPTURE]), encoding="utf-8")
    drifted = status("annot", registry_path=registry, cwd=str(tmp_path))
    assert drifted["state"] == "transcript-drift"
    assert drifted["drift"]["any"] is True
    assert drifted["detail"]
    assert drifted["next_steps"]
    marks = drifted["drift"]["marks_drifted"]
    assert [item["address"] for item in marks] == [
        f"{STORE_TARGET}@0:04",
        f"{STORE_TARGET}@0:25-1:00",
    ]
    assert "superseded by fake-asr-2" in marks[0]["reason"]
    assert "2 mark(s) drifted" in drifted["detail"]


def test_status_mark_drift_check_is_guarded(fake_store, monkeypatch, tmp_path):
    registry, _manifest = _registered_marked_context(tmp_path)
    monkeypatch.setenv("CONTEXTUALIZE_READER_COMMAND", str(tmp_path / "no-such-binary"))

    result = status("annot", registry_path=registry, cwd=str(tmp_path))
    assert result["state"] == "ok"
    assert result["drift"]["marks_drifted"] == []
    assert "unavailable" in result["drift"]["marks_unchecked"]
    assert "Mark drift unchecked" in result["detail"]


def test_status_registry_wide_counts_marks_drifted(fake_store, monkeypatch, tmp_path):
    registry, _manifest = _registered_marked_context(tmp_path)
    _stub_reader(monkeypatch, tmp_path, [SUPERSEDED_CAPTURE])

    result = status(None, registry_path=registry, cwd=str(tmp_path))
    assert result["drift_summary"] == {"drifted": 1, "total": 1, "marks_drifted": 2}
    assert result["contexts"][0]["state"] == "transcript-drift"


def test_show_mark_state_is_subprocess_free(fake_store, monkeypatch, tmp_path):
    registry, manifest = _registered_marked_context(tmp_path)
    monkeypatch.setenv("CONTEXTUALIZE_READER_COMMAND", str(tmp_path / "no-such-binary"))

    result = show(f"{manifest}:reckoning", registry_path=registry)
    marks = [m for m in _member(result)["marks"] if not m["disabled"]]
    assert [m["state"] for m in marks] == ["ok", "ok"]
