"""Hydrating `marks:` — sibling units, index records, edges, designed states.

Spec: .plans/2026-07-09-marks.md §4.4. Marks render as a `.marks.md` unit
beside the marked member; per-mark asr slices resolve through the plugin
path; authoring mistakes become per-mark states, never a failed hydration.
"""

from __future__ import annotations

import json
from pathlib import Path

from contextualize.manifest.hydrate import (
    HydrateOverrides,
    apply_hydration_plan,
    build_hydration_plan,
    build_hydration_plan_data,
)

from conftest import STORE_TARGET, UNTIMED_TARGET, flat_transcript

MARKED_MANIFEST = """config:
  context:
    dir: {ctx}
    include-meta: true
    path-strategy: by-component

components:
  - name: reckoning
    files:
      - path: "{target}"
        marks:
          # the case-study charge
          - at: 0:04            # opening
          - at: 0:25-1:00
            quote: |
              ...treat this as a case study.
              Two prompts.
            refs:
              - notes/op9f.md
              - dev/mood-kit
          # - at: 0:09
          - at: 1:05
"""


def _hydrate(tmp_path: Path, manifest_text: str) -> Path:
    context_dir = tmp_path / "ctx"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(manifest_text, encoding="utf-8")
    plan = build_hydration_plan(
        str(manifest), overrides=HydrateOverrides(), cwd=str(tmp_path)
    )
    apply_hydration_plan(plan)
    return context_dir


def _hydrate_marked(tmp_path: Path, target: str = STORE_TARGET) -> Path:
    return _hydrate(
        tmp_path,
        MARKED_MANIFEST.format(ctx=tmp_path / "ctx", target=target),
    )


def _index(context_dir: Path) -> dict:
    return json.loads((context_dir / "index.json").read_text(encoding="utf-8"))


def _entries(context_dir: Path, component: str = "reckoning") -> list[dict]:
    return _index(context_dir)["components"][component]


def _marks_entry(context_dir: Path, component: str = "reckoning") -> dict:
    entries = [e for e in _entries(context_dir, component) if e.get("kind") == "marks"]
    assert len(entries) == 1
    return entries[0]


def _mini_manifest(ctx: Path, member_lines: str) -> str:
    return (
        "config:\n"
        f"  context:\n    dir: {ctx}\n    include-meta: true\n"
        "components:\n"
        "  - name: reckoning\n"
        "    files:\n"
        f"{member_lines}"
    )


def test_marks_hydrate_as_sibling_unit(fake_store, tmp_path):
    context_dir = _hydrate_marked(tmp_path)
    entries = _entries(context_dir)
    member = entries[0]
    marks = _marks_entry(context_dir)

    assert member["context_path"].endswith("12-34-52.m4a.md")
    assert marks["context_path"] == member["context_path"].replace(
        ".m4a.md", ".m4a.marks.md"
    )
    assert marks["member_context_path"] == member["context_path"]
    assert marks["target"] == STORE_TARGET

    member_text = (context_dir / member["context_path"]).read_text(encoding="utf-8")
    assert member_text == flat_transcript(fake_store.segments)

    unit_text = (context_dir / marks["context_path"]).read_text(encoding="utf-8")
    assert f"--- {STORE_TARGET}@0:04 · opening ---" in unit_text
    assert "the case-study charge" in unit_text
    assert "asr:\nso the reckoning note starts here" in unit_text
    assert (
        "asr:\ntreat this as a case study for the work itself\n"
        "two prompts and the supplement around them\n"
        "\nquote:\n...treat this as a case study.\nTwo prompts." in unit_text
    )
    assert "refs:\n- notes/op9f.md\n- dev/mood-kit" in unit_text
    assert "0:09" not in unit_text


def test_index_records_per_mark(fake_store, tmp_path):
    context_dir = _hydrate_marked(tmp_path)
    marks = _marks_entry(context_dir)["marks"]
    assert [m["order"] for m in marks] == [0, 1, 2]

    point = marks[0]
    assert point["address"] == f"{STORE_TARGET}@0:04"
    assert point["authored"] == "0:04"
    assert point["start_s"] == 4.0
    assert point["end_s"] is None
    assert point["covered_start_s"] == 0.0
    assert point["covered_end_s"] == 20.0
    assert point["asr"] == "so the reckoning note starts here"
    assert point["capture"]["model"] == "fake-asr-1"
    assert point["quote"] is False
    assert point["refs"] == []
    assert point["comment"] == "the case-study charge"
    assert point["inline_comment"] == "opening"
    assert point["state"] == "ok"

    ranged = marks[1]
    assert ranged["authored"] == "0:25-1:00"
    assert ranged["start_s"] == 25.0
    assert ranged["end_s"] == 60.0
    assert ranged["covered_start_s"] == 20.0
    assert ranged["covered_end_s"] == 66.0
    assert ranged["quote"] is True
    assert ranged["refs"] == ["notes/op9f.md", "dev/mood-kit"]
    assert ranged["state"] == "ok"

    sexagesimal = marks[2]
    assert sexagesimal["authored"] == "1:05"
    assert sexagesimal["start_s"] == 65.0
    assert sexagesimal["asr"] == "two prompts and the supplement around them"


def test_mark_line_spans_address_unit_blocks(fake_store, tmp_path):
    context_dir = _hydrate_marked(tmp_path)
    entry = _marks_entry(context_dir)
    unit_lines = (
        (context_dir / entry["context_path"])
        .read_text(encoding="utf-8")
        .split("\n")
    )
    previous_end = 0
    for record in entry["marks"]:
        block = unit_lines[record["line_start"] - 1 : record["line_end"]]
        assert block[0].startswith("--- ")
        assert record["authored"] in block[0]
        assert record["line_start"] > previous_end
        previous_end = record["line_end"]


def test_mark_refs_recorded_as_edges_not_payload(fake_store, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "op9f.md").write_text("the op9f note\n", encoding="utf-8")
    context_dir = _hydrate_marked(tmp_path)

    edges = [
        e
        for e in _index(context_dir)["references"]["out"]
        if e.get("form") == "mark"
    ]
    assert len(edges) == 2
    by_spec = {e["spec"]: e for e in edges}

    local = by_spec["notes/op9f.md"]
    assert local["detected_via"] == "marks-key"
    assert local["mark_address"] == f"{STORE_TARGET}@0:25-1:00"
    assert local["target_path"] == str((notes / "op9f.md").resolve())
    assert "target" not in local

    loose = by_spec["dev/mood-kit"]
    assert loose["target"] == "dev/mood-kit"
    assert "target_path" not in loose

    assert not [p for p in context_dir.rglob("*") if "op9f" in p.name]


def test_point_mark_with_quote_is_a_state_not_a_crash(fake_store, tmp_path):
    context_dir = _hydrate(
        tmp_path,
        _mini_manifest(
            tmp_path / "ctx",
            f'      - path: "{STORE_TARGET}"\n'
            "        marks:\n"
            "          - at: 0:04\n"
            "            quote: |\n"
            "              solo line\n",
        ),
    )
    record = _marks_entry(context_dir)["marks"][0]
    assert record["state"] == "mark-quote-requires-range"
    assert record["quote"] is True
    assert record["asr"] is None
    assert record["covered_start_s"] == 0.0
    assert record["covered_end_s"] == 20.0
    unit_text = (
        context_dir / _marks_entry(context_dir)["context_path"]
    ).read_text(encoding="utf-8")
    assert "state: mark-quote-requires-range" in unit_text


def test_mark_beyond_duration_carries_plugin_state(fake_store, tmp_path):
    context_dir = _hydrate(
        tmp_path,
        _mini_manifest(
            tmp_path / "ctx",
            f'      - path: "{STORE_TARGET}"\n'
            "        marks:\n"
            "          - at: 50:00\n",
        ),
    )
    record = _marks_entry(context_dir)["marks"][0]
    assert record["state"] == "mark-beyond-duration"
    assert record["authored"] == "50:00"
    assert "duration 1:21" in record["detail"]
    unit_text = (
        context_dir / _marks_entry(context_dir)["context_path"]
    ).read_text(encoding="utf-8")
    assert "state: mark-beyond-duration" in unit_text
    assert "duration 1:21" in unit_text


def test_marks_on_untimed_store_target(fake_store, tmp_path):
    context_dir = _hydrate(
        tmp_path,
        _mini_manifest(
            tmp_path / "ctx",
            f'      - path: "{UNTIMED_TARGET}"\n'
            "        marks:\n"
            "          - at: 0:04\n",
        ),
    )
    record = _marks_entry(context_dir)["marks"][0]
    assert record["state"] == "marks-on-untimed-target"


def test_marks_on_local_file_target(fake_store, tmp_path):
    (tmp_path / "plain.md").write_text("prose\n", encoding="utf-8")
    context_dir = _hydrate(
        tmp_path,
        _mini_manifest(
            tmp_path / "ctx",
            "      - path: plain.md\n"
            "        marks:\n"
            "          - at: 0:04\n",
        ),
    )
    entries = _entries(context_dir)
    assert entries[0]["source_type"] == "local"
    record = _marks_entry(context_dir)["marks"][0]
    assert record["state"] == "marks-on-untimed-target"
    unit_text = (
        context_dir / _marks_entry(context_dir)["context_path"]
    ).read_text(encoding="utf-8")
    assert "Marks need timed media" in unit_text


def test_marks_on_multi_document_store_query(fake_store, tmp_path):
    context_dir = _hydrate(
        tmp_path,
        _mini_manifest(
            tmp_path / "ctx",
            '      - path: "store:all"\n'
            "        marks:\n"
            "          - at: 0:04\n",
        ),
    )
    record = _marks_entry(context_dir)["marks"][0]
    assert record["state"] == "marks-require-single-document"


def test_marks_on_local_directory(fake_store, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("a\n", encoding="utf-8")
    (docs / "b.md").write_text("b\n", encoding="utf-8")
    context_dir = _hydrate(
        tmp_path,
        _mini_manifest(
            tmp_path / "ctx",
            "      - path: docs\n"
            "        marks:\n"
            "          - at: 0:04\n",
        ),
    )
    record = _marks_entry(context_dir)["marks"][0]
    assert record["state"] == "marks-require-single-document"


def test_unparseable_mark_time_is_a_state(fake_store, tmp_path):
    context_dir = _hydrate(
        tmp_path,
        _mini_manifest(
            tmp_path / "ctx",
            f'      - path: "{STORE_TARGET}"\n'
            "        marks:\n"
            "          - at: nonsense\n",
        ),
    )
    record = _marks_entry(context_dir)["marks"][0]
    assert record["state"] == "mark-invalid-time"
    assert record["address"] is None


def test_marks_do_not_fracture_member_resolution(fake_store, tmp_path):
    manifest_text = (
        "config:\n"
        f"  context:\n    dir: {tmp_path / 'ctx'}\n    include-meta: true\n"
        "components:\n"
        "  - name: first\n"
        "    files:\n"
        f'      - path: "{STORE_TARGET}"\n'
        "        marks:\n"
        "          - at: 0:04\n"
        "  - name: second\n"
        "    files:\n"
        f'      - path: "{STORE_TARGET}"\n'
        "        marks:\n"
        "          - at: 0:04\n"
        "          - at: 0:25-1:00\n"
    )
    context_dir = _hydrate(tmp_path, manifest_text)
    assert fake_store.resolve_calls.count((STORE_TARGET, None)) == 1
    assert fake_store.resolve_calls.count((STORE_TARGET, "0:04")) == 1
    assert fake_store.resolve_calls.count((STORE_TARGET, "0:25-1:00")) == 1
    assert len(_marks_entry(context_dir, "first")["marks"]) == 1
    assert len(_marks_entry(context_dir, "second")["marks"]) == 2


def test_data_manifest_keeps_marks(fake_store, tmp_path):
    data = {
        "config": {
            "context": {"dir": str(tmp_path / "ctx"), "include-meta": True}
        },
        "components": [
            {
                "name": "reckoning",
                "files": [
                    {
                        "path": STORE_TARGET,
                        "marks": [{"at": "0:04"}],
                    }
                ],
            }
        ],
    }
    plan = build_hydration_plan_data(
        data,
        manifest_cwd=str(tmp_path),
        overrides=HydrateOverrides(),
        cwd=str(tmp_path),
    )
    manifest_text = next(
        text
        for path, text in plan.files_to_write
        if path.name == "manifest.yaml"
    )
    assert "marks:" in manifest_text
    assert "0:04" in manifest_text

    index_text = next(
        text for path, text in plan.files_to_write if path.name == "index.json"
    )
    marks_entries = [
        entry
        for entry in json.loads(index_text)["components"]["reckoning"]
        if entry.get("kind") == "marks"
    ]
    assert len(marks_entries) == 1
    record = marks_entries[0]["marks"][0]
    assert record["state"] == "ok"
    assert record["comment"] is None
    assert record["inline_comment"] is None


def test_bare_address_member_hydrates_excerpt(fake_store, tmp_path):
    context_dir = _hydrate(
        tmp_path,
        _mini_manifest(
            tmp_path / "ctx",
            f'      - "{STORE_TARGET}@0:25-1:00"\n',
        ),
    )
    entry = _entries(context_dir)[0]
    assert entry["source_type"] == "plugin:store"
    assert entry["context_path"].endswith("12-34-52.m4a@0-25-1-00.md")
    assert entry["mark"] == {
        "authored": "0:25-1:00",
        "start_s": 25.0,
        "end_s": 60.0,
        "covered_start_s": 20.0,
        "covered_end_s": 66.0,
        "segment_count": 2,
        "capture": dict(fake_store.capture),
        "state": "ok",
    }
    excerpt = (context_dir / entry["context_path"]).read_text(encoding="utf-8")
    assert excerpt == (
        "treat this as a case study for the work itself\n"
        "two prompts and the supplement around them"
    )


def test_bare_address_beyond_duration_is_a_state_member(fake_store, tmp_path):
    context_dir = _hydrate(
        tmp_path,
        _mini_manifest(
            tmp_path / "ctx",
            f'      - "{STORE_TARGET}@50:00"\n',
        ),
    )
    entry = _entries(context_dir)[0]
    assert entry["mark"]["state"] == "mark-beyond-duration"
    content = (context_dir / entry["context_path"]).read_text(encoding="utf-8")
    assert "beyond this recording" in content


def test_set_member_marks_hydrate(fake_store, tmp_path):
    manifest_text = (
        "config:\n"
        f"  context:\n    dir: {tmp_path / 'ctx'}\n    include-meta: true\n"
        "components:\n"
        "  - set: threads\n"
        "    files:\n"
        f'      - path: "{STORE_TARGET}"\n'
        "        marks:\n"
        "          - at: 0:25-0:40\n"
        "            quote: |\n"
        "              ...treat this as a case study.\n"
        "            refs:\n"
        "              - notes/op9f.md\n"
    )
    context_dir = _hydrate(tmp_path, manifest_text)
    entries = _entries(context_dir, "threads")
    assert [entry.get("kind") for entry in entries] == ["set", "marks"]

    marks_entry = entries[1]
    assert marks_entry["member_context_path"] == entries[0]["context_path"]
    assert marks_entry["target"] == STORE_TARGET
    record = marks_entry["marks"][0]
    assert record["state"] == "ok"
    assert record["address"] == f"{STORE_TARGET}@0:25-0:40"
    assert record["asr"] == "treat this as a case study for the work itself"

    unit_text = (context_dir / marks_entry["context_path"]).read_text(encoding="utf-8")
    assert "asr:\ntreat this as a case study for the work itself" in unit_text
    assert "quote:\n...treat this as a case study." in unit_text

    edges = [
        e for e in _index(context_dir)["references"]["out"] if e.get("form") == "mark"
    ]
    assert [e["spec"] for e in edges] == ["notes/op9f.md"]
    assert edges[0]["mark_address"] == f"{STORE_TARGET}@0:25-0:40"


def test_bare_addresses_fuse_into_set(fake_store, tmp_path):
    range_address = f"{STORE_TARGET}@0:25-1:00"
    point_address = f"{STORE_TARGET}@0:04"
    manifest_text = (
        "config:\n"
        f"  context:\n    dir: {tmp_path / 'ctx'}\n    include-meta: true\n"
        "components:\n"
        "  - set: threads\n"
        "    files:\n"
        f'      - "{range_address}"\n'
        f'      - "{point_address}"\n'
    )
    context_dir = _hydrate(tmp_path, manifest_text)
    entry = _entries(context_dir, "threads")[0]
    assert entry["kind"] == "set"
    assert [part["key"] for part in entry["parts"]] == [
        range_address,
        point_address,
    ]
    assert entry["parts"][1]["line_start"] > entry["parts"][0]["line_end"]

    fused = (context_dir / entry["context_path"]).read_text(encoding="utf-8")
    assert f"--- {range_address}" in fused
    assert f"--- {point_address}" in fused
    assert "so the reckoning note starts here" in fused
    assert "mood kit day begins" not in fused
