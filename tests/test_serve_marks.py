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
    assert marks[1]["state"] == "mark-beyond-duration"


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
