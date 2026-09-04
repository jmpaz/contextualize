from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from contextualize import cli
from contextualize.manifest.edition import (
    AUTHORED_EDITION_SCHEMA_VERSION,
    AuthoredPortal,
    compile_authored_manifest,
    compile_authored_registry,
)


def test_compiler_preserves_authored_world_and_voice_spans(tmp_path: Path) -> None:
    existing = tmp_path / "existing.md"
    existing.write_text("material", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """config:
  name: alpha
  root: .
  comment: Entrance framing
components:
  # speech as theory
  - group: voice-survey
    prefix: Read this as a designed sequence.
    components:
      - name: address-analogy  # principal strand
        text: The framing comes before its inventory.
        files:
          - path: "store:voice/2026-07-07/12-34-52.m4a"
            range: 1-20
            marks:
              # exact case-study span
              - at: 4:12-5:00  # central turn
                quote: a precise quotation
      # - name: omitted-strand
      #   files:
      #     - missing.md
  - name: materials
    files:
      - existing.md
      # - intentionally-omitted.md
""",
        encoding="utf-8",
    )

    edition = compile_authored_manifest(
        manifest,
        context_name="alpha",
        compiled_at="2026-07-12T12:00:00Z",
    )
    payload = edition.to_dict()

    assert payload["schemaVersion"] == AUTHORED_EDITION_SCHEMA_VERSION
    assert payload["context"]["authority"] == "file-backed-authored"
    assert payload["context"]["hydration"] == "optional-projection"
    assert payload["positions"][0]["framing"] == [
        {"kind": "comment", "text": "Entrance framing"}
    ]

    by_locator = {
        locator: position
        for position in payload["positions"]
        for locator in position["locators"]
    }
    group = by_locator["alpha:voice-survey"]
    strand = by_locator["alpha:voice-survey/address-analogy"]
    omitted = by_locator["alpha:voice-survey/omitted-strand"]
    assert group["framing"] == [
        {"kind": "source-comment", "text": "speech as theory", "line_start": 6, "line_end": 6},
        {"kind": "prefix", "text": "Read this as a designed sequence."},
    ]
    assert strand["parentId"] == group["id"]
    assert strand["framing"][-1] == {
        "kind": "text",
        "text": "The framing comes before its inventory.",
    }
    assert omitted["disabled"] is True
    assert omitted["raw"].startswith("- name: omitted-strand")

    voice = next(portal for portal in payload["portals"] if portal["role"] == "material")
    assert voice["positionId"] == strand["id"]
    assert voice["reverse"]["positionStableId"] == strand["stableId"]
    assert "voice/2026-07-07/12-34-52.m4a" in voice["targetAliases"]
    assert "store:voice/2026-07-07/12-34-52.m4a#t=252,300" in voice["targetAliases"]
    assert "voice/2026-07-07/12-34-52.m4a@4:12-5:00" in voice["targetAliases"]
    assert voice["ranges"][1] == {
        "kind": "voice-span",
        "origin": "mark",
        "order": 0,
        "disabled": False,
        "authored": "4:12-5:00",
        "startSeconds": 252.0,
        "endSeconds": 300.0,
        "quote": "a precise quotation",
        "refs": [],
        "comment": {
            "text": "exact case-study span",
                "line_start": 16,
                "line_end": 16,
        },
        "inlineComment": "central turn",
        "lineStart": 17,
        "lineEnd": 18,
    }
    assert all(
        "intentionally-omitted.md" not in portal.get("targetAliases", [])
        for portal in payload["portals"]
        if not portal["disabled"]
    )


def test_recursive_relative_manifests_and_cycle_are_diagnostic(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    root = tmp_path / "root.yaml"
    child = nested / "child.yaml"
    leaf = nested / "leaf.yaml"
    root.write_text(
        "components:\n  - name: worlds\n    manifests:\n      - nested/child.yaml\n",
        encoding="utf-8",
    )
    child.write_text(
        "components:\n  - name: leaf-link\n    manifests:\n      - leaf.yaml\n",
        encoding="utf-8",
    )
    leaf.write_text(
        "components:\n  - name: return\n    manifests:\n      - ../root.yaml\n",
        encoding="utf-8",
    )

    payload = compile_authored_manifest(root, context_name="recursive").to_dict()

    assert {str(root.resolve()), str(child.resolve()), str(leaf.resolve())} <= set(
        payload["context"]["sources"]
    )
    assert any(item["code"] == "included-manifest-cycle" for item in payload["diagnostics"])
    cycle = next(portal for portal in payload["portals"] if portal["status"] == "cycle")
    assert "targetPositionId" not in cycle
    resolved = [portal for portal in payload["portals"] if portal["status"] == "resolved"]
    assert len(resolved) == 2
    assert all(portal.get("targetPositionId") for portal in resolved)
    position_ids = {position["id"] for position in payload["positions"]}
    assert all(portal["positionId"] in position_ids for portal in resolved)


def test_linked_manifest_resolution_honors_config_root_and_nested_cwd(
    tmp_path: Path,
) -> None:
    authored_root = tmp_path / "authored"
    package = tmp_path / "package"
    nested = authored_root / "nested"
    authored_root.mkdir()
    package.mkdir()
    nested.mkdir()
    linked = authored_root / "linked.yaml"
    linked.write_text(
        "components:\n  - name: nested\n    manifests:\n      - nested/leaf.yaml\n",
        encoding="utf-8",
    )
    leaf = nested / "leaf.yaml"
    leaf.write_text("components: []\n", encoding="utf-8")
    manifest = package / "manifest.yaml"
    manifest.write_text(
        f"""config:
  root: {authored_root}
components:
  - name: linked-world
    manifests:
      - linked.yaml
""",
        encoding="utf-8",
    )

    payload = compile_authored_manifest(manifest, context_name="demo").to_dict()
    resolved = [
        portal for portal in payload["portals"] if portal["status"] == "resolved"
    ]

    assert {str(manifest.resolve()), str(linked.resolve()), str(leaf.resolve())} == set(
        payload["context"]["sources"]
    )
    assert {portal["authoredTarget"] for portal in resolved} == {
        "linked.yaml",
        "nested/leaf.yaml",
    }
    assert all(portal.get("targetPositionId") for portal in resolved)
    assert not any(
        diagnostic["code"] == "included-manifest-unresolved"
        for diagnostic in payload["diagnostics"]
    )


def test_linked_manifest_exposes_friendly_authored_locator_and_exact_return(
    tmp_path: Path,
) -> None:
    voice = tmp_path / "voice"
    voice.mkdir()
    root = tmp_path / "alpha.yaml"
    linked = voice / "address-analogy.md"
    root.write_text(
        """components:
  - name: voice-survey
    manifests:
      - voice/address-analogy.md
""",
        encoding="utf-8",
    )
    linked.write_text(
        """```yaml
components:
  - name: address-analogy
    text: Exact authored framing.
    files:
      - path: store:voice/2026-07-07/example.m4a
        marks:
          - at: 1:00-1:30
            quote: Exact evidence.
```
""",
        encoding="utf-8",
    )

    payload = compile_authored_manifest(root, context_name="alpha").to_dict()
    position = next(
        item
        for item in payload["positions"]
        if "alpha:voice-survey/address-analogy" in item["locators"]
    )
    portal = next(item for item in payload["portals"] if item["role"] == "material")

    assert position["parentId"] in {item["id"] for item in payload["positions"]}
    assert portal["reverse"]["positionId"] == position["id"]
    assert portal["reverse"]["positionStableId"] == position["stableId"]
    assert portal["ranges"][0]["quote"] == "Exact evidence."


def test_unresolved_reference_has_no_invented_target_id(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "components:\n  - name: missing\n    files:\n      - nowhere.md\n",
        encoding="utf-8",
    )

    payload = compile_authored_manifest(manifest, context_name="demo").to_dict()
    portal = payload["portals"][0]

    assert portal["status"] == "unresolved"
    assert "targetPositionId" not in portal
    assert "targetId" not in portal
    assert portal["targetAliases"] == ["nowhere.md"]
    assert payload["diagnostics"][0]["code"] == "reference-unresolved"
    assert payload["diagnostics"][0]["portalKey"] == portal["key"]


def test_mark_diagnostic_locates_exact_authored_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """components:
  - group: survey
    components:
      - name: evidence
        files:
          - path: store:voice/2026-07-01/first.m4a
            marks:
              - at: "0:10-0:20"
          - path: store:voice/2026-07-01/second.m4a
            marks:
              - at: "1:00-1:10"
              - at: "4:12"
                quote: This quote has only a point address.
""",
        encoding="utf-8",
    )

    payload = compile_authored_manifest(manifest, context_name="alpha").to_dict()
    diagnostic = payload["diagnostics"][0]

    assert diagnostic == {
        "code": "mark-quote-requires-range",
        "message": "Invalid authored range: mark-quote-requires-range",
        "severity": "error",
        "sourcePath": str(manifest),
        "line": 12,
        "positionKey": "root/p0000/p0000",
        "portalKey": "root/p0000/p0000/m0001",
        "details": {
            "authoredLocation": {
                "context": "alpha",
                "component": {
                    "key": "root/p0000/p0000",
                    "locator": "alpha:survey/evidence",
                },
                "reference": {
                    "role": "material",
                    "index": 1,
                    "target": "store:voice/2026-07-01/second.m4a",
                    "lineStart": 9,
                    "lineEnd": 13,
                },
                "range": {
                    "origin": "mark",
                    "index": 1,
                    "authored": "4:12",
                    "lineStart": 12,
                    "lineEnd": 13,
                },
                "mark": {
                    "index": 1,
                    "authored": "4:12",
                    "lineStart": 12,
                    "lineEnd": 13,
                },
            }
        },
    }


def test_legacy_point_quote_resolution_preserves_evidence_and_states(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """components:
  - name: evidence
    files:
      - path: store:voice/exact.m4a
        marks:
          - at: "0:10"
            quote: exact evidence
      - path: store:voice/missing.m4a
        marks:
          - at: "0:10"
            quote: absent evidence
      - path: store:voice/ambiguous.m4a
        marks:
          - at: "0:10"
            quote: repeated evidence
""",
        encoding="utf-8",
    )

    representations = {
        "store:voice/exact.m4a": {
            "source": "voice/exact.m4a#capture-1",
            "segments": [
                {"segmentIndex": 0, "startSeconds": 0, "endSeconds": 30, "text": "exact evidence"},
            ],
        },
        "store:voice/missing.m4a": {
            "source": "voice/missing.m4a#capture-1",
            "segments": [
                {"segmentIndex": 0, "startSeconds": 0, "endSeconds": 30, "text": "other evidence"},
            ],
        },
        "store:voice/ambiguous.m4a": {
            "source": "voice/ambiguous.m4a#capture-1",
            "segments": [
                {"segmentIndex": 0, "startSeconds": 0, "endSeconds": 30, "text": "repeated evidence"},
                {"segmentIndex": 1, "startSeconds": 60, "endSeconds": 90, "text": "repeated evidence"},
            ],
        },
    }

    payload = compile_authored_manifest(
        manifest,
        context_name="voice",
        quote_resolver=representations.get,
    ).to_dict()

    exact, missing, ambiguous = payload["portals"]
    assert exact["ranges"][0]["startSeconds"] == 10.0
    assert "endSeconds" not in exact["ranges"][0]
    assert exact["ranges"][0]["quoteResolution"] == {
        "state": "resolved",
        "target": "store:voice/exact.m4a",
        "source": "voice/exact.m4a#capture-1",
        "quote": "exact evidence",
        "matchMode": "exact",
        "range": {"startSeconds": 0.0, "endSeconds": 30.0},
        "evidence": {"text": "exact evidence", "segmentIndexes": [0]},
    }
    assert [diagnostic["code"] for diagnostic in payload["diagnostics"]] == [
        "mark-quote-unresolved",
        "mark-quote-ambiguous",
    ]
    assert missing["ranges"][0]["quoteResolution"]["state"] == "unresolved"
    assert ambiguous["ranges"][0]["quoteResolution"]["state"] == "ambiguous"
    assert ambiguous["ranges"][0]["quoteResolution"]["candidates"] == [
        {
            "range": {"startSeconds": 0.0, "endSeconds": 30.0},
            "segmentIndexes": [0],
            "text": "repeated evidence",
            "matchMode": "exact",
        },
        {
            "range": {"startSeconds": 60.0, "endSeconds": 90.0},
            "segmentIndexes": [1],
            "text": "repeated evidence",
            "matchMode": "exact",
        },
    ]


def test_dynamic_member_coverage_and_edition_are_explicit(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """components:
  - name: recent-index
    files:
      - "store:source:voice?after=2m&order=desc&list=1"
""",
        encoding="utf-8",
    )

    first = compile_authored_manifest(
        manifest,
        context_name="dynamic",
        compiled_at="2026-07-12T12:00:00Z",
    ).to_dict()
    second = compile_authored_manifest(
        manifest,
        context_name="dynamic",
        compiled_at="2026-07-12T13:00:00Z",
    ).to_dict()
    dynamic = first["portals"][0]["dynamic"]

    assert dynamic["editionTime"] == "2026-07-12T12:00:00Z"
    assert dynamic["coverage"] == {
        "state": "unacquired",
        "exact": False,
        "missing": ["provider acquisition is outside authored compilation"],
    }
    assert first["context"]["edition"] == second["context"]["edition"]
    assert first["positions"][1]["id"] == second["positions"][1]["id"]


def test_stable_ids_follow_authored_positions_across_reordering(tmp_path: Path) -> None:
    child = tmp_path / "child.yaml"
    child.write_text(
        """components:
  - name: inside
    files:
      - https://example.test/one
      - https://example.test/two
""",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """components:
  - name: before
    files:
      - https://example.test/before
  - name: world
    manifests:
      - child.yaml
  - name: after
    files:
      - https://example.test/after
""",
        encoding="utf-8",
    )
    first = compile_authored_manifest(manifest, context_name="demo").to_dict()

    manifest.write_text(
        """components:
  - name: after
    files:
      - https://example.test/after
  - name: before
    files:
      - https://example.test/before
  - name: world
    manifests:
      - child.yaml
""",
        encoding="utf-8",
    )
    second = compile_authored_manifest(manifest, context_name="demo").to_dict()

    first_positions = {
        position["locators"][0]: position["stableId"]
        for position in first["positions"]
    }
    second_positions = {
        position["locators"][0]: position["stableId"]
        for position in second["positions"]
    }
    assert first_positions == second_positions
    assert first["context"]["edition"] != second["context"]["edition"]

    def portal_identity(portal: dict[str, Any]) -> tuple[Any, Any, Any]:
        return (
            portal["reverse"]["positionLocators"][0],
            portal["role"],
            portal["authoredTarget"],
        )

    first_portals = {
        portal_identity(portal): portal["stableId"] for portal in first["portals"]
    }
    second_portals = {
        portal_identity(portal): portal["stableId"] for portal in second["portals"]
    }
    assert first_portals == second_portals


def test_framing_and_evidence_edits_preserve_stable_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """config:
  comment: old entrance framing
components:
  # old component framing
  - name: item  # old inline framing
    text: old component prose
    files:
      - path: https://example.test/source
        comment: old material framing
        marks:
          # old evidence framing
          - at: 1:00-1:30  # old mark framing
            quote: old evidence prose
            refs: [old-reference]
""",
        encoding="utf-8",
    )
    first = compile_authored_manifest(manifest, context_name="demo").to_dict()

    revised = manifest.read_text(encoding="utf-8")
    for old, new in (
        ("old entrance framing", "new entrance framing"),
        ("old component framing", "new component framing"),
        ("old inline framing", "new inline framing"),
        ("old component prose", "new component prose"),
        ("old material framing", "new material framing"),
        ("old evidence framing", "new evidence framing"),
        ("old mark framing", "new mark framing"),
        ("old evidence prose", "new evidence prose"),
        ("old-reference", "new-reference"),
    ):
        revised = revised.replace(old, new)
    manifest.write_text(revised, encoding="utf-8")
    second = compile_authored_manifest(manifest, context_name="demo").to_dict()

    assert {
        position["locators"][0]: position["stableId"]
        for position in first["positions"]
    } == {
        position["locators"][0]: position["stableId"]
        for position in second["positions"]
    }
    assert [portal["stableId"] for portal in first["portals"]] == [
        portal["stableId"] for portal in second["portals"]
    ]
    assert first["context"]["edition"] != second["context"]["edition"]
    assert [position["id"] for position in first["positions"]] != [
        position["id"] for position in second["positions"]
    ]


def test_target_and_range_edits_change_portal_stable_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """components:
  - name: item
    files:
      - path: https://example.test/one
        marks:
          - at: 1:00-1:30
""",
        encoding="utf-8",
    )
    first = compile_authored_manifest(manifest, context_name="demo").to_dict()

    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "https://example.test/one", "https://example.test/two"
        ),
        encoding="utf-8",
    )
    target_edit = compile_authored_manifest(manifest, context_name="demo").to_dict()
    assert first["positions"][1]["stableId"] == target_edit["positions"][1]["stableId"]
    assert first["portals"][0]["stableId"] != target_edit["portals"][0]["stableId"]

    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("1:00-1:30", "1:00-1:31"),
        encoding="utf-8",
    )
    range_edit = compile_authored_manifest(manifest, context_name="demo").to_dict()
    assert target_edit["portals"][0]["stableId"] != range_edit["portals"][0]["stableId"]


def test_rename_and_structural_move_change_position_and_portal_ids(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """components:
  - group: first
    components:
      - name: item
        files:
          - https://example.test/source
""",
        encoding="utf-8",
    )
    first = compile_authored_manifest(manifest, context_name="demo").to_dict()
    first_position = next(
        position for position in first["positions"] if position.get("name") == "item"
    )
    first_portal = first["portals"][0]

    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("name: item", "name: renamed"),
        encoding="utf-8",
    )
    renamed = compile_authored_manifest(manifest, context_name="demo").to_dict()
    renamed_position = next(
        position
        for position in renamed["positions"]
        if position.get("name") == "renamed"
    )
    assert first_position["stableId"] != renamed_position["stableId"]
    assert first_portal["stableId"] != renamed["portals"][0]["stableId"]

    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("group: first", "group: second"),
        encoding="utf-8",
    )
    moved = compile_authored_manifest(manifest, context_name="demo").to_dict()
    moved_position = next(
        position
        for position in moved["positions"]
        if position.get("name") == "renamed"
    )
    assert renamed_position["stableId"] != moved_position["stableId"]
    assert renamed["portals"][0]["stableId"] != moved["portals"][0]["stableId"]


def test_duplicate_linked_locators_are_occurrence_suffixed_and_collision_free(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child.md"
    child.write_text(
        """```yaml
components:
  - name: leaf
    files:
      - https://example.test/source
```
""",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """components:
  - name: world
    manifests:
      - child.md
      - child.md
""",
        encoding="utf-8",
    )

    payload = compile_authored_manifest(manifest, context_name="demo").to_dict()
    linked_positions = [
        position
        for position in payload["positions"]
        if position["locators"] == ["demo:world/child/leaf"]
    ]
    assert [position["stableId"] for position in linked_positions] == [
        "ctx://context/demo/demo%3Aworld/child/leaf",
        "ctx://context/demo/demo%3Aworld/child/leaf~2",
    ]
    linked_portals = [
        portal for portal in payload["portals"] if portal["role"] == "context"
    ]
    assert len({portal["stableId"] for portal in linked_portals}) == 2
    assert linked_portals[1]["stableId"].endswith("~2")


def test_linked_friendly_locator_stable_id_survives_revisions(tmp_path: Path) -> None:
    voice = tmp_path / "voice"
    voice.mkdir()
    linked = voice / "address-analogy.md"
    root = tmp_path / "alpha.yaml"
    root.write_text(
        """components:
  - name: voice-survey
    manifests:
      - voice/address-analogy.md
""",
        encoding="utf-8",
    )
    linked.write_text(
        """```yaml
components:
  - name: address-analogy
    text: old framing
    files:
      - path: store:voice/2026-07-07/example.m4a
        marks:
          - at: 1:00-1:30
            quote: old evidence
```
""",
        encoding="utf-8",
    )

    first = compile_authored_manifest(root, context_name="alpha").to_dict()
    linked.write_text(
        linked.read_text(encoding="utf-8")
        .replace("old framing", "new framing")
        .replace("old evidence", "new evidence"),
        encoding="utf-8",
    )
    second = compile_authored_manifest(root, context_name="alpha").to_dict()

    locator = "alpha:voice-survey/address-analogy"
    first_position = next(
        position for position in first["positions"] if locator in position["locators"]
    )
    second_position = next(
        position for position in second["positions"] if locator in position["locators"]
    )
    assert first_position["stableId"] == second_position["stableId"]
    assert first_position["stableId"] == (
        "ctx://context/alpha/alpha%3Avoice-survey/address-analogy"
    )
    first_portal = next(portal for portal in first["portals"] if portal["role"] == "material")
    second_portal = next(portal for portal in second["portals"] if portal["role"] == "material")
    assert first_portal["stableId"] == second_portal["stableId"]
    assert first["context"]["edition"] != second["context"]["edition"]
    assert first_position["id"] != second_position["id"]


def test_voice_span_in_path_exposes_recording_and_exact_aliases(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """components:
  - name: exact
    files:
      - "store:voice/2026-03-10/example.m4a@25:50-26:15"
""",
        encoding="utf-8",
    )

    portal = compile_authored_manifest(manifest, context_name="voice").to_dict()[
        "portals"
    ][0]

    assert "voice/2026-03-10/example.m4a" in portal["targetAliases"]
    assert "store:voice/2026-03-10/example.m4a" in portal["targetAliases"]
    assert "store:voice/2026-03-10/example.m4a#t=1550,1575" in portal["targetAliases"]
    assert portal["ranges"] == [
        {
            "kind": "voice-span",
            "origin": "path",
            "order": 0,
            "disabled": False,
            "authored": "25:50-26:15",
            "startSeconds": 1550.0,
            "endSeconds": 1575.0,
            "refs": [],
        }
    ]


def test_repo_collection_preserves_grouping_and_each_source_portal(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """components:
  - name: implementations
    repos:
      - root: ExampleOrg
        items:
          - https://github.com/ExampleOrg/one
          - https://github.com/ExampleOrg/two
""",
        encoding="utf-8",
    )

    portals = compile_authored_manifest(manifest, context_name="repos").to_dict()[
        "portals"
    ]

    assert [portal["authoredTarget"] for portal in portals] == [
        "https://github.com/ExampleOrg/one",
        "https://github.com/ExampleOrg/two",
    ]
    assert [portal["options"]["collection"] for portal in portals] == [
        {"root": "ExampleOrg", "order": 0},
        {"root": "ExampleOrg", "order": 1},
    ]
    assert all(portal["status"] == "external" for portal in portals)


def test_registry_api_and_cli_emit_machine_contract(tmp_path: Path) -> None:
    target = tmp_path / "project"
    target.mkdir()
    manifest = target / "manifest.yaml"
    manifest.write_text(
        "config:\n  name: Demo World\ncomponents:\n  - name: entrance\n    text: hello\n",
        encoding="utf-8",
    )
    registry = tmp_path / "contexts.json"
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "contexts": {
                    "demo": {
                        "targetDir": str(target),
                        "manifest": {"source": "manifest.yaml"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    editions = compile_authored_registry(registry, names=["demo"])
    assert [edition.context["name"] for edition in editions] == ["demo"]

    result = CliRunner().invoke(
        cli.cli,
        ["contexts", "compile", "demo", "--registry", str(registry)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["schemaVersion"] == AUTHORED_EDITION_SCHEMA_VERSION
    assert payload["editions"][0]["context"]["name"] == "demo"
    assert payload["editions"][0]["positions"][1]["locators"] == ["demo:entrance"]


def test_markdown_lead_prose_lands_once_on_manifest_position(tmp_path: Path) -> None:
    linked = tmp_path / "address-analogy.md"
    linked.write_text(
        """---
title: voice survey — address-analogy
tags:
  - ctx/manifest
---


felt mechanisms of the exchange itself, and rulings on mappings.
Selection and salience stand for redline.


```yaml
config:
  root: ~

components:
  - name: address-analogy
    files:
      - path: "store:voice/2024-11-07/23-05-58.m4a"
        marks:
          - at: 0:50-1:15    # a single wrapping annotation
            quote: |
              handle the loom
      - path: "store:voice/2024-12-16/17-10-23.m4a"
        marks:
          - at: 4:35-5:25    # first of two, not promoted
            quote: |
              quote one
          - at: 5:25-6:40    # second of two, not promoted
            quote: |
              quote two
```
""",
        encoding="utf-8",
    )
    root = tmp_path / "alpha.yaml"
    root.write_text(
        """components:
  - name: voice-survey
    manifests:
      - address-analogy.md
""",
        encoding="utf-8",
    )

    payload = compile_authored_manifest(root, context_name="alpha").to_dict()

    manifest_position = next(
        position
        for position in payload["positions"]
        if position["kind"] == "manifest" and position["sourcePath"] == str(linked.resolve())
    )
    assert manifest_position["framing"] == [
        {
            "kind": "text",
            "text": (
                "felt mechanisms of the exchange itself, and rulings on mappings.\n"
                "Selection and salience stand for redline."
            ),
        }
    ]

    component_position = next(
        position
        for position in payload["positions"]
        if "alpha:voice-survey/address-analogy" in position["locators"]
    )
    assert component_position["framing"] == []

    portals = {
        portal["authoredTarget"]: portal
        for portal in payload["portals"]
        if portal["positionId"] == component_position["id"]
    }
    single = portals["store:voice/2024-11-07/23-05-58.m4a"]
    multi = portals["store:voice/2024-12-16/17-10-23.m4a"]

    assert single["inlineComment"] == "a single wrapping annotation"
    assert "comment" not in single

    assert "inlineComment" not in multi
    assert "comment" not in multi
    assert multi["ranges"][0]["inlineComment"] == "first of two, not promoted"
    assert multi["ranges"][1]["inlineComment"] == "second of two, not promoted"


def test_prose_free_manifest_framing_and_portal_annotation_are_unchanged(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """components:
  - name: plain
    files:
      - path: store:voice/2026-01-01/example.m4a
        marks:
          - at: 0:10-0:20
            quote: unannotated evidence
""",
        encoding="utf-8",
    )

    payload = compile_authored_manifest(manifest, context_name="demo").to_dict()

    assert payload["positions"][0]["framing"] == []
    portal = payload["portals"][0]
    assert "comment" not in portal
    assert "inlineComment" not in portal


def test_manifest_cli_rejects_registry_selection(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("components: []\n", encoding="utf-8")
    result = CliRunner().invoke(
        cli.cli,
        ["contexts", "compile", "demo", "--manifest", str(manifest)],
    )
    assert result.exit_code != 0
    assert "--manifest cannot be combined" in result.output


def test_portal_reverse_omits_absent_line_bounds() -> None:
    portal = AuthoredPortal(
        id="ctx://authored-portal/alpha@one/root/target",
        stable_id="ctx://authored-portal/alpha/root/target",
        key="root/target",
        position_id="ctx://context/alpha@one/position/root",
        position_stable_id="ctx://context/alpha/position/root",
        position_locators=("alpha",),
        role="material",
        order=0,
        disabled=False,
        authored_target="notes.md",
        target_aliases=("notes.md",),
        target_position_id=None,
        source_path="manifest.yaml",
        line_start=None,
        line_end=None,
        comment=None,
        inline_comment=None,
        options={},
        ranges=(),
        dynamic=None,
        status="resolved",
    ).to_dict()

    assert "lineStart" not in portal
    assert "lineEnd" not in portal
    assert "lineStart" not in portal["reverse"]
    assert "lineEnd" not in portal["reverse"]
    assert None not in portal["reverse"].values()
