from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from contextualize import cli
from contextualize.manifest.edition import (
    AUTHORED_EDITION_SCHEMA_VERSION,
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
    payload = json.loads(result.output)
    assert payload["schemaVersion"] == AUTHORED_EDITION_SCHEMA_VERSION
    assert payload["editions"][0]["context"]["name"] == "demo"
    assert payload["editions"][0]["positions"][1]["locators"] == ["demo:entrance"]


def test_manifest_cli_rejects_registry_selection(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("components: []\n", encoding="utf-8")
    result = CliRunner().invoke(
        cli.cli,
        ["contexts", "compile", "demo", "--manifest", str(manifest)],
    )
    assert result.exit_code != 0
    assert "--manifest cannot be combined" in result.output
