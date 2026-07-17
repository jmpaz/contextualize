from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from contextualize.manifest import source as manifest_source
from contextualize.manifest.manifest import normalize_components
from contextualize.manifest.source import (
    iter_active_leaves,
    load_manifest_source,
    load_manifest_text,
)


def test_load_manifest_source_extracts_yaml_block_from_text_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "note.md"
    path.write_text(
        """---
title: context
---

```yaml
config:
  context:
    include-meta: false
components:
  - name: main
    text: hello
```
""",
        encoding="utf-8",
    )

    source = load_manifest_source(path)

    assert source.manifest_cwd == str(tmp_path)
    assert source.data["components"] == [{"name": "main", "text": "hello"}]
    assert source.source_format is not None
    assert source.source_format.source_path == str(path.resolve())
    assert source.source_format.line == 6
    assert source.source_format.body == (
        "config:\n"
        "  context:\n"
        "    include-meta: false\n"
        "components:\n"
        "  - name: main\n"
        "    text: hello\n"
    )


def test_load_manifest_source_captures_lead_prose_ahead_of_fence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "survey.md"
    path.write_text(
        """---
title: voice survey
tags:
  - ctx/manifest
---


felt mechanisms of the exchange itself. Mined from the voice corpus;
every quote verified against segments.


```yaml
components:
  - name: main
    text: hello
```
""",
        encoding="utf-8",
    )

    source = load_manifest_source(path)

    assert source.source_format is not None
    assert source.source_format.lead_text == (
        "felt mechanisms of the exchange itself. Mined from the voice corpus;\n"
        "every quote verified against segments."
    )


def test_load_manifest_source_has_no_lead_prose_without_document_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "note.md"
    path.write_text(
        """---
title: context
---

```yaml
components:
  - name: main
    text: hello
```
""",
        encoding="utf-8",
    )

    source = load_manifest_source(path)

    assert source.source_format is not None
    assert source.source_format.lead_text is None


def test_load_manifest_source_preserves_group_slices(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        """config:
  context:
    include-meta: true
components:
  # speech materials
  - group: speech
    components:
      - name: anchor  # main voice note
        text: hello
      # - name: skipped
      #   text: not hydrated

  - name: refs
    text: reference
""",
        encoding="utf-8",
    )

    source = load_manifest_source(path)

    assert source.source_format is not None
    assert "# - name: skipped" in source.source_format.body
    assert "not hydrated" in source.source_format.body
    assert "# speech materials" in source.source_format.body
    speech_slice = source.source_format.group_slices[("speech",)]
    assert speech_slice.line == 6
    assert speech_slice.body == (
        "components:\n"
        "  # speech materials\n"
        "  - group: speech\n"
        "    components:\n"
        "      - name: anchor  # main voice note\n"
        "        text: hello\n"
        "      # - name: skipped\n"
        "      #   text: not hydrated\n"
        "\n"
    )

    outline = source.source_format.outline
    assert [node["kind"] for node in outline] == ["group", "component"]
    speech_group = outline[0]
    assert speech_group["name"] == "speech"
    assert speech_group["order"] == 0
    assert speech_group["comment"] == {
        "text": "speech materials",
        "line_start": 5,
        "line_end": 5,
    }
    anchor, skipped = speech_group["children"]
    assert anchor["name"] == "anchor"
    assert anchor["disabled"] is False
    assert anchor["inline_comment"] == "main voice note"
    assert skipped["name"] == "skipped"
    assert skipped["disabled"] is True
    assert skipped["raw"] == "- name: skipped\n  text: not hydrated"
    refs = outline[1]
    assert refs["name"] == "refs"
    assert refs["order"] == 1


def test_load_manifest_text_rejects_text_without_manifest() -> None:
    with pytest.raises(ValueError, match="No contextualize manifest found"):
        load_manifest_text("hello\n")


def test_load_manifest_source_evaluates_nix_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "context.nix"
    path.write_text("{ components = []; }\n", encoding="utf-8")

    def run(args, **kwargs):
        assert args == ["nix", "eval", "--json", "--file", str(path.resolve())]
        assert kwargs["cwd"] == str(tmp_path)
        return CompletedProcess(
            args,
            0,
            stdout='{"config":{},"components":[{"name":"main","text":"hello"}]}',
            stderr="",
        )

    monkeypatch.setattr(manifest_source.subprocess, "run", run)

    source = load_manifest_source(path)

    assert source.manifest_cwd == str(tmp_path)
    assert source.data["components"] == [{"name": "main", "text": "hello"}]


def test_outline_handles_zero_indent_block_sequences(tmp_path: Path) -> None:
    # YAML permits list items at the same indentation as their parent key;
    # this style is common in the wild and previously produced an empty
    # outline (the "components:"/"files:" block-end scan stopped on the
    # first item instead of the next sibling key).
    path = tmp_path / "manifest.yaml"
    path.write_text(
        """config:
  root: '~'
components:
- name: percept span and facets
  files:
  - dev/mood-board/packages/percept/src/percept/span.py
  - dev/mood-board/packages/percept/src/percept/contracts.py
- name: current ad-hoc extraction to replace
  files:
  - dev/mood-board/src/mood_board/core/director/orchestrator.py
""",
        encoding="utf-8",
    )

    source = load_manifest_source(path)
    assert source.source_format is not None

    normalized = normalize_components(source.data["components"])
    leaves = list(iter_active_leaves(source.source_format.outline))
    assert len(leaves) == len(normalized) == 2
    assert [leaf["name"] for leaf in leaves] == [
        "percept span and facets",
        "current ad-hoc extraction to replace",
    ]
    assert [len(leaf["members"]["files"]) for leaf in leaves] == [2, 1]


MARKED_MANIFEST = """config:
  context:
    include-meta: true
components:
  - name: reckoning
    files:
      - path: "store:voice/2026-07-07/12-34-52.m4a"
        marks:
          # the case-study charge
          - at: 4:12  # absentee parent
          # - at: 9:00
          #   quote: |
          #     gone for now
          - at: 12:04-13:26
            quote: |
              ...treat this as a case study.
              # not a comment
              marks:
              Two prompts.
            refs:
              - notes/op9f.md
              - dev/mood-kit
      - plain.md  # untouched neighbor
  - name: refs  # sibling component
    files:
      - c.md
"""


def test_outline_parses_nested_marks_with_comment_grammar(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(MARKED_MANIFEST, encoding="utf-8")

    source = load_manifest_source(path)
    assert source.source_format is not None
    leaves = list(iter_active_leaves(source.source_format.outline))
    members = leaves[0]["members"]["files"]
    assert len(members) == 2

    marks = members[0]["marks"]
    assert [mark["order"] for mark in marks] == [0, 1, 2]

    point = marks[0]
    assert point["disabled"] is False
    assert point["comment"] == {
        "text": "the case-study charge",
        "line_start": 9,
        "line_end": 9,
    }
    assert point["inline_comment"] == "absentee parent"
    assert point["raw"] == "- at: 4:12"
    assert point["line_start"] == point["line_end"] == 10

    disabled = marks[1]
    assert disabled["disabled"] is True
    assert disabled["raw"] == "- at: 9:00\n  quote: |\n    gone for now"
    assert disabled["line_start"] == 11
    assert disabled["line_end"] == 13

    span = marks[2]
    assert span["raw"] == "- at: 12:04-13:26"
    assert span["line_start"] == 14
    assert span["line_end"] == 22
    assert "marks" not in span

    assert "marks" not in members[1]
    assert members[1]["inline_comment"] == "untouched neighbor"
    assert leaves[1]["inline_comment"] == "sibling component"


def test_nested_marks_pair_with_data_layer(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(MARKED_MANIFEST, encoding="utf-8")

    source = load_manifest_source(path)
    normalized = normalize_components(source.data["components"])
    data_marks = normalized[0]["files"][0]["marks"]

    # YAML 1.1 sexagesimal: unquoted `at: 4:12` arrives as int 252
    assert data_marks[0] == {"at": 252}
    assert data_marks[1]["at"] == "12:04-13:26"
    assert data_marks[1]["quote"] == (
        "...treat this as a case study.\n# not a comment\nmarks:\nTwo prompts.\n"
    )
    assert data_marks[1]["refs"] == ["notes/op9f.md", "dev/mood-kit"]

    outline_members = list(iter_active_leaves(source.source_format.outline))[0][
        "members"
    ]["files"]
    enabled_outline_marks = [
        mark for mark in outline_members[0]["marks"] if not mark["disabled"]
    ]
    assert len(enabled_outline_marks) == len(data_marks)


def test_nested_marks_zero_indent_style(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        """components:
- name: main
  files:
  - path: a.md
    marks:
    - at: 0:30  # zero-indent style
""",
        encoding="utf-8",
    )

    source = load_manifest_source(path)
    member = list(iter_active_leaves(source.source_format.outline))[0]["members"][
        "files"
    ][0]
    assert member["marks"][0]["inline_comment"] == "zero-indent style"
    assert member["marks"][0]["raw"] == "- at: 0:30"
    assert source.data["components"][0]["files"][0]["marks"] == [{"at": "0:30"}]


def test_flow_style_marks_are_data_only(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        """components:
  - name: main
    files:
      - {path: a.md, marks: [{at: "4:12"}]}
""",
        encoding="utf-8",
    )

    source = load_manifest_source(path)
    member = list(iter_active_leaves(source.source_format.outline))[0]["members"][
        "files"
    ][0]
    assert "marks" not in member
    assert source.data["components"][0]["files"][0]["marks"] == [{"at": "4:12"}]


def test_outline_finds_members_on_bare_unnamed_component(tmp_path: Path) -> None:
    # `- files:` with no `name:`/`group:`/`set:` key puts the member list key
    # directly on the item's dash line rather than on its own line.
    path = tmp_path / "manifest.yaml"
    path.write_text(
        """components:
  - files:
      - a.md
      - b.md
  - name: named
    files:
      - c.md
""",
        encoding="utf-8",
    )

    source = load_manifest_source(path)
    assert source.source_format is not None
    leaves = list(iter_active_leaves(source.source_format.outline))
    assert leaves[0]["name"] is None
    assert len(leaves[0]["members"]["files"]) == 2
    assert leaves[1]["name"] == "named"
    assert len(leaves[1]["members"]["files"]) == 1
