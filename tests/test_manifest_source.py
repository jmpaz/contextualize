from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from contextualize.manifest import source as manifest_source
from contextualize.manifest.source import load_manifest_source, load_manifest_text


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
