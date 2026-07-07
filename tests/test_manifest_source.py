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
    assert "# - name: skipped" not in source.source_format.body
    assert "not hydrated" not in source.source_format.body
    assert "# speech materials" in source.source_format.body
    assert source.source_format.group_slices[("speech",)] == (
        "components:\n"
        "  - group: speech\n"
        "    components:\n"
        "      - name: anchor  # main voice note\n"
        "        text: hello\n"
        "\n"
    )


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
