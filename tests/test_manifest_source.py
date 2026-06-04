from __future__ import annotations

from pathlib import Path

import pytest

from contextualize.manifest.source import load_manifest_source, load_manifest_text


def test_load_manifest_source_extracts_yaml_block_from_text_file(tmp_path: Path) -> None:
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


def test_load_manifest_text_rejects_text_without_manifest() -> None:
    with pytest.raises(ValueError, match="No contextualize manifest found"):
        load_manifest_text("hello\n")
