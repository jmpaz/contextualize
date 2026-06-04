from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from contextualize import cli
from contextualize.manifest.contexts import hydrate_contexts


def test_hydrate_context_from_registry_data(tmp_path: Path) -> None:
    target_dir = tmp_path / "repo"
    target_dir.mkdir()
    registry_path = tmp_path / "registry.json"
    status_path = tmp_path / "status.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "contexts": {
                    "demo": {
                        "targetDir": str(target_dir),
                        "replace": "guarded",
                        "manifest": {
                            "data": {
                                "config": {
                                    "context": {
                                        "dir": ".context",
                                        "include-meta": False,
                                    }
                                },
                                "components": [{"name": "main", "text": "hello"}],
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    statuses = hydrate_contexts(
        ["demo"],
        registry_path=registry_path,
        status_path=status_path,
    )

    assert statuses[0].result == "hydrated"
    assert (target_dir / ".context/main/notes/text-001.md").read_text(
        encoding="utf-8"
    ) == "hello"
    status_data = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_data["contexts"]["demo"]["result"] == "hydrated"


def test_hydrate_context_guarded_skips_untracked_context(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "repo"
    target_dir.mkdir()
    context_dir = target_dir / ".context"
    context_dir.mkdir()
    (context_dir / "handmade.md").write_text("keep", encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "contexts": {
                    "demo": {
                        "targetDir": str(target_dir),
                        "replace": "guarded",
                        "manifest": {
                            "data": {
                                "config": {
                                    "context": {
                                        "dir": ".context",
                                        "include-meta": False,
                                    }
                                },
                                "components": [{"name": "main", "text": "hello"}],
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    statuses = hydrate_contexts(
        ["demo"],
        registry_path=registry_path,
        status_path=tmp_path / "status.json",
    )

    assert statuses[0].result == "skipped"
    assert "untracked files" in (statuses[0].reason or "")
    assert (context_dir / "handmade.md").read_text(encoding="utf-8") == "keep"


def test_contexts_hydrate_cli_uses_registry(tmp_path: Path) -> None:
    target_dir = tmp_path / "repo"
    target_dir.mkdir()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "contexts": {
                    "demo": {
                        "targetDir": str(target_dir),
                        "manifest": {
                            "data": {
                                "config": {
                                    "context": {
                                        "dir": ".context",
                                        "include-meta": False,
                                    }
                                },
                                "components": [{"name": "main", "text": "hello"}],
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["contexts", "hydrate", "--registry", str(registry_path), "demo"],
        env={"XDG_STATE_HOME": str(tmp_path / "state")},
    )

    assert result.exit_code == 0
    assert "context demo: hydrated" in result.output
    assert (target_dir / ".context/main/notes/text-001.md").exists()


def test_contexts_list_cli_uses_registry(tmp_path: Path) -> None:
    target_dir = tmp_path / "repo"
    target_dir.mkdir()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "contexts": {
                    "demo": {
                        "targetDir": str(target_dir),
                        "manifest": {
                            "data": {
                                "config": {
                                    "context": {
                                        "dir": ".context",
                                        "include-meta": False,
                                    }
                                },
                                "components": [{"name": "main", "text": "hello"}],
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["contexts", "list", "--registry", str(registry_path)],
    )

    assert result.exit_code == 0
    assert f"demo\t{target_dir}" in result.output


def test_hydrate_cli_treats_text_file_with_manifest_block_as_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest-note.md"
    context_dir = tmp_path / "ctx"
    manifest_path.write_text(
        f"""---
title: context
---

```yaml
config:
  context:
    dir: {context_dir}
    include-meta: false
components:
  - name: main
    text: hello
```
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["hydrate", str(manifest_path)])

    assert result.exit_code == 0
    assert "Hydrated" in result.output
    assert (context_dir / "main/notes/text-001.md").read_text(encoding="utf-8") == "hello"
