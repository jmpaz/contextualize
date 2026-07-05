from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from contextualize import cli
from contextualize.manifest.contexts import ContextHydrationStatus, hydrate_contexts
from contextualize.progress import record_progress


def _isolate_manifest_link_cache(monkeypatch, tmp_path: Path) -> None:
    """`ensure_manifest_link_hydrated` derives its canonical cache/status
    paths from XDG_DATA_HOME/XDG_STATE_HOME (falling back to HOME) -- all
    three must be redirected under tmp_path, since XDG_DATA_HOME/XDG_STATE_HOME
    take priority over HOME and are commonly set in the ambient environment,
    which would otherwise leak test artifacts into the real user cache."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "home" / ".local" / "share"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "home" / ".local" / "state"))


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


def test_hydrate_context_guarded_allows_files_owned_by_new_plan(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "repo"
    target_dir.mkdir()
    context_dir = target_dir / ".context"
    existing = context_dir / "excerpts/notes/text-001.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("old", encoding="utf-8")
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
                                        "path-strategy": "by-component",
                                    }
                                },
                                "components": [
                                    {"name": "excerpts", "text": "new"},
                                ],
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

    assert statuses[0].result == "hydrated"
    assert existing.read_text(encoding="utf-8") == "new"


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
    assert "Progress summary:" in result.output
    assert "  hydrate context: total=1 done=1" in result.output
    assert (target_dir / ".context/main/notes/text-001.md").exists()


def test_contexts_hydrate_quiet_suppresses_default_progress_summary(
    tmp_path: Path,
) -> None:
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
        ["contexts", "hydrate", "--quiet", "--registry", str(registry_path), "demo"],
        env={"XDG_STATE_HOME": str(tmp_path / "state")},
    )

    assert result.exit_code == 0
    assert "context demo: hydrated" in result.output
    assert "Progress summary:" not in result.output


def test_contexts_hydrate_global_quiet_suppresses_default_progress_summary(
    tmp_path: Path,
) -> None:
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
        ["--quiet", "contexts", "hydrate", "--registry", str(registry_path), "demo"],
        env={"XDG_STATE_HOME": str(tmp_path / "state")},
    )

    assert result.exit_code == 0
    assert "context demo: hydrated" in result.output
    assert "Progress summary:" not in result.output


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
    assert "Context registry: total=1" in result.output
    assert "name  source       target" in result.output
    assert f"demo  inline data  {target_dir}" in result.output


def test_contexts_list_cli_handles_empty_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"version": 1, "contexts": {}}),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["contexts", "list", "--registry", str(registry_path)],
    )

    assert result.exit_code == 0
    assert result.output == "Context registry: total=0\n"


def test_contexts_hydrate_prefixes_context_warnings(
    monkeypatch,
    tmp_path: Path,
) -> None:
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
                                "config": {"context": {"dir": ".context"}},
                                "components": [],
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def _warn_and_hydrate(context, _overrides):
        print("Warning: plugin failed", file=__import__("sys").stderr)
        return ContextHydrationStatus(
            name=context.name,
            target_dir=str(context.target_dir),
            manifest_source="inline data",
            context_dir=str(target_dir / ".context"),
            result="hydrated",
            reason=None,
            timestamp="2026-06-04T00:00:00Z",
        )

    monkeypatch.setattr(
        "contextualize.manifest.contexts._hydrate_one",
        _warn_and_hydrate,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "contexts",
            "hydrate",
            "--registry",
            str(registry_path),
            "--status",
            str(tmp_path / "status.json"),
            "demo",
        ],
    )

    assert result.exit_code == 0
    assert "context demo: hydrated" in result.output
    assert "context demo: Warning: plugin failed" in result.output


def test_contexts_hydrate_verbose_reports_progress_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
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
                                "config": {"context": {"dir": ".context"}},
                                "components": [],
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def _record_and_hydrate(context, _overrides):
        record_progress("arena", "channel", "cache_hit", target="Cached Channel")
        return ContextHydrationStatus(
            name=context.name,
            target_dir=str(context.target_dir),
            manifest_source="inline data",
            context_dir=str(target_dir / ".context"),
            result="hydrated",
            reason=None,
            timestamp="2026-06-04T00:00:00Z",
        )

    monkeypatch.setattr(
        "contextualize.manifest.contexts._hydrate_one",
        _record_and_hydrate,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "--verbose",
            "contexts",
            "hydrate",
            "--registry",
            str(registry_path),
            "--status",
            str(tmp_path / "status.json"),
            "demo",
        ],
    )

    assert result.exit_code == 0
    assert "ignoring global options [--verbose]" not in result.output
    assert "[progress]" not in result.output
    assert "progress:" not in result.output
    assert "Progress summary:" in result.output
    assert "  hydrate context: total=1 done=1" in result.output
    assert "  context demo:" in result.output
    assert "    arena channel: cache_hit=1" in result.output


def test_contexts_hydrate_verbose_reports_failed_component_reason(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "repo"
    target_dir.mkdir()
    registry_path = tmp_path / "registry.json"
    status_path = tmp_path / "status.json"
    missing_path = target_dir / "missing.md"
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
                                "components": [
                                    {"name": "ready", "text": "hello"},
                                    {"name": "voice", "files": ["missing.md"]},
                                ],
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
        [
            "--verbose",
            "contexts",
            "hydrate",
            "--registry",
            str(registry_path),
            "--status",
            str(status_path),
            "demo",
        ],
    )

    assert result.exit_code == 0
    assert "Progress summary:" in result.output
    assert "  hydrate context: total=1 failed=1" in result.output
    assert "latest_failure=demo: Component 'voice' path not found" in result.output
    assert "  context demo:" in result.output
    assert "    hydrate component: total=2 failed=1 done=1" in result.output
    assert "latest_failure=voice: Component 'voice' path not found" in result.output
    assert str(missing_path) in result.output


def test_hydrate_context_with_manifests_component(
    monkeypatch, tmp_path: Path
) -> None:
    _isolate_manifest_link_cache(monkeypatch, tmp_path)

    sub_manifest = tmp_path / "sub.yaml"
    sub_manifest.write_text(
        "config:\n"
        "  context:\n"
        "    include-meta: false\n"
        "components:\n"
        "  - name: main\n"
        "    text: hello from sub\n",
        encoding="utf-8",
    )

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
                                        "path-strategy": "on-disk",
                                    }
                                },
                                "components": [
                                    {
                                        "name": "contexts",
                                        "manifests": [str(sub_manifest)],
                                    }
                                ],
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
    linked = target_dir / ".context" / "contexts" / "sub"
    assert linked.is_symlink()
    assert (
        linked / "main" / "notes" / "text-001.md"
    ).read_text(encoding="utf-8").strip() == "hello from sub"


def test_contexts_hydrate_partial_when_sub_manifest_fails(
    monkeypatch, tmp_path: Path
) -> None:
    _isolate_manifest_link_cache(monkeypatch, tmp_path)

    good_manifest = tmp_path / "good.yaml"
    good_manifest.write_text(
        "config:\n"
        "  context:\n"
        "    include-meta: false\n"
        "components:\n"
        "  - name: main\n"
        "    text: hello from sub\n",
        encoding="utf-8",
    )
    bad_manifest = tmp_path / "bad.yaml"
    bad_manifest.write_text(
        "config:\n"
        "  context:\n"
        "    include-meta: false\n"
        "components:\n"
        "  - name: broken\n"
        "    files: [missing.md]\n",
        encoding="utf-8",
    )

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
                                        "path-strategy": "on-disk",
                                    }
                                },
                                "components": [
                                    {
                                        "name": "contexts",
                                        "manifests": [
                                            str(good_manifest),
                                            str(bad_manifest),
                                        ],
                                    }
                                ],
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
        [
            "contexts",
            "hydrate",
            "--registry",
            str(registry_path),
            "--status",
            str(status_path),
            "demo",
        ],
    )

    assert result.exit_code == 0
    assert "context demo: partial" in result.output
    assert str(bad_manifest) in result.output
    assert "missing.md" in result.output
    status_data = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_data["contexts"]["demo"]["result"] == "partial"
    assert (target_dir / ".context" / "contexts" / "good").is_symlink()
    assert (target_dir / ".context" / "contexts" / "bad" / "FAILED.md").is_file()

    strict_result = runner.invoke(
        cli.cli,
        [
            "contexts",
            "hydrate",
            "--strict",
            "--registry",
            str(registry_path),
            "--status",
            str(status_path),
            "demo",
        ],
    )

    assert strict_result.exit_code != 0
    assert "context demo: partial" in strict_result.output
    assert "1 context(s) failed or partial" in strict_result.output


def test_hydrate_manifests_shared_sub_manifest_reuses_canonical_dir(
    monkeypatch, tmp_path: Path
) -> None:
    _isolate_manifest_link_cache(monkeypatch, tmp_path)

    sub_manifest = tmp_path / "sub.yaml"
    sub_manifest.write_text(
        "config:\n"
        "  context:\n"
        "    include-meta: false\n"
        "components:\n"
        "  - name: main\n"
        "    text: shared content\n",
        encoding="utf-8",
    )

    def _parent_manifest_data(context_dir: Path) -> dict:
        return {
            "config": {
                "context": {
                    "dir": str(context_dir),
                    "include-meta": False,
                    "path-strategy": "on-disk",
                }
            },
            "components": [
                {"name": "contexts", "manifests": [str(sub_manifest)]}
            ],
        }

    target_dir_a = tmp_path / "repo-a"
    target_dir_a.mkdir()
    target_dir_b = tmp_path / "repo-b"
    target_dir_b.mkdir()
    registry_path = tmp_path / "registry.json"
    status_path = tmp_path / "status.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "contexts": {
                    "parent-a": {
                        "targetDir": str(target_dir_a),
                        "replace": "guarded",
                        "manifest": {
                            "data": _parent_manifest_data(target_dir_a / ".context")
                        },
                    },
                    "parent-b": {
                        "targetDir": str(target_dir_b),
                        "replace": "guarded",
                        "manifest": {
                            "data": _parent_manifest_data(target_dir_b / ".context")
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    statuses = hydrate_contexts(
        ["parent-a", "parent-b"],
        registry_path=registry_path,
        status_path=status_path,
    )

    assert [s.result for s in statuses] == ["hydrated", "hydrated"]

    linked_a = (target_dir_a / ".context" / "contexts" / "sub").resolve()
    linked_b = (target_dir_b / ".context" / "contexts" / "sub").resolve()
    assert linked_a == linked_b

    second_statuses = hydrate_contexts(
        ["parent-a", "parent-b"],
        registry_path=registry_path,
        status_path=status_path,
    )
    assert [s.result for s in second_statuses] == ["up-to-date", "up-to-date"]


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
    assert "Progress summary:" in result.output
    assert "  hydrate component: total=1 done=1" in result.output
    assert (context_dir / "main/notes/text-001.md").read_text(encoding="utf-8") == "hello"


def test_hydrate_cli_quiet_suppresses_default_progress_summary(
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
    result = runner.invoke(cli.cli, ["hydrate", "--quiet", str(manifest_path)])

    assert result.exit_code == 0
    assert "Hydrated" in result.output
    assert "Progress summary:" not in result.output
