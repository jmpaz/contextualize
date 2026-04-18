from __future__ import annotations

import subprocess
from pathlib import Path

from click.testing import CliRunner

from contextualize import cli
from contextualize.git.listing import GitListItem, list_git_target_refs
from contextualize.git.target import GitTarget


def _init_tracked_repo(path: Path) -> Path:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    for rel_path, content in {
        "README.md": "readme",
        "docs/guide.md": "guide",
        "docs/nested/deep.md": "deep",
        "src/lib.py": "lib",
        "src/nested/mod.py": "mod",
    }.items():
        file_path = path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    return path


def test_list_git_target_refs_lists_repo_root(tmp_path: Path) -> None:
    repo = _init_tracked_repo(tmp_path)
    target = GitTarget(
        repo_url="https://example.com/org/repo",
        cache_dir=str(repo),
        path=None,
        rev=None,
    )

    items = list_git_target_refs(target, str(repo))

    assert [item.target for item in items] == [
        "https://example.com/org/repo:README.md",
        "https://example.com/org/repo:docs/guide.md",
        "https://example.com/org/repo:docs/nested/deep.md",
        "https://example.com/org/repo:src/lib.py",
        "https://example.com/org/repo:src/nested/mod.py",
    ]


def test_list_git_target_refs_supports_file_directory_glob_and_revision(
    tmp_path: Path,
) -> None:
    repo = _init_tracked_repo(tmp_path)

    assert [
        item.target
        for item in list_git_target_refs(
            GitTarget(
                repo_url="https://example.com/org/repo",
                cache_dir=str(repo),
                path="README.md",
                rev=None,
            ),
            str(repo),
        )
    ] == ["https://example.com/org/repo:README.md"]
    assert [
        item.target
        for item in list_git_target_refs(
            GitTarget(
                repo_url="https://example.com/org/repo",
                cache_dir=str(repo),
                path="src",
                rev=None,
            ),
            str(repo),
        )
    ] == [
        "https://example.com/org/repo:src/lib.py",
        "https://example.com/org/repo:src/nested/mod.py",
    ]
    assert [
        item.target
        for item in list_git_target_refs(
            GitTarget(
                repo_url="https://example.com/org/repo",
                cache_dir=str(repo),
                path="docs/*.md",
                rev="main",
            ),
            str(repo),
        )
    ] == ["https://example.com/org/repo@main:docs/guide.md"]


def test_cat_list_routes_git_targets_without_reading_content(
    monkeypatch, tmp_path: Path
) -> None:
    from contextualize.plugins import clear_loaded_plugins_cache
    from contextualize.plugins import loader as plugin_loader

    def _ensure_repo(target, *, pull=False, reclone=False):
        return str(tmp_path)

    def _list_git_target_refs(target, repo_dir: str):
        return [
            GitListItem(target=f"{target.repo_url}:README.md", path="README.md"),
            GitListItem(target=f"{target.repo_url}:src/lib.py", path="src/lib.py"),
        ]

    def _create_file_references(*_args, **_kwargs):
        raise AssertionError("cat --list should not read file content")

    monkeypatch.setattr(plugin_loader, "_iter_plugin_entrypoints", lambda: [])
    clear_loaded_plugins_cache()
    monkeypatch.setattr("contextualize.git.cache.ensure_repo", _ensure_repo)
    monkeypatch.setattr(
        "contextualize.git.listing.list_git_target_refs",
        _list_git_target_refs,
    )
    monkeypatch.setattr(
        "contextualize.references.create_file_references",
        _create_file_references,
    )

    result = CliRunner().invoke(
        cli.cli,
        ["cat", "--list", "https://github.com/octocat/Hello-World"],
    )

    assert result.exit_code == 0
    assert result.output == (
        "- `https://github.com/octocat/Hello-World:README.md`\n"
        "- `https://github.com/octocat/Hello-World:src/lib.py`\n"
    )
