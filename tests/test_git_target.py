from __future__ import annotations

import os
from pathlib import Path

import pytest

from contextualize.git.target import parse_git_target


def test_parse_git_target_rejects_http_fragment_on_non_git_host() -> None:
    assert parse_git_target("https://en.wikipedia.org/wiki/Wikipedia#History") is None


def test_parse_git_target_rejects_http_path_suffix_on_non_git_host() -> None:
    assert parse_git_target("https://en.wikipedia.org/wiki/Wikipedia:About") is None


def test_parse_git_target_keeps_dot_git_urls_on_unknown_hosts() -> None:
    target = parse_git_target(
        "https://git.example.com/org/repo.git@main:path/in/repo.txt"
    )

    assert target is not None
    assert target.repo_url == "https://git.example.com/org/repo.git"
    assert target.rev == "main"
    assert target.path == "path/in/repo.txt"


@pytest.mark.parametrize(
    "repo_url",
    [
        "https://github.com/octocat/Hello-World",
        "https://gitlab.com/pages/plain-html",
        "https://codeberg.org/Codeberg/Documentation",
        "https://git.sr.ht/~sircmpwn/git.sr.ht",
        "https://tangled.org/tangled.org/core",
    ],
)
def test_parse_git_target_supports_known_http_git_host_roots(repo_url: str) -> None:
    target = parse_git_target(repo_url)

    assert target is not None
    assert target.repo_url == repo_url
    assert target.rev is None
    assert target.path is None


def test_parse_git_target_normalizes_www_for_supported_host_detection() -> None:
    target = parse_git_target("https://www.codeberg.org/Codeberg/Documentation")

    assert target is not None
    assert target.repo_url == "https://www.codeberg.org/Codeberg/Documentation"
    assert target.path is None


@pytest.mark.parametrize(
    ("spec", "repo_url", "path"),
    [
        (
            "https://github.com/octocat/Hello-World:README",
            "https://github.com/octocat/Hello-World",
            "README",
        ),
        (
            "https://gitlab.com/pages/plain-html:README.md",
            "https://gitlab.com/pages/plain-html",
            "README.md",
        ),
        (
            "https://codeberg.org/Codeberg/Documentation:README.md",
            "https://codeberg.org/Codeberg/Documentation",
            "README.md",
        ),
        (
            "https://git.sr.ht/~sircmpwn/git.sr.ht:README.md",
            "https://git.sr.ht/~sircmpwn/git.sr.ht",
            "README.md",
        ),
        (
            "https://tangled.org/tangled.org/core:readme.md",
            "https://tangled.org/tangled.org/core",
            "readme.md",
        ),
    ],
)
def test_parse_git_target_supports_known_http_git_hosts(
    spec: str, repo_url: str, path: str
) -> None:
    target = parse_git_target(spec)

    assert target is not None
    assert target.repo_url == repo_url
    assert target.rev is None
    assert target.path == path


@pytest.mark.parametrize(
    ("spec", "repo_url", "rev", "path"),
    [
        (
            "https://codeberg.org/Codeberg/Documentation@main:README.md",
            "https://codeberg.org/Codeberg/Documentation",
            "main",
            "README.md",
        ),
        (
            "https://tangled.org/tangled.org/core@main:readme.md",
            "https://tangled.org/tangled.org/core",
            "main",
            "readme.md",
        ),
    ],
)
def test_parse_git_target_supports_known_http_git_host_revisions(
    spec: str, repo_url: str, rev: str, path: str
) -> None:
    target = parse_git_target(spec)

    assert target is not None
    assert target.repo_url == repo_url
    assert target.rev == rev
    assert target.path == path


def test_cat_routes_known_http_git_hosts_through_git_cache(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from contextualize import cli
    from contextualize.git.target import GitTarget
    from contextualize.plugins import clear_loaded_plugins_cache
    from contextualize.plugins import loader as plugin_loader

    routed_targets: list[GitTarget] = []
    ref_paths: list[str] = []
    source_path = tmp_path / "README.md"
    source_path.write_text("from git cache", encoding="utf-8")

    def _ensure_repo(target, *, pull=False, reclone=False):
        routed_targets.append(target)
        return str(tmp_path)

    def _expand_git_paths(repo_dir: str, spec: str):
        return [str(Path(repo_dir) / spec)]

    def _create_file_references(paths, *_args, **_kwargs):
        ref_paths.extend(paths)
        return {
            "refs": [],
            "concatenated": "",
            "ignored_files": [],
            "ignored_folders": {},
        }

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(plugin_loader, "_iter_plugin_entrypoints", lambda: [])
    clear_loaded_plugins_cache()
    monkeypatch.setattr("contextualize.git.cache.ensure_repo", _ensure_repo)
    monkeypatch.setattr("contextualize.git.cache.expand_git_paths", _expand_git_paths)
    monkeypatch.setattr(
        "contextualize.references.create_file_references",
        _create_file_references,
    )

    specs = [
        "https://github.com/octocat/Hello-World:README.md",
        "https://gitlab.com/pages/plain-html:README.md",
        "https://codeberg.org/Codeberg/Documentation:README.md",
        "https://git.sr.ht/~sircmpwn/git.sr.ht:README.md",
        "https://tangled.org/tangled.org/core:readme.md",
    ]
    runner = CliRunner()

    for spec in specs:
        result = runner.invoke(cli.cli, ["cat", spec])

        assert result.exit_code == 0, result.output

    assert [target.repo_url for target in routed_targets] == [
        "https://github.com/octocat/Hello-World",
        "https://gitlab.com/pages/plain-html",
        "https://codeberg.org/Codeberg/Documentation",
        "https://git.sr.ht/~sircmpwn/git.sr.ht",
        "https://tangled.org/tangled.org/core",
    ]
    assert ref_paths == [
        str(tmp_path / "README.md"),
        str(tmp_path / "README.md"),
        str(tmp_path / "README.md"),
        str(tmp_path / "README.md"),
        str(tmp_path / "readme.md"),
    ]


@pytest.mark.skipif(
    os.environ.get("CONTEXTUALIZE_RUN_LIVE_GIT_TESTS") != "1",
    reason="set CONTEXTUALIZE_RUN_LIVE_GIT_TESTS=1 to run live git provider tests",
)
@pytest.mark.parametrize(
    ("spec", "expected_path"),
    [
        ("https://github.com/octocat/Hello-World", None),
        ("https://github.com/octocat/Hello-World:README", "README"),
        ("https://gitlab.com/pages/plain-html", None),
        ("https://gitlab.com/pages/plain-html:README.md", "README.md"),
        ("https://codeberg.org/Codeberg/Documentation", None),
        ("https://codeberg.org/Codeberg/Documentation:README.md", "README.md"),
        ("https://git.sr.ht/~sircmpwn/git.sr.ht", None),
        ("https://git.sr.ht/~sircmpwn/git.sr.ht:README.md", "README.md"),
        ("https://tangled.org/tangled.org/core", None),
        ("https://tangled.org/tangled.org/core:readme.md", "readme.md"),
    ],
)
def test_live_git_provider_targets_resolve(
    monkeypatch, tmp_path: Path, spec: str, expected_path: str | None
):
    from contextualize.git.cache import ensure_repo, expand_git_paths

    monkeypatch.setattr(
        "contextualize.git.target.CACHE_ROOT", str(tmp_path / "git-cache")
    )
    target = parse_git_target(spec)

    assert target is not None
    repo_dir = ensure_repo(target, reclone=True)
    if expected_path is None:
        assert Path(repo_dir).is_dir()
    else:
        [resolved_path] = expand_git_paths(repo_dir, target.path)
        assert Path(resolved_path).is_file()
