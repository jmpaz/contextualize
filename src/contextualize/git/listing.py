from __future__ import annotations

import glob
import os
import subprocess
from dataclasses import dataclass

from ..utils import _split_brace_options, brace_expand
from .target import GitTarget


@dataclass(frozen=True)
class GitListItem:
    target: str
    path: str


def _tracked_files(repo_dir: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", repo_dir, "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [item for item in result.stdout.split("\0") if item]


def _expanded_specs(spec: str) -> list[str]:
    parts = _split_brace_options(spec) if "{" in spec else [spec]
    expanded: list[str] = []
    for part in parts:
        expanded.extend(brace_expand(part))
    return expanded


def _is_glob_spec(spec: str) -> bool:
    return any(ch in spec for ch in "*?[")


def _glob_matches(repo_dir: str, spec: str, tracked: set[str]) -> set[str]:
    matches: set[str] = set()
    for path in glob.glob(os.path.join(repo_dir, spec), recursive=True):
        rel_path = os.path.relpath(path, repo_dir).replace(os.sep, "/")
        if rel_path in tracked:
            matches.add(rel_path)
    return matches


def _matches_spec(path: str, spec: str, tracked: set[str]) -> bool:
    normalized = spec.strip("/")
    if not normalized or normalized == ".":
        return True
    if normalized in tracked:
        return path == normalized
    return path.startswith(f"{normalized}/")


def _render_git_ref(target: GitTarget, path: str) -> str:
    repo_ref = target.repo_url
    if target.rev:
        repo_ref = f"{repo_ref}@{target.rev}"
    return f"{repo_ref}:{path}"


def list_git_target_refs(target: GitTarget, repo_dir: str) -> list[GitListItem]:
    tracked = _tracked_files(repo_dir)
    tracked_set = set(tracked)
    selected: list[str] = []
    seen: set[str] = set()
    specs = _expanded_specs(target.path) if target.path else [""]

    for spec in specs:
        glob_matches = (
            _glob_matches(repo_dir, spec.strip("/"), tracked_set)
            if _is_glob_spec(spec)
            else None
        )
        for path in tracked:
            if path in seen:
                continue
            matched = (
                path in glob_matches
                if glob_matches is not None
                else _matches_spec(path, spec, tracked_set)
            )
            if matched:
                selected.append(path)
                seen.add(path)

    return [
        GitListItem(target=_render_git_ref(target, path), path=path)
        for path in selected
    ]
