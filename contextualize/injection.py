"""resolve {cx::...} patterns with http, git or local file content."""

import os
import re
from typing import Any

from .gitcache import ensure_repo, expand_git_paths, parse_git_target
from .reference import create_file_references, process_text

# match {cx:: ... } allowing braces within the target
_PATTERN = re.compile(r"\{cx::((?:[^{}]|\{[^{}]*\})*)\}")


def _parse(piece: str) -> dict[str, Any]:
    parts = piece.split("::")
    opts: dict[str, str | None] = {}
    target_parts: list[str] = []
    for part in parts:
        m = re.fullmatch(r'(filename|params|root)=(?:"([^\"]*)"|([^\"]*))', part)
        if m:
            opts[m.group(1)] = m.group(2) or m.group(3)
        else:
            target_parts.append(part)
    opts["target"] = "::".join(target_parts)
    return opts


def _http_fetch(url: str, name: str | None, depth: int) -> str:
    try:
        import json

        import requests

        r = requests.get(url, timeout=30, headers={"User-Agent": "contextualize"})
        r.raise_for_status()
        text = r.text
        if "json" in r.headers.get("Content-Type", ""):
            try:
                text = json.dumps(r.json(), indent=2)
            except Exception:
                pass
        if depth > 0:
            text = inject_content_in_text(text, depth)
        return process_text(text, format="md", label=name or url)
    except Exception as e:
        raise Exception(f"http fetch failed for {url}: {e}")


def _git_fetch(target: str, params: str | None, depth: int) -> str:
    git_opts = params.split() if params else []
    fmt = "md"
    lbl = "relative"
    pull = "--git-pull" in git_opts
    reclone = "--git-reclone" in git_opts
    for i, p in enumerate(git_opts):
        if p == "--format" and i + 1 < len(git_opts):
            fmt = git_opts[i + 1]
        if p == "--label" and i + 1 < len(git_opts):
            lbl = git_opts[i + 1]

    tgt = parse_git_target(target)
    if not tgt:
        raise Exception(f"invalid git target: {target}")
    repo = ensure_repo(tgt, pull=pull, reclone=reclone)
    paths = [repo] if not tgt.path else expand_git_paths(repo, tgt.path)
    refs = create_file_references(paths, format=fmt, label=lbl, inject=True, depth=depth)
    return refs["concatenated"]


def _local_fetch(path: str, root: str | None, params: str | None, depth: int) -> str:
    fmt = "md"
    lbl = "relative"
    if params:
        parts = params.split()
        for i, p in enumerate(parts):
            if p == "--format" and i + 1 < len(parts):
                fmt = parts[i + 1]
            if p == "--label" and i + 1 < len(parts):
                lbl = parts[i + 1]
    base = os.path.expanduser(root) if root else os.getcwd()
    if os.path.isabs(path):
        base = ""
    paths = expand_git_paths(base, os.path.expanduser(path))
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        raise Exception(f"path not found: {path}")
    refs = create_file_references(existing, format=fmt, label=lbl, inject=True, depth=depth)
    return refs["concatenated"]


def _process(opts: dict[str, Any], depth: int) -> str:
    tgt = opts.get("target") or ""
    if tgt.startswith("http://") or tgt.startswith("https://"):
        try:
            return _http_fetch(tgt, opts.get("filename"), depth)
        except Exception:
            if parse_git_target(tgt):
                return _git_fetch(tgt, opts.get("params"), depth)
            raise
    if parse_git_target(tgt):
        return _git_fetch(tgt, opts.get("params"), depth)
    return _local_fetch(tgt, opts.get("root"), opts.get("params"), depth)


def inject_content_in_text(text: str, depth: int = 5) -> str:
    if depth <= 0:
        return text

    def repl(m: re.Match) -> str:
        opts = _parse(m.group(1))
        return _process(opts, depth - 1)

    new = _PATTERN.sub(repl, text)
    return inject_content_in_text(new, depth - 1) if _PATTERN.search(new) else new

