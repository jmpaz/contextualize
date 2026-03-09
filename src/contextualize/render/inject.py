"""Content injection via {cx::...} markers."""

import os
import re
from datetime import timedelta
from typing import Any, Optional

from ..git.cache import ensure_repo, expand_git_paths
from ..git.target import parse_git_target
from ..references import create_file_references
from ..references.helpers import (
    is_http_url,
    looks_like_windows_drive,
    parse_git_url_target,
    parse_target_spec,
)
from ..utils import count_tokens, wrap_text

_INJECTION_PATTERN = re.compile(r"\{cx::((?:[^{}]|\{[^{}]*\})*)\}")
_EXTERNAL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _resolve_reference_targets(
    targets: list[str],
    *,
    format: str,
    label: str,
    depth: int,
    trace_collector: Optional[list],
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
    plugin_overrides: dict[str, Any] | None,
) -> str:
    refs = create_file_references(
        targets,
        format=format,
        label=label,
        inject=True,
        depth=depth,
        trace_collector=trace_collector,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
        plugin_overrides=plugin_overrides,
    )
    return refs["concatenated"]


def _http_fetch(
    url: str,
    name: str | None,
    depth: int,
    wrap: str | None = None,
    trace_collector: Optional[list] = None,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    plugin_overrides: dict[str, Any] | None = None,
) -> str:
    try:
        result = _resolve_reference_targets(
            [url],
            format="raw",
            label="relative",
            depth=depth,
            trace_collector=trace_collector,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            plugin_overrides=plugin_overrides,
        )
        return wrap_text(result, wrap or "md", name)
    except Exception as e:
        raise Exception(f"http fetch failed for {url}: {e}")


def _git_fetch(
    target: str,
    params: str | None,
    depth: int,
    wrap: str | None = None,
    trace_collector: Optional[list] = None,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    plugin_overrides: dict[str, Any] | None = None,
) -> str:
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
    result = _resolve_reference_targets(
        paths,
        format=fmt,
        label=lbl,
        depth=depth,
        trace_collector=trace_collector,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
        plugin_overrides=plugin_overrides,
    )
    return wrap_text(result, wrap) if wrap else result


def _local_fetch(
    path: str,
    root: str | None,
    params: str | None,
    depth: int,
    wrap: str | None = None,
    filename: str | None = None,
    trace_collector: Optional[list] = None,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    plugin_overrides: dict[str, Any] | None = None,
) -> str:
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

    label = filename if filename and len(existing) == 1 else lbl
    result = _resolve_reference_targets(
        existing,
        format=fmt,
        label=label,
        depth=depth,
        trace_collector=trace_collector,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
        plugin_overrides=plugin_overrides,
    )
    return wrap_text(result, wrap, filename) if wrap else result


def _external_fetch(
    target: str,
    depth: int,
    wrap: str | None = None,
    filename: str | None = None,
    trace_collector: Optional[list] = None,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    plugin_overrides: dict[str, Any] | None = None,
) -> str:
    result = _resolve_reference_targets(
        [target],
        format="raw",
        label="relative",
        depth=depth,
        trace_collector=trace_collector,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
        plugin_overrides=plugin_overrides,
    )
    return wrap_text(result, wrap or "md", filename)


def _process_injection(
    opts: dict[str, Any],
    depth: int,
    trace_collector: Optional[list] = None,
    source_file: Optional[str] = None,
    pattern_text: Optional[str] = None,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    plugin_overrides: dict[str, Any] | None = None,
) -> str:
    tgt = opts.get("target") or ""

    def _add_trace(result, resolved_tgt=None):
        if trace_collector is not None and source_file and pattern_text:
            trace_collector.append(
                (
                    "injection",
                    resolved_tgt or tgt,
                    source_file,
                    pattern_text,
                    count_tokens(result).get("count", 0),
                )
            )

    if is_http_url(tgt):
        try:
            result = _http_fetch(
                tgt,
                opts.get("filename"),
                depth,
                opts.get("wrap"),
                trace_collector,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
                plugin_overrides=plugin_overrides,
            )
        except Exception:
            if not parse_git_url_target(tgt):
                raise
            result = _git_fetch(
                tgt,
                opts.get("params"),
                depth,
                opts.get("wrap"),
                trace_collector,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
                plugin_overrides=plugin_overrides,
            )
        _add_trace(result)
    elif parse_git_target(tgt):
        result = _git_fetch(
            tgt,
            opts.get("params"),
            depth,
            opts.get("wrap"),
            trace_collector,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            plugin_overrides=plugin_overrides,
        )
        _add_trace(result)
    elif (
        _EXTERNAL_SCHEME_RE.match(tgt)
        and not looks_like_windows_drive(tgt)
        and parse_git_target(tgt) is None
    ):
        result = _external_fetch(
            tgt,
            depth,
            opts.get("wrap"),
            opts.get("filename"),
            trace_collector,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            plugin_overrides=plugin_overrides,
        )
        _add_trace(result)
    else:
        result = _local_fetch(
            tgt,
            opts.get("root"),
            opts.get("params"),
            depth,
            opts.get("wrap"),
            opts.get("filename"),
            trace_collector,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            plugin_overrides=plugin_overrides,
        )
        root = opts.get("root")
        base = os.path.expanduser(root) if root else os.getcwd()
        resolved = (
            tgt
            if "{" in tgt and "}" in tgt
            else (
                os.path.expanduser(tgt)
                if os.path.isabs(tgt)
                else os.path.join(base, os.path.expanduser(tgt))
            )
        )
        _add_trace(result, resolved)
    return result


def inject_content_in_text(
    text: str,
    depth: int = 5,
    trace_collector: Optional[list] = None,
    source_file: Optional[str] = None,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    plugin_overrides: dict[str, Any] | None = None,
) -> str:
    if depth <= 0:
        return text
    new = _INJECTION_PATTERN.sub(
        lambda m: _process_injection(
            parse_target_spec(m.group(1)),
            depth - 1,
            trace_collector,
            source_file,
            m.group(0),
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            plugin_overrides=plugin_overrides,
        ),
        text,
    )
    return (
        inject_content_in_text(
            new,
            depth - 1,
            trace_collector,
            source_file,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            plugin_overrides=plugin_overrides,
        )
        if _INJECTION_PATTERN.search(new)
        else new
    )
