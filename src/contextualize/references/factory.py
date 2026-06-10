"""Factory function for resolving targets into references."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..concurrency import media_task_semaphore, run_indexed_tasks_fail_fast
from ..plugins.resolve import (
    list_plugin_targets,
    materialize_plugin_target,
    resolve_plugin_references,
)
from ..progress import log_progress, record_progress
from ..runtime import get_payload_media_jobs
from ..git.target import parse_git_target
from ..utils import brace_expand, count_tokens
from .file import FileExistenceReference, FileReference
from .audio_transcription import is_media_suffix
from .helpers import (
    MARKITDOWN_PREFERRED_EXTENSIONS,
    fetch_gist_files,
    is_http_url,
    is_utf8_file,
    looks_like_text_content_type,
    looks_like_windows_drive,
    parse_gist_url,
    parse_target_spec,
    split_spec_symbols,
)
from .url import URLReference

_EXTERNAL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_CONVERTIBLE_CONTENT_TYPES = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
)


def _target_suffix(target: str) -> str:
    if is_http_url(target):
        return Path(urlparse(target).path).suffix.lower()
    return Path(target).suffix.lower()


def _allows_embedded_direct_resolution(target: str, *, plugin_matched: bool) -> bool:
    if plugin_matched:
        return True
    if is_http_url(target):
        suffix = _target_suffix(target)
        return suffix in MARKITDOWN_PREFERRED_EXTENSIONS or is_media_suffix(suffix)
    return not _EXTERNAL_SCHEME_RE.match(target)


def _text_only_materialized_mode(file_item: dict[str, Any]) -> str:
    content_type = str(file_item.get("content_type") or "").split(";", 1)[0].lower()
    filename = str(file_item.get("filename") or "")
    suffix = Path(filename).suffix.lower()
    if content_type and looks_like_text_content_type(content_type):
        return "text"
    if (
        content_type.startswith(("audio/", "image/", "video/"))
        or content_type in _CONVERTIBLE_CONTENT_TYPES
        or suffix in MARKITDOWN_PREFERRED_EXTENSIONS
        or is_media_suffix(suffix)
    ):
        return "convertible"
    if content_type:
        return "skip"
    return "text"


def create_file_references(
    paths,
    ignore_patterns=None,
    format="md",
    label="relative",
    label_suffix: str | None = None,
    include_token_count=False,
    token_target="cl100k_base",
    inject=False,
    depth=5,
    trace_collector=None,
    text_only=False,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    plugin_overrides: dict[str, Any] | None = None,
    arena_overrides: dict | None = None,
    discord_overrides: dict | None = None,
    atproto_overrides: dict | None = None,
    soundcloud_overrides: dict[str, Any] | None = None,
    target_depth: int = 0,
    target_scope: str = "all",
    include_parent: bool = True,
    binary_policy: str = "error",
    _embedded_seen: set[str] | None = None,
):
    """
    Build a list of file references from the specified paths.

    If `inject` is true, {cx::...} markers are resolved before wrapping.
    """
    target_scope = (target_scope or "all").lower()
    if target_scope not in {"first", "all"}:
        raise ValueError("target_scope must be 'first' or 'all'")
    if binary_policy not in {"error", "placeholder", "skip"}:
        raise ValueError("binary_policy must be 'error', 'placeholder', or 'skip'")

    def is_ignored(path, gitignore_patterns):
        from pathspec import PathSpec

        path_spec = PathSpec.from_lines("gitwildmatch", gitignore_patterns)
        return path_spec.match_file(path)

    def get_file_token_count(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return count_tokens(content, target=token_target)["count"]
        except Exception:
            return 0

    def append_binary_reference(file_path, symbols):
        if symbols:
            raise ValueError(
                f"Symbol selection is only supported for text files: {file_path}"
            )
        if binary_policy == "placeholder":
            file_references.append(
                FileExistenceReference(
                    file_path,
                    format=format,
                    label=label,
                    label_suffix=label_suffix,
                    include_token_count=include_token_count,
                    token_target=token_target,
                )
            )
            return
        if binary_policy == "skip":
            return
        raise ValueError(
            f"Unsupported binary file type (not convertible): {file_path}"
        )

    file_references = []
    ignored_files = []
    ignored_folders = {}
    dirs_with_non_ignored_files = set()
    default_ignore_patterns = [
        ".git/",
        ".gitignore",
        ".venv/",
        "venv/",
        "__pycache__/",
        "__init__.py",
        ".tox/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        "*.egg-info/",
        ".gradle/",
        ".cache/",
        "node_modules/",
        "target/",
        "vendor/",
    ]

    all_ignore_patterns = default_ignore_patterns[:]
    if ignore_patterns:
        expanded_ignore_patterns = []
        for pattern in ignore_patterns:
            if "{" in pattern and "}" in pattern:
                expanded_ignore_patterns.extend(brace_expand(pattern))
            else:
                expanded_ignore_patterns.append(pattern)
        all_ignore_patterns.extend(expanded_ignore_patterns)

    expanded_user_patterns = []
    if ignore_patterns:
        for pattern in ignore_patterns:
            if "{" in pattern and "}" in pattern:
                expanded_user_patterns.extend(brace_expand(pattern))
            else:
                expanded_user_patterns.append(pattern)

    expanded_all_paths = []
    for raw_path in paths:
        if "{" in raw_path and "}" in raw_path:
            expanded_all_paths.extend(brace_expand(raw_path))
        else:
            expanded_all_paths.append(raw_path)

    effective_plugin_overrides: dict[str, Any] = {}
    if isinstance(plugin_overrides, dict):
        effective_plugin_overrides.update(plugin_overrides)
    for provider_name, provider_overrides in (
        ("arena", arena_overrides),
        ("discord", discord_overrides),
        ("atproto", atproto_overrides),
        ("soundcloud", soundcloud_overrides),
    ):
        if provider_overrides is not None:
            effective_plugin_overrides.setdefault(provider_name, provider_overrides)

    embedded_seed_targets = [
        str(parse_target_spec(raw_path).get("target", raw_path))
        for raw_path in expanded_all_paths
    ]
    if target_depth > 0 and not include_parent:
        seed_targets = embedded_seed_targets
        if target_scope == "first":
            seed_targets = seed_targets[:1]
        file_references = _resolve_embedded_target_refs(
            seed_targets,
            ignore_patterns=ignore_patterns,
            format=format,
            label=label,
            label_suffix=label_suffix,
            include_token_count=include_token_count,
            token_target=token_target,
            inject=inject,
            depth=depth,
            trace_collector=trace_collector,
            text_only=text_only,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            plugin_overrides=effective_plugin_overrides,
            arena_overrides=arena_overrides,
            discord_overrides=discord_overrides,
            atproto_overrides=atproto_overrides,
            soundcloud_overrides=soundcloud_overrides,
            target_depth=target_depth,
            target_scope=target_scope,
            binary_policy=binary_policy,
            seen=_embedded_seen,
        )
        return {
            "refs": file_references,
            "concatenated": concat_refs(file_references),
            "ignored_files": ignored_files,
            "ignored_folders": {},
        }

    for raw_path in expanded_all_paths:
        spec_opts = parse_target_spec(raw_path)
        target = spec_opts.get("target", raw_path)
        path, symbols = split_spec_symbols(target)
        plugin_refs, plugin_claimed = resolve_plugin_references(
            target,
            format=format,
            label=label,
            label_suffix=label_suffix,
            include_token_count=include_token_count,
            token_target=token_target,
            inject=inject,
            depth=depth,
            trace_collector=trace_collector,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            overrides=effective_plugin_overrides,
        )
        if plugin_refs:
            file_references.extend(plugin_refs)
            continue
        if plugin_claimed:
            continue

        if (
            _EXTERNAL_SCHEME_RE.match(target)
            and not is_http_url(target)
            and not looks_like_windows_drive(target)
            and parse_git_target(target) is None
        ):
            raise ValueError(f"No plugin could resolve external target: {target}")

        if is_http_url(target):
            gist_id = parse_gist_url(target)
            if gist_id:
                gist_files = fetch_gist_files(gist_id)
                if gist_files:
                    for _, raw_url in gist_files:
                        file_references.append(
                            URLReference(
                                raw_url,
                                format=format,
                                label=label,
                                label_suffix=label_suffix,
                                include_token_count=include_token_count,
                                token_target=token_target,
                                inject=inject,
                                depth=depth,
                                trace_collector=trace_collector,
                                use_cache=use_cache,
                                cache_ttl=cache_ttl,
                                refresh_cache=refresh_cache,
                                plugin_overrides=effective_plugin_overrides or None,
                            )
                        )
                    continue
            file_references.append(
                URLReference(
                    target,
                    format=format,
                    label=label,
                    label_suffix=label_suffix,
                    include_token_count=include_token_count,
                    token_target=token_target,
                    inject=inject,
                    depth=depth,
                    trace_collector=trace_collector,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                    plugin_overrides=effective_plugin_overrides or None,
                )
            )
        elif os.path.isfile(path):
            if is_ignored(path, all_ignore_patterns):
                if (
                    expanded_user_patterns
                    and is_ignored(path, expanded_user_patterns)
                    and is_utf8_file(path)
                ):
                    token_count = get_file_token_count(path)
                    ignored_files.append((path, token_count))
            elif is_utf8_file(path) or (
                not text_only
                and (
                    Path(path).suffix.lower() in MARKITDOWN_PREFERRED_EXTENSIONS
                    or is_media_suffix(Path(path).suffix)
                )
            ):
                ranges = None
                if symbols:
                    if not is_utf8_file(path):
                        raise ValueError(
                            f"Symbol selection is only supported for text files: {path}"
                        )
                    try:
                        from ..render.map import find_symbol_ranges

                        match_map = find_symbol_ranges(path, symbols)
                    except Exception:
                        match_map = {}

                    matched = [s for s in symbols if s in match_map]
                    if not matched:
                        print(
                            f"Warning: symbol(s) not found in {path}: {', '.join(symbols)}",
                            file=sys.stderr,
                        )
                        continue
                    symbols = matched
                    ranges = [match_map[s] for s in matched]

                file_references.append(
                    FileReference(
                        path,
                        ranges=ranges,
                        symbols=symbols,
                        format=format,
                        label=label,
                        label_suffix=label_suffix,
                        include_token_count=include_token_count,
                        token_target=token_target,
                        inject=inject,
                        depth=depth,
                        trace_collector=trace_collector,
                        use_cache=use_cache,
                        cache_ttl=cache_ttl,
                        refresh_cache=refresh_cache,
                        plugin_overrides=effective_plugin_overrides or None,
                    )
                )
            else:
                append_binary_reference(path, symbols)
        elif os.path.isdir(path):
            dir_ignored_files = {}
            for root, dirs, files in os.walk(path):
                dirs[:] = [
                    d
                    for d in dirs
                    if not is_ignored(os.path.join(root, d), all_ignore_patterns)
                ]
                for file in files:
                    file_path = os.path.join(root, file)
                    if is_ignored(file_path, all_ignore_patterns):
                        if (
                            expanded_user_patterns
                            and is_ignored(file_path, expanded_user_patterns)
                            and is_utf8_file(file_path)
                        ):
                            token_count = get_file_token_count(file_path)
                            if root not in dir_ignored_files:
                                dir_ignored_files[root] = []
                            dir_ignored_files[root].append((file_path, token_count))
                    elif is_utf8_file(file_path) or (
                        not text_only
                        and (
                            Path(file_path).suffix.lower()
                            in MARKITDOWN_PREFERRED_EXTENSIONS
                            or is_media_suffix(Path(file_path).suffix)
                        )
                    ):
                        dirs_with_non_ignored_files.add(root)
                        parent = os.path.dirname(root)
                        while (
                            parent
                            and parent != path
                            and parent not in dirs_with_non_ignored_files
                        ):
                            dirs_with_non_ignored_files.add(parent)
                            new_parent = os.path.dirname(parent)
                            if new_parent == parent:
                                break
                            parent = new_parent
                        file_references.append(
                            FileReference(
                                file_path,
                                format=format,
                                label=label,
                                label_suffix=label_suffix,
                                include_token_count=include_token_count,
                                token_target=token_target,
                                inject=inject,
                                depth=depth,
                                trace_collector=trace_collector,
                                use_cache=use_cache,
                                cache_ttl=cache_ttl,
                                refresh_cache=refresh_cache,
                                plugin_overrides=effective_plugin_overrides or None,
                            )
                        )
                    else:
                        append_binary_reference(file_path, None)

            for dir_path, files_list in dir_ignored_files.items():
                if dir_path not in dirs_with_non_ignored_files:
                    total_tokens = sum(tokens for _, tokens in files_list)
                    ignored_folders[dir_path] = (len(files_list), total_tokens)
                else:
                    ignored_files.extend(files_list)
        else:
            raise ValueError(f"path not found: {path}")

    consolidated_folders = {}
    for folder_path in sorted(ignored_folders.keys()):
        parent_is_ignored = False
        parent = os.path.dirname(folder_path)
        while parent:
            if parent in ignored_folders:
                parent_is_ignored = True
                break
            new_parent = os.path.dirname(parent)
            if new_parent == parent:
                break
            parent = new_parent
        if not parent_is_ignored:
            consolidated_folders[folder_path] = ignored_folders[folder_path]

    file_references = [
        r for r in file_references if not getattr(r, "cache_miss", False)
    ]

    if target_depth > 0:
        seed_targets = embedded_seed_targets
        if target_scope == "first":
            seed_targets = seed_targets[:1]
        embedded_refs = _resolve_embedded_target_refs(
            seed_targets,
            ignore_patterns=ignore_patterns,
            format=format,
            label=label,
            label_suffix=label_suffix,
            include_token_count=include_token_count,
            token_target=token_target,
            inject=inject,
            depth=depth,
            trace_collector=trace_collector,
            text_only=text_only,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            plugin_overrides=effective_plugin_overrides,
            arena_overrides=arena_overrides,
            discord_overrides=discord_overrides,
            atproto_overrides=atproto_overrides,
            soundcloud_overrides=soundcloud_overrides,
            target_depth=target_depth,
            target_scope=target_scope,
            binary_policy=binary_policy,
            seen=_embedded_seen,
        )
        file_references = (
            [*file_references, *embedded_refs] if include_parent else embedded_refs
        )

    return {
        "refs": file_references,
        "concatenated": concat_refs(file_references),
        "ignored_files": ignored_files,
        "ignored_folders": consolidated_folders,
    }


def concat_refs(file_references):
    """Concatenate references into a single string with the chosen format."""
    return "\n\n".join(ref.output for ref in file_references)


def _resolve_embedded_target_refs(
    seed_targets: list[str],
    *,
    ignore_patterns,
    format: str,
    label: str,
    label_suffix: str | None,
    include_token_count: bool,
    token_target: str,
    inject: bool,
    depth: int,
    trace_collector,
    text_only: bool,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
    plugin_overrides: dict[str, Any],
    arena_overrides: dict | None,
    discord_overrides: dict | None,
    atproto_overrides: dict | None,
    soundcloud_overrides: dict[str, Any] | None,
    target_depth: int,
    target_scope: str,
    binary_policy: str,
    seen: set[str] | None,
) -> list[Any]:
    if target_depth <= 0:
        return []
    if seen is None:
        seen = set(seed_targets)

    refs: list[Any] = []
    frontier = list(seed_targets)
    for depth_index in range(target_depth):
        next_frontier: list[str] = []
        media_jobs = get_payload_media_jobs()
        log_progress(
            "plugins",
            "embedded-list",
            "start",
            detail=f"depth={depth_index + 1}/{target_depth} targets={len(frontier)} jobs={media_jobs}",
        )
        list_tasks = [
            (
                index,
                (
                    lambda current=target: _list_embedded_targets_safely(
                        current,
                        plugin_overrides=plugin_overrides,
                        use_cache=use_cache,
                        cache_ttl=cache_ttl,
                        refresh_cache=refresh_cache,
                    )
                ),
            )
            for index, target in enumerate(frontier)
        ]
        listed_results = run_indexed_tasks_fail_fast(
            list_tasks,
            max_workers=media_jobs,
            semaphore=media_task_semaphore(),
            on_complete=_record_embedded_list_completion,
        )
        child_jobs: list[tuple[int, str]] = []
        encounter_index = 0
        for _, (target, listed, error) in listed_results:
            if error is not None:
                print(
                    f"Warning: embedded target listing failed for {target}: {error}",
                    file=sys.stderr,
                )
                record_progress(
                    "plugins",
                    "embedded-list",
                    "failed",
                    target=target,
                    detail=str(error),
                )
                continue
            provider_name = getattr(listed, "plugin_name", None) or "plugins"
            item_count = len(getattr(listed, "items", ()) or ())
            record_progress(
                str(provider_name),
                "list-targets",
                "processed",
                target=target,
                count=item_count,
            )
            for item in listed.items:
                child = item.get("target")
                if not isinstance(child, str) or not child or child in seen:
                    continue
                if item.get("traverse") is False:
                    continue
                seen.add(child)
                next_frontier.append(child)
                child_jobs.append((encounter_index, child))
                encounter_index += 1

        if child_jobs:
            log_progress(
                "plugins",
                "embedded-resolve",
                "start",
                detail=f"depth={depth_index + 1}/{target_depth} targets={len(child_jobs)} jobs={media_jobs}",
            )
            resolve_tasks = [
                (
                    index,
                    (
                        lambda current=child: _resolve_embedded_child_refs(
                            current,
                            ignore_patterns=ignore_patterns,
                            format=format,
                            label=label,
                            label_suffix=label_suffix,
                            include_token_count=include_token_count,
                            token_target=token_target,
                            inject=inject,
                            depth=depth,
                            trace_collector=trace_collector,
                            text_only=text_only,
                            use_cache=use_cache,
                            cache_ttl=cache_ttl,
                            refresh_cache=refresh_cache,
                            plugin_overrides=plugin_overrides,
                            arena_overrides=arena_overrides,
                            discord_overrides=discord_overrides,
                            atproto_overrides=atproto_overrides,
                            soundcloud_overrides=soundcloud_overrides,
                            binary_policy=binary_policy,
                            seen=seen,
                        )
                    ),
                )
                for index, child in child_jobs
            ]
            child_targets = {index: child for index, child in child_jobs}

            def _record_child_resolved(index, child_refs):
                record_progress(
                    "plugins",
                    "embedded-resolve",
                    "processed",
                    target=child_targets.get(index),
                    count=len(child_refs),
                )

            resolved_total = 0
            for _, child_refs in run_indexed_tasks_fail_fast(
                resolve_tasks,
                max_workers=media_jobs,
                semaphore=media_task_semaphore(),
                on_complete=_record_child_resolved,
            ):
                resolved_total += len(child_refs)
                refs.extend(child_refs)
            log_progress(
                "plugins",
                "embedded-resolve",
                "done",
                detail=f"depth={depth_index + 1}/{target_depth}",
                count=resolved_total,
            )
        else:
            log_progress(
                "plugins",
                "embedded-resolve",
                "done",
                detail=f"depth={depth_index + 1}/{target_depth}",
                count=0,
            )
        log_progress(
            "plugins",
            "embedded-list",
            "done",
            detail=f"depth={depth_index + 1}/{target_depth}",
            count=len(next_frontier),
        )
        if target_scope == "first" and next_frontier:
            next_frontier = next_frontier[:1]
        frontier = next_frontier
        if not frontier:
            break
    return refs


def _record_embedded_list_completion(_index: int, result: Any) -> None:
    target, _listed, error = result
    if error is None:
        record_progress("plugins", "embedded-list", "processed", target=target)


def _list_embedded_targets_safely(
    target: str,
    *,
    plugin_overrides: dict[str, Any],
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> tuple[str, Any | None, Exception | None]:
    try:
        listed = list_plugin_targets(
            target,
            overrides=plugin_overrides,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
    except Exception as exc:
        return target, None, exc
    return target, listed, None


def _resolve_embedded_child_refs(
    target: str,
    *,
    ignore_patterns,
    format: str,
    label: str,
    label_suffix: str | None,
    include_token_count: bool,
    token_target: str,
    inject: bool,
    depth: int,
    trace_collector,
    text_only: bool,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
    plugin_overrides: dict[str, Any],
    arena_overrides: dict | None,
    discord_overrides: dict | None,
    atproto_overrides: dict | None,
    soundcloud_overrides: dict[str, Any] | None,
    binary_policy: str,
    seen: set[str],
) -> list[Any]:
    try:
        plugin_refs, plugin_claimed = resolve_plugin_references(
            target,
            format=format,
            label=label,
            label_suffix=label_suffix,
            include_token_count=include_token_count,
            token_target=token_target,
            inject=inject,
            depth=depth,
            trace_collector=trace_collector,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            overrides=plugin_overrides,
        )
    except Exception as exc:
        print(
            f"Warning: embedded plugin target resolution failed for {target}: {exc}",
            file=sys.stderr,
        )
        plugin_refs = []
        plugin_claimed = False
    if plugin_refs:
        return plugin_refs

    try:
        materialized = materialize_plugin_target(
            target,
            overrides=plugin_overrides,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
    except Exception as exc:
        print(
            f"Warning: embedded target materialization failed for {target}: {exc}",
            file=sys.stderr,
        )
        materialized = None

    plugin_matched = plugin_claimed or bool(
        materialized is not None and materialized.matched
    )

    if materialized is not None and materialized.files:
        refs: list[Any] = []
        with tempfile.TemporaryDirectory(prefix="contextualize-target-") as tmpdir:
            for file_item in materialized.files:
                materialized_mode = (
                    _text_only_materialized_mode(file_item) if text_only else "text"
                )
                if materialized_mode == "skip":
                    continue
                path = Path(tmpdir) / _safe_materialized_filename(
                    str(file_item.get("filename") or "attachment")
                )
                path.write_bytes(file_item["content"])
                try:
                    result = create_file_references(
                        [str(path)],
                        ignore_patterns=ignore_patterns,
                        format=format,
                        label=label,
                        label_suffix=label_suffix,
                        include_token_count=include_token_count,
                        token_target=token_target,
                        inject=inject,
                        depth=depth,
                        trace_collector=trace_collector,
                        text_only=text_only and materialized_mode != "convertible",
                        use_cache=use_cache,
                        cache_ttl=cache_ttl,
                        refresh_cache=refresh_cache,
                        plugin_overrides=plugin_overrides,
                        arena_overrides=arena_overrides,
                        discord_overrides=discord_overrides,
                        atproto_overrides=atproto_overrides,
                        soundcloud_overrides=soundcloud_overrides,
                        target_depth=0,
                        binary_policy="skip" if text_only else binary_policy,
                        _embedded_seen=seen,
                    )
                    refs.extend(result["refs"])
                except Exception as exc:
                    print(
                        f"Warning: embedded materialized target failed for {target}: {exc}",
                        file=sys.stderr,
                    )
        return refs

    if not _allows_embedded_direct_resolution(target, plugin_matched=plugin_matched):
        return []

    try:
        result = create_file_references(
            [target],
            ignore_patterns=ignore_patterns,
            format=format,
            label=label,
            label_suffix=label_suffix,
            include_token_count=include_token_count,
            token_target=token_target,
            inject=inject,
            depth=depth,
            trace_collector=trace_collector,
            text_only=text_only,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            plugin_overrides=plugin_overrides,
            arena_overrides=arena_overrides,
            discord_overrides=discord_overrides,
            atproto_overrides=atproto_overrides,
            soundcloud_overrides=soundcloud_overrides,
            target_depth=0,
            binary_policy=binary_policy,
            _embedded_seen=seen,
        )
    except Exception as exc:
        print(
            f"Warning: embedded target resolution failed for {target}: {exc}",
            file=sys.stderr,
        )
        return []
    return list(result["refs"])


def _safe_materialized_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name or name in {".", ".."}:
        return "attachment"
    return re.sub(r"[^A-Za-z0-9._ -]+", "-", name).strip() or "attachment"
