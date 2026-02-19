"""Internal payload building logic."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any, Dict, List, Optional
from threading import Lock

from ..concurrency import run_indexed_tasks_fail_fast
from ..git.cache import ensure_repo, expand_git_paths
from ..git.target import parse_git_target
from ..runtime import get_payload_media_jobs, get_payload_spec_jobs
from ..render.links import add_markdown_link_refs
from .hydrate import (
    _merge_arena_overrides,
    _merge_atproto_overrides,
    _merge_discord_overrides,
    _parse_arena_config_mapping,
    _parse_atproto_config_mapping,
)
from ..references.discord import parse_discord_config_mapping
from .manifest import coerce_file_spec, component_selectors
from ..references import URLReference, YouTubeReference, create_file_references
from ..references.arena import is_arena_url
from ..references.atproto import (
    AtprotoReference,
    build_atproto_settings,
    is_atproto_url,
    resolve_atproto_url,
)
from ..references.discord import (
    DiscordResolutionError,
    DiscordReference,
    build_discord_settings,
    is_discord_url,
    render_discord_document_with_metadata,
    resolve_discord_url,
    split_discord_document_by_utc_day,
    with_discord_document_rendered,
)
from ..references.helpers import is_http_url, parse_git_url_target, parse_target_spec
from ..references.youtube import is_youtube_url
from ..utils import wrap_text


class _SimpleReference:
    def __init__(
        self,
        output: str,
        *,
        path: str | None = None,
        trace_path: str | None = None,
        content: str | None = None,
    ):
        self.output = output
        self.path = path or ""
        if trace_path:
            self.trace_path = trace_path
        if content is not None:
            self.file_content = content
            self.original_file_content = content


class _MapReference:
    def __init__(self, path: str, output: str, content: str):
        self.path = path
        self.output = output
        self.file_content = content
        self.original_file_content = content
        self.is_map = True


class _TraceReference:
    def __init__(self, path: str, content: str):
        self.path = path
        self.file_content = content
        self.original_file_content = content


_DEFAULT_MAP_TOKENS = 10000
_MIN_MAP_NONEMPTY_LINES = 2


def _arena_settings_cache_key(settings: Any) -> tuple[Any, ...]:
    recurse_users = settings.recurse_users
    recurse_key: tuple[str, ...] | None
    if recurse_users is None:
        recurse_key = None
    else:
        recurse_key = tuple(sorted(str(v) for v in recurse_users))
    return (
        settings.max_depth,
        settings.sort_order,
        settings.max_blocks_per_channel,
        settings.include_descriptions,
        settings.include_comments,
        settings.include_link_image_descriptions,
        settings.include_pdf_content,
        settings.include_media_descriptions,
        recurse_key,
    )


def _arena_channel_identity(
    channel: dict[str, Any], settings_key: tuple[Any, ...]
) -> tuple[Any, ...] | None:
    slug = channel.get("slug")
    if isinstance(slug, str) and slug:
        return ("arena-channel", f"slug:{slug}", settings_key)
    channel_id = channel.get("id")
    if channel_id is not None:
        return ("arena-channel", f"id:{channel_id}", settings_key)
    return None


def _arena_channel_depth(channel_path: str, root_slug: str) -> int:
    parts = [p for p in channel_path.split("/") if p]
    if parts and parts[0] == root_slug:
        parts = parts[1:]
    return len(parts)


@dataclass
class _ArenaChannelTracker:
    canonical: dict[tuple[Any, ...], tuple[int, int]]
    counter: int = 0
    lock: Lock = field(default_factory=Lock)

    def should_expand(self, identity: tuple[Any, ...], *, depth: int) -> bool:
        with self.lock:
            rank = (depth, self.counter)
            self.counter += 1
            current = self.canonical.get(identity)
            if current is None or rank < current:
                self.canonical[identity] = rank
                return True
            return False

    def observe(self, identity: tuple[Any, ...], *, depth: int) -> None:
        with self.lock:
            rank = (depth, self.counter)
            self.counter += 1
            current = self.canonical.get(identity)
            if current is None or rank < current:
                self.canonical[identity] = rank


@dataclass
class _SpecResolution:
    refs: list[Any]
    input_refs: list[Any]
    trace_items: list[Any]
    skipped_paths: set[str]
    skip_impact: dict[str, Any]


def _arena_channel_label(block: dict[str, Any], fallback: str) -> str:
    slug = block.get("slug")
    if isinstance(slug, str) and slug:
        owner = block.get("owner") or block.get("user") or {}
        owner_slug = owner.get("slug")
        if isinstance(owner_slug, str) and owner_slug:
            return f"https://are.na/{owner_slug}/{slug}"
        return f"https://are.na/channel/{slug}"
    channel_id = block.get("id")
    if channel_id is not None:
        return f"https://www.are.na/block/{channel_id}"
    return fallback


def _format_comment(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Comment must be a string, got: {type(value)}")
    text = value.strip()
    if not text:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    import json

    escaped = json.dumps(text, ensure_ascii=False)
    return f"comment={escaped}"


def _combine_comment(comment: str | None, output: str) -> str:
    if comment:
        return f"{comment}\n{output}"
    return output


def _wrapped_url_reference(
    url: str,
    *,
    filename: Optional[str],
    wrap: Optional[str],
    inject: bool,
    depth: int,
    label_suffix: str | None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> _SimpleReference:
    url_ref = URLReference(
        url,
        format="raw",
        label=filename or url,
        inject=inject,
        depth=depth,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    label = filename or url
    if label_suffix:
        label = f"{label} {label_suffix}"
    wrapped = wrap_text(url_ref.output, wrap or "md", label)
    return _SimpleReference(
        wrapped,
        path=url_ref.path,
        content=url_ref.file_content,
    )


def _wrapped_arena_references(
    url: str,
    *,
    filename: Optional[str],
    wrap: Optional[str],
    inject: bool,
    depth: int,
    label_suffix: str | None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    arena_overrides: dict | None = None,
    channel_tracker: _ArenaChannelTracker | None = None,
) -> tuple[list[_SimpleReference], list[Any], list[tuple[str, str, int]]]:
    from ..references.arena import (
        ArenaReference,
        build_arena_settings,
        extract_block_id,
        extract_channel_slug,
        is_arena_channel_url,
        resolve_channel,
        warmup_arena_network_stack,
        _fetch_block,
    )

    settings = build_arena_settings(arena_overrides)
    settings_key = _arena_settings_cache_key(settings)

    refs: list[_SimpleReference] = []
    trace_inputs: list[Any] = []
    channel_trace_items: list[tuple[str, str, int]] = []

    def _append_channel_blocks(
        flat_blocks: list[tuple[str, dict[str, Any]]], slug: str
    ) -> list[str]:
        root_label = url
        channel_contents: list[str] = []
        observed_flattened_channels: set[tuple[str, int]] = set()
        emitted_channel_edges: set[tuple[str, str, int]] = set()

        def _append_channel_edge(child: str, parent: str, depth_value: int) -> None:
            edge = (child, parent, depth_value)
            if edge in emitted_channel_edges:
                return
            emitted_channel_edges.add(edge)
            channel_trace_items.append(edge)

        block_entries: list[tuple[int, str, dict[str, Any]]] = []
        for block_index, (channel_path, block) in enumerate(flat_blocks):
            channel_slug_path = block.get("_channel_slug_path")
            if isinstance(channel_slug_path, list):
                for idx, nested_slug in enumerate(channel_slug_path[1:], start=1):
                    if not isinstance(nested_slug, str) or not nested_slug:
                        continue
                    nested_depth = idx - 1
                    if channel_tracker is not None:
                        observe_key = (nested_slug, nested_depth)
                        if observe_key in observed_flattened_channels:
                            continue
                        observed_flattened_channels.add(observe_key)
                        channel_tracker.observe(
                            ("arena-channel", f"slug:{nested_slug}", settings_key),
                            depth=nested_depth,
                        )
                    parent_slug = channel_slug_path[idx - 1]
                    parent_label = (
                        root_label
                        if idx == 1
                        else f"https://www.are.na/channel/{parent_slug}"
                    )
                    child_label = f"https://www.are.na/channel/{nested_slug}"
                    _append_channel_edge(child_label, parent_label, idx)
            block_type = block.get("type", "")
            is_channel = block_type == "Channel" or block.get("base_type") == "Channel"
            channel_identity = (
                _arena_channel_identity(block, settings_key) if is_channel else None
            )
            if channel_tracker is not None and channel_identity is not None:
                channel_tracker.observe(
                    channel_identity,
                    depth=_arena_channel_depth(channel_path, slug),
                )
                nested_contents = block.get("_nested_contents")
                if isinstance(nested_contents, list) and nested_contents:
                    child_depth = _arena_channel_depth(channel_path, slug) + 1
                    parent_label = (
                        root_label
                        if child_depth == 1
                        else f"{root_label}#{channel_path.rsplit('/', 1)[-1]}"
                    )
                    child_label = _arena_channel_label(
                        block, f"{root_label}#{channel_path}"
                    )
                    _append_channel_edge(child_label, parent_label, child_depth)
            block_entries.append((block_index, channel_path, block))

        def _render_channel_block(
            channel_path: str, block: dict[str, Any]
        ) -> tuple[_SimpleReference, str]:
            arena_ref = ArenaReference(
                url,
                block=block,
                channel_path=channel_path,
                format="raw",
                inject=inject,
                depth=depth,
                include_descriptions=settings.include_descriptions,
                include_comments=settings.include_comments,
                include_link_image_descriptions=settings.include_link_image_descriptions,
                include_pdf_content=settings.include_pdf_content,
                include_media_descriptions=settings.include_media_descriptions,
            )
            label = filename or arena_ref.get_label()
            if label_suffix:
                label = f"{label} {label_suffix}"
            wrapped = wrap_text(arena_ref.output, wrap or "md", label)
            simple_ref = _SimpleReference(
                wrapped,
                path=arena_ref.path,
                trace_path=arena_ref.trace_path,
                content=arena_ref.file_content,
            )
            return simple_ref, arena_ref.file_content

        tasks = [
            (
                block_index,
                (lambda p=channel_path, b=block: _render_channel_block(p, b)),
            )
            for block_index, channel_path, block in block_entries
        ]
        for _, (simple_ref, block_content) in run_indexed_tasks_fail_fast(
            tasks,
            max_workers=get_payload_media_jobs(),
        ):
            refs.append(simple_ref)
            if block_content:
                channel_contents.append(block_content)
        return channel_contents

    if is_arena_channel_url(url):
        slug = extract_channel_slug(url)
        if slug:
            warmup_arena_network_stack()
            root_identity = ("arena-channel", f"slug:{slug}", settings_key)
            should_expand = True
            if channel_tracker is not None:
                should_expand = channel_tracker.should_expand(root_identity, depth=0)

            resolve_settings = settings
            if not should_expand and settings.max_depth > 0:
                resolve_settings = replace(settings, max_depth=0)

            metadata, flat_blocks = resolve_channel(
                slug,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
                settings=resolve_settings,
            )
            channel_contents = _append_channel_blocks(flat_blocks, slug)
            trace_inputs.append(
                _TraceReference(
                    url,
                    "\n\n".join(channel_contents)
                    if channel_contents
                    else (metadata.get("title") or metadata.get("slug") or slug),
                )
            )
    else:
        block_id = extract_block_id(url)
        if block_id is not None:
            block = _fetch_block(block_id)
            arena_ref = ArenaReference(
                url,
                block=block,
                format="raw",
                inject=inject,
                depth=depth,
                include_descriptions=settings.include_descriptions,
                include_comments=settings.include_comments,
                include_link_image_descriptions=settings.include_link_image_descriptions,
                include_pdf_content=settings.include_pdf_content,
                include_media_descriptions=settings.include_media_descriptions,
            )
            label = filename or arena_ref.get_label()
            if label_suffix:
                label = f"{label} {label_suffix}"
            wrapped = wrap_text(arena_ref.output, wrap or "md", label)
            refs.append(
                _SimpleReference(
                    wrapped,
                    path=arena_ref.path,
                    trace_path=arena_ref.trace_path,
                    content=arena_ref.file_content,
                )
            )
            trace_inputs.append(refs[-1])
    return refs, trace_inputs, channel_trace_items


def _wrapped_youtube_reference(
    url: str,
    *,
    filename: Optional[str],
    wrap: Optional[str],
    inject: bool,
    depth: int,
    label_suffix: str | None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> _SimpleReference:
    yt_ref = YouTubeReference(
        url,
        format="raw",
        label=filename or url,
        inject=inject,
        depth=depth,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    label = filename or url
    if label_suffix:
        label = f"{label} {label_suffix}"
    wrapped = wrap_text(yt_ref.output, wrap or "md", label)
    return _SimpleReference(
        wrapped,
        path=yt_ref.path,
        content=yt_ref.file_content,
    )


def _wrapped_discord_references(
    url: str,
    *,
    filename: Optional[str],
    wrap: Optional[str],
    inject: bool,
    depth: int,
    label_suffix: str | None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    discord_overrides: dict | None = None,
) -> tuple[list[_SimpleReference], list[Any], list[tuple[str, str, int]]]:
    settings = build_discord_settings(discord_overrides)
    try:
        documents = resolve_discord_url(
            url,
            settings=settings,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
    except ValueError as exc:
        if isinstance(exc, DiscordResolutionError) and exc.is_skippable:
            print(
                f"Warning: skipping Discord URL: {url} ({exc})",
                file=sys.stderr,
                flush=True,
            )
            return [], [], []
        raise

    refs: list[_SimpleReference] = []
    trace_inputs: list[Any] = []
    trace_items: list[tuple[str, str, int]] = []

    for document in documents:
        day_documents = split_discord_document_by_utc_day(document, settings=settings)
        prepared_documents = [
            with_discord_document_rendered(
                day_document,
                rendered=render_discord_document_with_metadata(
                    day_document,
                    settings=settings,
                    source_url=url,
                    include_message_bounds=False,
                ),
            )
            for day_document in day_documents
        ]
        for prepared_document in prepared_documents:
            discord_ref = DiscordReference(
                url,
                document=prepared_document,
                format="raw",
                inject=inject,
                depth=depth,
            )
            label = filename or discord_ref.get_label()
            if label_suffix:
                label = f"{label} {label_suffix}"
            wrapped = wrap_text(discord_ref.output, wrap or "md", label)
            simple_ref = _SimpleReference(
                wrapped,
                path=discord_ref.path,
                trace_path=discord_ref.trace_path,
                content=discord_ref.file_content,
            )
            refs.append(simple_ref)
            trace_inputs.append(simple_ref)

            if prepared_document.thread_id and prepared_document.parent_channel_id:
                parent_trace = f"discord/{prepared_document.guild_id}/{prepared_document.parent_channel_id}"
                trace_items.append((discord_ref.trace_path, parent_trace, 1))

    return refs, trace_inputs, trace_items


def _wrapped_atproto_references(
    url: str,
    *,
    filename: Optional[str],
    wrap: Optional[str],
    inject: bool,
    depth: int,
    label_suffix: str | None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    atproto_overrides: dict[str, Any] | None = None,
) -> tuple[list[_SimpleReference], list[Any], list[tuple[str, str, int]]]:
    settings = build_atproto_settings(atproto_overrides)
    documents = resolve_atproto_url(
        url,
        settings=settings,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )

    refs: list[_SimpleReference] = []
    trace_inputs: list[Any] = []
    trace_items: list[tuple[str, str, int]] = []
    for document in documents:
        atproto_ref = AtprotoReference(
            url,
            document=document,
            format="raw",
            inject=inject,
            depth=depth,
        )
        label = filename or atproto_ref.get_label()
        if label_suffix:
            label = f"{label} {label_suffix}"
        wrapped = wrap_text(atproto_ref.output, wrap or "md", label)
        simple_ref = _SimpleReference(
            wrapped,
            path=atproto_ref.path,
            trace_path=atproto_ref.trace_path,
            content=atproto_ref.file_content,
        )
        refs.append(simple_ref)
        trace_inputs.append(simple_ref)
    return refs, trace_inputs, trace_items


def _resolve_spec_to_paths(
    raw_spec: str,
    base_dir: str,
    *,
    component_name: str,
) -> list[str]:
    spec = os.path.expanduser(raw_spec)

    if is_http_url(spec):
        opts = parse_target_spec(spec)
        url = opts.get("target", spec)
        tgt = parse_git_url_target(url)
        if not tgt:
            return []
        repo_dir = ensure_repo(tgt)
        paths = [repo_dir] if not tgt.path else expand_git_paths(repo_dir, tgt.path)
    else:
        tgt = parse_git_target(spec)
        if tgt:
            repo_dir = ensure_repo(tgt)
            paths = [repo_dir] if not tgt.path else expand_git_paths(repo_dir, tgt.path)
        else:
            base = "" if os.path.isabs(spec) else base_dir
            paths = expand_git_paths(base, spec)

    resolved = []
    for full in paths:
        if not os.path.exists(full):
            raise FileNotFoundError(
                f"Component '{component_name}' path not found: {full}"
            )
        resolved.append(full)
    return resolved


def _compute_map_root(paths: list[str]) -> str | None:
    if not paths:
        return None
    if len(paths) == 1:
        path = paths[0]
        if os.path.isdir(path):
            return path
        return os.path.dirname(path)
    try:
        return os.path.commonpath(paths)
    except ValueError:
        return None


def _generate_repo_map_output(
    paths: list[str],
    *,
    token_target: str,
) -> str:
    from ..render.map import generate_repo_map_data

    root = _compute_map_root(paths)
    result = generate_repo_map_data(
        paths,
        _DEFAULT_MAP_TOKENS,
        "raw",
        ignore=None,
        annotate_tokens=False,
        token_target=token_target,
        root=root,
    )
    if "error" in result:
        return result["error"]
    return result["repo_map"]


def _map_output_is_compatible(output: str) -> bool:
    lines = [line for line in output.splitlines() if line.strip()]
    return len(lines) >= _MIN_MAP_NONEMPTY_LINES


def _build_map_reference(
    label: str,
    output: str,
    *,
    label_suffix: str | None,
    token_target: str,
) -> _MapReference:
    from ..render.text import process_text

    rendered = process_text(
        output,
        format="md",
        label=label,
        label_suffix=label_suffix,
        token_target=token_target,
        include_token_count=False,
    )
    return _MapReference(label, rendered, output)


def _resolve_spec_to_seed_refs(
    raw_spec: Any,
    file_opts: Dict[str, Any],
    base_dir: str,
    *,
    inject: bool,
    depth: int,
    component_name: str,
    label_suffix: str | None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    arena_overrides: dict | None = None,
    atproto_overrides: dict[str, Any] | None = None,
    discord_overrides: dict | None = None,
    channel_tracker: _ArenaChannelTracker | None = None,
) -> tuple[List[Any], List[Any], List[tuple[str, str, int]]]:
    spec = os.path.expanduser(raw_spec)
    seed_refs: List[Any] = []
    trace_inputs: List[Any] = []
    channel_trace_items: list[tuple[str, str, int]] = []

    if is_http_url(spec):
        opts = parse_target_spec(spec)
        url = opts.get("target", spec)
        filename = file_opts.get("filename") or opts.get("filename")
        wrap = file_opts.get("wrap") or opts.get("wrap")

        tgt = parse_git_url_target(url)
        if tgt:
            repo_dir = ensure_repo(tgt)
            paths = [repo_dir] if not tgt.path else expand_git_paths(repo_dir, tgt.path)
            for full in paths:
                if not os.path.exists(full):
                    raise FileNotFoundError(
                        f"Component '{component_name}' path not found: {full}"
                    )
                refs = create_file_references(
                    [full],
                    ignore_patterns=None,
                    format="md",
                    label="relative",
                    label_suffix=label_suffix,
                    inject=inject,
                    depth=depth,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                )["refs"]
                seed_refs.extend(refs)
        elif is_atproto_url(url):
            atproto_refs, atproto_trace_inputs, atproto_trace_items = (
                _wrapped_atproto_references(
                    url,
                    filename=filename,
                    wrap=wrap,
                    inject=inject,
                    depth=depth,
                    label_suffix=label_suffix,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                    atproto_overrides=atproto_overrides,
                )
            )
            seed_refs.extend(atproto_refs)
            trace_inputs.extend(atproto_trace_inputs)
            channel_trace_items.extend(atproto_trace_items)
        elif is_youtube_url(url):
            seed_refs.append(
                _wrapped_youtube_reference(
                    url,
                    filename=filename,
                    wrap=wrap,
                    inject=inject,
                    depth=depth,
                    label_suffix=label_suffix,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                )
            )
        elif is_arena_url(url):
            arena_refs, arena_trace_inputs, arena_trace_items = (
                _wrapped_arena_references(
                    url,
                    filename=filename,
                    wrap=wrap,
                    inject=inject,
                    depth=depth,
                    label_suffix=label_suffix,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                    arena_overrides=arena_overrides,
                    channel_tracker=channel_tracker,
                )
            )
            seed_refs.extend(arena_refs)
            trace_inputs.extend(arena_trace_inputs)
            channel_trace_items.extend(arena_trace_items)
        elif is_discord_url(url):
            discord_refs, discord_trace_inputs, discord_trace_items = (
                _wrapped_discord_references(
                    url,
                    filename=filename,
                    wrap=wrap,
                    inject=inject,
                    depth=depth,
                    label_suffix=label_suffix,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                    discord_overrides=discord_overrides,
                )
            )
            seed_refs.extend(discord_refs)
            trace_inputs.extend(discord_trace_inputs)
            channel_trace_items.extend(discord_trace_items)
        else:
            seed_refs.append(
                _wrapped_url_reference(
                    url,
                    filename=filename,
                    wrap=wrap,
                    inject=inject,
                    depth=depth,
                    label_suffix=label_suffix,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                )
            )
        if (
            not is_arena_url(url)
            and not is_discord_url(url)
            and not is_atproto_url(url)
        ):
            trace_inputs.extend([r for r in seed_refs if hasattr(r, "path")])
        return seed_refs, trace_inputs, channel_trace_items

    if is_atproto_url(spec):
        filename = file_opts.get("filename")
        wrap = file_opts.get("wrap")
        atproto_refs, atproto_trace_inputs, atproto_trace_items = (
            _wrapped_atproto_references(
                spec,
                filename=filename,
                wrap=wrap,
                inject=inject,
                depth=depth,
                label_suffix=label_suffix,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
                atproto_overrides=atproto_overrides,
            )
        )
        seed_refs.extend(atproto_refs)
        trace_inputs.extend(atproto_trace_inputs)
        channel_trace_items.extend(atproto_trace_items)
        return seed_refs, trace_inputs, channel_trace_items

    tgt = parse_git_target(spec)
    if tgt:
        repo_dir = ensure_repo(tgt)
        paths = [repo_dir] if not tgt.path else expand_git_paths(repo_dir, tgt.path)
    else:
        base = "" if os.path.isabs(spec) else base_dir
        paths = expand_git_paths(base, spec)

    for full in paths:
        if not os.path.exists(full):
            raise FileNotFoundError(
                f"Component '{component_name}' path not found: {full}"
            )
        custom_label = file_opts.get("filename")
        if custom_label and os.path.isfile(full):
            from ..references import FileReference

            fr = FileReference(
                full,
                format="md",
                label=str(custom_label),
                label_suffix=label_suffix,
                inject=inject,
                depth=depth,
            )
            seed_refs.append(fr)
        else:
            refs = create_file_references(
                [full],
                ignore_patterns=None,
                format="md",
                label="relative",
                label_suffix=label_suffix,
                inject=inject,
                depth=depth,
            )["refs"]
            seed_refs.extend(refs)

    trace_inputs.extend([r for r in seed_refs if hasattr(r, "path")])
    return seed_refs, trace_inputs, channel_trace_items


def _render_attachment_block(
    name: str,
    refs: List[Any],
    wrap_mode: Optional[str],
    prefix: str,
    suffix: str,
    comment: str | None,
) -> str:
    """Render an <attachment> block with optional wrap/prefix/suffix."""
    comment_attr = f" {comment}" if comment else ""
    attachment_lines = [f'<attachment label="{name}"{comment_attr}>']
    for idx, ref in enumerate(refs):
        attachment_lines.append(ref.output)
        if idx < len(refs) - 1:
            attachment_lines.append("")
    attachment_lines.append("</attachment>")
    inner = "\n".join(attachment_lines)

    if wrap_mode:
        if wrap_mode.lower() == "md":
            inner = "```\n" + inner + "\n```"
        else:
            inner = f"<{wrap_mode}>\n{inner}\n</{wrap_mode}>"

    block_lines: List[str] = []
    if prefix:
        block_lines.append(prefix)
    block_lines.append(inner)
    if suffix:
        block_lines.append(suffix)
    return "\n".join(block_lines)


def _append_refs(target: list[Any], refs: list[Any]) -> None:
    target.extend(refs)


def _resolve_spec(
    *,
    raw_spec: str,
    file_opts: Dict[str, Any],
    base_dir: str,
    inject: bool,
    depth: int,
    component_name: str,
    item_comment: str | None,
    map_component: bool,
    token_target: str,
    comp_link_depth: int,
    per_file_link_depth: int | None,
    per_file_link_scope: str,
    resolved_link_skip: list[str],
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
    effective_arena_overrides: dict | None,
    effective_atproto_overrides: dict[str, Any] | None,
    effective_discord_overrides: dict | None,
    channel_tracker: _ArenaChannelTracker | None,
) -> _SpecResolution:
    if map_component:
        map_paths = _resolve_spec_to_paths(
            raw_spec,
            base_dir,
            component_name=component_name,
        )
        if map_paths:
            map_output = _generate_repo_map_output(map_paths, token_target=token_target)
            if _map_output_is_compatible(map_output):
                map_ref = _build_map_reference(
                    raw_spec,
                    map_output,
                    label_suffix=item_comment,
                    token_target=token_target,
                )
                return _SpecResolution(
                    refs=[map_ref],
                    input_refs=[map_ref],
                    trace_items=[],
                    skipped_paths=set(),
                    skip_impact={},
                )

    seed_refs, spec_trace_inputs, arena_channel_trace_items = (
        _resolve_spec_to_seed_refs(
            raw_spec,
            file_opts,
            base_dir,
            inject=inject,
            depth=depth,
            component_name=component_name,
            label_suffix=item_comment,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            arena_overrides=effective_arena_overrides,
            atproto_overrides=effective_atproto_overrides,
            discord_overrides=effective_discord_overrides,
            channel_tracker=channel_tracker,
        )
    )

    trace_items: list[Any] = list(arena_channel_trace_items)
    effective_link_depth = int(
        per_file_link_depth if per_file_link_depth is not None else comp_link_depth
    )
    if effective_link_depth <= 0:
        return _SpecResolution(
            refs=seed_refs,
            input_refs=spec_trace_inputs,
            trace_items=trace_items,
            skipped_paths=set(),
            skip_impact={},
        )

    expanded_refs, comp_trace_items, comp_skip_impact = add_markdown_link_refs(
        seed_refs,
        link_depth=effective_link_depth,
        scope=per_file_link_scope,
        format_="md",
        label="relative",
        inject=inject,
        link_skip=resolved_link_skip if resolved_link_skip else None,
    )
    trace_items.extend(comp_trace_items)
    skipped_paths: set[str] = set()
    if resolved_link_skip:
        for skip_path in resolved_link_skip:
            abs_skip_path = os.path.abspath(skip_path)
            if os.path.exists(abs_skip_path):
                skipped_paths.add(abs_skip_path)
    return _SpecResolution(
        refs=expanded_refs,
        input_refs=spec_trace_inputs,
        trace_items=trace_items,
        skipped_paths=skipped_paths,
        skip_impact=comp_skip_impact or {},
    )


def _should_map_component(
    selectors: set[str],
    *,
    map_mode: bool,
    map_keys: set[str],
) -> bool:
    if not selectors:
        return False
    if map_mode:
        if map_keys:
            return bool(selectors & map_keys)
        return True
    if map_keys:
        return bool(selectors & map_keys)
    return False


def build_payload_impl(
    components: List[Dict[str, Any]],
    base_dir: str,
    *,
    inject: bool = False,
    depth: int = 5,
    link_depth_default: int = 0,
    link_scope_default: str = "all",
    link_skip_default: List[str] = None,
    exclude_keys: list[str] | None = None,
    map_mode: bool = False,
    map_keys: list[str] | None = None,
    token_target: str = "cl100k_base",
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    arena_overrides: dict | None = None,
    atproto_overrides: dict[str, Any] | None = None,
    discord_overrides: dict | None = None,
):
    parts: List[str] = []
    all_input_refs = []
    all_trace_items = []
    all_skipped_paths = set()
    all_skip_impact = {}
    exclude_set = {k.strip() for k in (exclude_keys or []) if k and k.strip()}
    map_key_set = {k.strip() for k in (map_keys or []) if k and k.strip()}
    overlap = sorted(exclude_set & map_key_set)
    if overlap:
        names = ", ".join(overlap)
        raise ValueError(f"Components cannot be both mapped and excluded: {names}")
    channel_tracker = _ArenaChannelTracker(canonical={})
    spec_jobs = get_payload_spec_jobs()

    for comp in components:
        selectors = component_selectors(comp)
        if selectors and exclude_set and selectors & exclude_set:
            continue
        name = comp.get("name")
        wrap_mode = comp.get("wrap")
        component_comment = _format_comment(comp.get("comment"))

        if "text" in comp:
            text = comp["text"].rstrip()
            if component_comment:
                text = (
                    _combine_comment(component_comment, text)
                    if text
                    else component_comment
                )
            if wrap_mode:
                if wrap_mode.lower() == "md":
                    text = "```\n" + text + "\n```"
                else:
                    text = f"<{wrap_mode}>\n{text}\n</{wrap_mode}>"
            parts.append(text)
            continue

        files = comp.get("files")
        if not name or not files:
            raise ValueError(
                f"Component must have either 'text' or both 'name' & 'files': {comp}"
            )

        prefix = comp.get("prefix", "").rstrip()
        suffix = comp.get("suffix", "").lstrip()
        map_component = _should_map_component(
            selectors, map_mode=map_mode, map_keys=map_key_set
        )

        comp_link_depth = int(comp.get("link-depth", link_depth_default) or 0)
        comp_link_scope = (comp.get("link-scope", link_scope_default) or "all").lower()
        component_arena_overrides = _parse_arena_config_mapping(
            comp.get("arena"), prefix=f"component '{name}'.arena"
        )
        component_atproto_overrides = _parse_atproto_config_mapping(
            comp.get("atproto"),
            prefix=f"component '{name}'.atproto",
        )
        component_discord_overrides = parse_discord_config_mapping(
            comp.get("discord"), prefix=f"component '{name}'.discord"
        )

        comp_link_skip = comp.get("link-skip", link_skip_default)
        if comp_link_skip is None:
            comp_link_skip = []
        elif isinstance(comp_link_skip, str):
            comp_link_skip = [comp_link_skip]

        resolved_link_skip_default: List[str] = []
        for skip_path in comp_link_skip:
            skip_path = os.path.expanduser(skip_path)
            if not os.path.isabs(skip_path):
                skip_path = os.path.join(base_dir, skip_path)
            resolved_link_skip_default.append(skip_path)

        refs_for_attachment = []
        input_refs_for_comp = []
        spec_tasks: list[tuple[int, Any]] = []

        for spec_index, spec in enumerate(files, 1):
            spec, file_opts = coerce_file_spec(spec)
            raw_spec = spec
            item_comment = _format_comment(file_opts.get("comment"))
            file_arena_overrides = _parse_arena_config_mapping(
                file_opts.get("arena"),
                prefix=f"component '{name}' file[{spec_index}].arena",
            )
            effective_arena_overrides = _merge_arena_overrides(
                arena_overrides,
                component_arena_overrides,
                file_arena_overrides,
            )
            file_atproto_overrides = _parse_atproto_config_mapping(
                file_opts.get("atproto"),
                prefix=f"component '{name}' file[{spec_index}].atproto",
            )
            effective_atproto_overrides = _merge_atproto_overrides(
                atproto_overrides,
                component_atproto_overrides,
                file_atproto_overrides,
            )
            file_discord_overrides = parse_discord_config_mapping(
                file_opts.get("discord"),
                prefix=f"component '{name}' file[{spec_index}].discord",
            )
            effective_discord_overrides = _merge_discord_overrides(
                discord_overrides,
                component_discord_overrides,
                file_discord_overrides,
            )

            per_file_link_depth = file_opts.get("link-depth")
            per_file_link_scope = (
                (file_opts.get("link-scope") or comp_link_scope).lower()
                if file_opts
                else comp_link_scope
            )
            per_file_link_skip = file_opts.get("link-skip") if file_opts else None
            resolved_link_skip = list(resolved_link_skip_default)
            if per_file_link_skip:
                if isinstance(per_file_link_skip, str):
                    per_file_link_skip = [per_file_link_skip]
                for skip_path in per_file_link_skip:
                    skip_path = os.path.expanduser(skip_path)
                    if not os.path.isabs(skip_path):
                        skip_path = os.path.join(base_dir, skip_path)
                    resolved_link_skip.append(skip_path)

            spec_tasks.append(
                (
                    spec_index,
                    lambda rs=raw_spec, fo=file_opts, ic=item_comment, pld=per_file_link_depth, pls=per_file_link_scope, rls=resolved_link_skip, eao=effective_arena_overrides, eatpo=effective_atproto_overrides, edo=effective_discord_overrides: (
                        _resolve_spec(
                            raw_spec=rs,
                            file_opts=fo,
                            base_dir=base_dir,
                            inject=inject,
                            depth=depth,
                            component_name=name,
                            item_comment=ic,
                            map_component=map_component,
                            token_target=token_target,
                            comp_link_depth=comp_link_depth,
                            per_file_link_depth=pld,
                            per_file_link_scope=pls,
                            resolved_link_skip=rls,
                            use_cache=use_cache,
                            cache_ttl=cache_ttl,
                            refresh_cache=refresh_cache,
                            effective_arena_overrides=eao,
                            effective_atproto_overrides=eatpo,
                            effective_discord_overrides=edo,
                            channel_tracker=channel_tracker,
                        )
                    ),
                )
            )

        for _, spec_result in run_indexed_tasks_fail_fast(
            spec_tasks,
            max_workers=spec_jobs,
        ):
            _append_refs(refs_for_attachment, spec_result.refs)
            input_refs_for_comp.extend(spec_result.input_refs)
            all_trace_items.extend(spec_result.trace_items)
            all_skipped_paths.update(spec_result.skipped_paths)
            if spec_result.skip_impact:
                all_skip_impact.update(spec_result.skip_impact)

        all_input_refs.extend(input_refs_for_comp)

        if refs_for_attachment or prefix or suffix or component_comment:
            parts.append(
                _render_attachment_block(
                    name,
                    refs_for_attachment,
                    wrap_mode,
                    prefix,
                    suffix,
                    component_comment,
                )
            )

    return (
        "\n\n".join(parts),
        all_input_refs,
        all_trace_items,
        base_dir,
        list(all_skipped_paths),
        all_skip_impact,
    )
