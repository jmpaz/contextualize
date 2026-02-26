"""Factory function for resolving targets into references."""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

from ..concurrency import run_indexed_tasks_fail_fast
from ..utils import brace_expand, count_tokens
from .file import FileReference
from .helpers import (
    MARKITDOWN_PREFERRED_EXTENSIONS,
    fetch_gist_files,
    is_http_url,
    is_utf8_file,
    parse_gist_url,
    parse_target_spec,
    split_spec_symbols,
)
from .url import URLReference
from .arena import (
    ArenaReference,
    build_arena_settings,
    is_arena_block_url,
    is_arena_channel_url,
)
from .atproto import (
    AtprotoReference,
    build_atproto_settings,
    is_atproto_url,
    resolve_atproto_url,
)
from .discord import (
    DiscordResolutionError,
    DiscordReference,
    build_discord_settings,
    is_discord_url,
    render_discord_document_with_metadata,
    resolve_discord_url,
    split_discord_document_by_utc_day,
    with_discord_document_rendered,
)
from .youtube import YouTubeReference, is_youtube_url


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
    arena_overrides: dict | None = None,
):
    """
    Build a list of file references from the specified paths.

    If `inject` is true, {cx::...} markers are resolved before wrapping.
    """

    def is_ignored(path, gitignore_patterns):
        from pathspec import PathSpec

        path_spec = PathSpec.from_lines("gitwildmatch", gitignore_patterns)
        return path_spec.match_file(path)

    def get_file_token_count(file_path):
        """Get token count for a file if it's readable."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return count_tokens(content, target=token_target)["count"]
        except Exception:
            return 0

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

    arena_settings = None

    def resolve_arena_settings():
        nonlocal arena_settings
        if arena_settings is None:
            arena_settings = build_arena_settings(arena_overrides)
        return arena_settings

    for raw_path in expanded_all_paths:
        spec_opts = parse_target_spec(raw_path)
        target = spec_opts.get("target", raw_path)
        path, symbols = split_spec_symbols(target)

        if is_youtube_url(target):
            file_references.append(
                YouTubeReference(
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
                )
            )
            continue

        if is_atproto_url(target):
            atproto_settings = build_atproto_settings()
            atproto_documents = resolve_atproto_url(
                target,
                settings=atproto_settings,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
            )
            for document in atproto_documents:
                file_references.append(
                    AtprotoReference(
                        target,
                        document=document,
                        format=format,
                        label=label,
                        label_suffix=label_suffix,
                        include_token_count=include_token_count,
                        token_target=token_target,
                        inject=inject,
                        depth=depth,
                        trace_collector=trace_collector,
                    )
                )
            continue

        if is_arena_channel_url(target):
            from .arena import (
                extract_channel_slug,
                resolve_channel,
                warmup_arena_network_stack,
            )
            from ..runtime import get_payload_media_jobs

            slug = extract_channel_slug(target)
            if slug:
                warmup_arena_network_stack()
                settings = resolve_arena_settings()
                metadata, flat_blocks = resolve_channel(
                    slug,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                    settings=settings,
                )
                tasks = [
                    (
                        index,
                        (
                            lambda cp=channel_path, b=block: ArenaReference(
                                target,
                                block=b,
                                channel_path=cp,
                                format=format,
                                label=label,
                                label_suffix=label_suffix,
                                include_token_count=include_token_count,
                                token_target=token_target,
                                inject=inject,
                                depth=depth,
                                trace_collector=trace_collector,
                                include_descriptions=settings.include_descriptions,
                                include_comments=settings.include_comments,
                                include_link_image_descriptions=settings.include_link_image_descriptions,
                                include_pdf_content=settings.include_pdf_content,
                                include_media_descriptions=settings.include_media_descriptions,
                            )
                        ),
                    )
                    for index, (channel_path, block) in enumerate(flat_blocks)
                ]
                for _, arena_ref in run_indexed_tasks_fail_fast(
                    tasks,
                    max_workers=get_payload_media_jobs(),
                ):
                    file_references.append(arena_ref)
                continue

        if is_arena_block_url(target):
            from .arena import extract_block_id, _fetch_block

            block_id = extract_block_id(target)
            if block_id is not None:
                block = _fetch_block(block_id)
                settings = resolve_arena_settings()
                file_references.append(
                    ArenaReference(
                        target,
                        block=block,
                        format=format,
                        label=label,
                        label_suffix=label_suffix,
                        include_token_count=include_token_count,
                        token_target=token_target,
                        inject=inject,
                        depth=depth,
                        trace_collector=trace_collector,
                        include_descriptions=settings.include_descriptions,
                        include_comments=settings.include_comments,
                        include_link_image_descriptions=settings.include_link_image_descriptions,
                        include_pdf_content=settings.include_pdf_content,
                        include_media_descriptions=settings.include_media_descriptions,
                    )
                )
                continue

        if is_discord_url(target):
            discord_settings = build_discord_settings()
            try:
                discord_docs = resolve_discord_url(
                    target,
                    settings=discord_settings,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                )
            except ValueError as exc:
                if isinstance(exc, DiscordResolutionError) and exc.is_skippable:
                    print(
                        f"Warning: skipping Discord URL: {target} ({exc})",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                raise
            prepared_docs = []
            for document in discord_docs:
                day_docs = split_discord_document_by_utc_day(
                    document, settings=discord_settings
                )
                for day_doc in day_docs:
                    rendered = render_discord_document_with_metadata(
                        day_doc,
                        settings=discord_settings,
                        source_url=target,
                        include_message_bounds=False,
                    )
                    prepared_docs.append(
                        with_discord_document_rendered(day_doc, rendered=rendered)
                    )
            for document in prepared_docs:
                file_references.append(
                    DiscordReference(
                        target,
                        document=document,
                        format=format,
                        label=label,
                        label_suffix=label_suffix,
                        include_token_count=include_token_count,
                        token_target=token_target,
                        inject=inject,
                        depth=depth,
                        trace_collector=trace_collector,
                    )
                )
            continue

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
                and Path(path).suffix.lower() in MARKITDOWN_PREFERRED_EXTENSIONS
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
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported binary file type (not convertible): {path}"
                )
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
                        and Path(file_path).suffix.lower()
                        in MARKITDOWN_PREFERRED_EXTENSIONS
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
                            )
                        )

            for dir_path, files_list in dir_ignored_files.items():
                if dir_path not in dirs_with_non_ignored_files:
                    total_tokens = sum(tokens for _, tokens in files_list)
                    ignored_folders[dir_path] = (len(files_list), total_tokens)
                else:
                    ignored_files.extend(files_list)

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

    return {
        "refs": file_references,
        "concatenated": concat_refs(file_references),
        "ignored_files": ignored_files,
        "ignored_folders": consolidated_folders,
    }


resolve = create_file_references


def concat_refs(file_references):
    """Concatenate references into a single string with the chosen format."""
    return "\n\n".join(ref.output for ref in file_references)
