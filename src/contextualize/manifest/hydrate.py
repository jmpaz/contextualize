from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..concurrency import run_indexed_tasks_fail_fast
from ..git.cache import ensure_repo, expand_git_paths
from ..git.target import parse_git_target
from ..git.rev import get_repo_root, read_gitignore_patterns
from ..references import URLReference, YouTubeReference, create_file_references
from ..references.arena import is_arena_channel_url, is_arena_url
from ..references.atproto import (
    atproto_settings_cache_key,
    build_atproto_settings,
    is_atproto_url,
    resolve_atproto_url,
)
from ..references.discord import (
    DiscordResolutionError,
    build_discord_settings,
    discord_document_timestamps,
    discord_overrides_cache_key,
    discord_settings_cache_key,
    is_discord_url,
    merge_discord_overrides,
    parse_discord_url,
    parse_discord_config_mapping,
    render_discord_document_with_metadata,
    resolve_discord_url,
    split_discord_document_by_utc_day,
)
from ..runtime import get_payload_media_jobs, get_payload_spec_jobs
from ..references.helpers import (
    fetch_gist_files,
    is_http_url,
    parse_timestamp_or_duration_value,
    parse_gist_url,
    parse_git_url_target,
    parse_target_spec,
    resolve_symbol_ranges,
    split_spec_symbols,
)
from ..references.youtube import extract_video_id, is_youtube_url
from ..utils import extract_ranges
from .manifest import (
    GROUP_BASE_KEY,
    GROUP_PATH_KEY,
    coerce_file_spec,
    normalize_components,
)


@dataclass(frozen=True)
class HydrateOverrides:
    context_dir: str | None = None
    access: str | None = None
    path_strategy: str | None = None
    agents_prompt: str | None = None
    agents_filenames: tuple[str, ...] = ()
    omit_meta: bool = False
    copy: bool = False
    use_cache: bool | None = None
    cache_ttl: timedelta | None = None
    refresh_cache: bool = False


@dataclass(frozen=True)
class HydrateResult:
    context_dir: str
    component_count: int
    file_count: int
    manifest_written: bool


@dataclass
class ResolvedItem:
    source_type: str
    source_ref: str
    source_rev: str | None
    source_path: str
    context_subpath: str
    content: str
    manifest_spec: str
    alias: str | None = None
    source_full_path: str | None = None
    source_created: str | None = None
    source_modified: str | None = None
    dir_created: str | None = None
    dir_modified: str | None = None
    arena_kind: str | None = None
    arena_channel_id: str | None = None
    arena_depth: int | None = None
    arena_settings_key: tuple[Any, ...] | None = None
    discord_kind: str | None = None
    discord_scope_id: str | None = None
    discord_depth: int | None = None
    discord_settings_key: tuple[Any, ...] | None = None
    atproto_kind: str | None = None
    atproto_settings_key: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class HydratePlan:
    context_dir: Path
    files_to_write: list[tuple[Path, str]]
    files_to_symlink: list[tuple[Path, Path]]
    used_paths: set[str]
    component_count: int
    include_meta: bool
    access: str
    file_timestamps: dict[Path, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class _ArenaPendingWrite:
    rel_path: Path
    content: str
    should_write: bool
    item: ResolvedItem
    encounter_index: int


def apply_hydration_plan(plan: HydratePlan) -> HydrateResult:
    _write_files(plan.files_to_write)
    _create_symlinks(plan.files_to_symlink)
    if plan.file_timestamps:
        _apply_timestamps(plan.file_timestamps)
    if plan.access == "read-only":
        _apply_read_only(plan.context_dir)

    written_paths = {path.as_posix() for path, _ in plan.files_to_write}
    symlinked_paths = {path.as_posix() for path, _ in plan.files_to_symlink}
    file_count = len(written_paths | symlinked_paths)
    return HydrateResult(
        context_dir=str(plan.context_dir),
        component_count=plan.component_count,
        file_count=file_count,
        manifest_written=plan.include_meta,
    )


def build_inline_hydration_plan(
    targets: list[str],
    *,
    context_dir: Path,
    access: str = "writable",
    copy: bool = False,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> HydratePlan:
    from ..utils import brace_expand

    cwd = os.getcwd()
    used_paths: set[str] = set()
    files_to_write: list[tuple[Path, str]] = []
    files_to_symlink: list[tuple[Path, Path]] = []
    file_timestamps: dict[Path, tuple[float, float]] = {}

    expanded: list[str] = []
    for t in targets:
        if "{" in t and "}" in t:
            expanded.extend(brace_expand(t))
        else:
            expanded.append(t)

    for target in expanded:
        items = _resolve_spec_items(
            target,
            cwd,
            component_name="inline",
            alias_hint=None,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        for item in items:
            subpath = _split_subpath(item.context_subpath)
            rel_path = _dedupe_path(subpath, used_paths)
            _ensure_relative(rel_path)

            can_symlink = (
                not copy and item.source_type == "local" and item.source_full_path
            )
            if can_symlink:
                files_to_symlink.append(
                    (context_dir / rel_path, Path(item.source_full_path))
                )
            else:
                files_to_write.append((context_dir / rel_path, item.content))
            file_ts = _item_file_ts(item)
            if file_ts:
                file_timestamps[context_dir / rel_path] = file_ts
            dir_ts = _item_dir_ts(item)
            if dir_ts:
                file_timestamps.setdefault((context_dir / rel_path).parent, dir_ts)

    return HydratePlan(
        context_dir=context_dir,
        files_to_write=files_to_write,
        files_to_symlink=files_to_symlink,
        used_paths=used_paths,
        component_count=0,
        include_meta=False,
        access=access,
        file_timestamps=file_timestamps,
    )


def plan_matches_existing(plan: HydratePlan) -> bool:
    context_dir = plan.context_dir
    if not context_dir.exists() or not context_dir.is_dir():
        return False

    expected_hashes: dict[str, tuple[int, str]] = {}
    expected_symlinks: dict[str, Path] = {}
    expected_dirs: set[str] = set()

    for path, content in plan.files_to_write:
        rel = path.relative_to(context_dir).as_posix()
        data = content.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        expected_hashes[rel] = (len(data), digest)
        _collect_parent_dirs(rel, expected_dirs)

    for dest, source in plan.files_to_symlink:
        rel = dest.relative_to(context_dir).as_posix()
        expected_symlinks[rel] = source.resolve()
        _collect_parent_dirs(rel, expected_dirs)

    all_expected_files = set(expected_hashes.keys()) | set(expected_symlinks.keys())

    seen_files: set[str] = set()
    for root, dirs, files in os.walk(context_dir, followlinks=False):
        rel_root = os.path.relpath(root, context_dir)
        if rel_root != "." and rel_root not in expected_dirs:
            return False
        for name in files:
            file_path = Path(root) / name
            rel_file = file_path.relative_to(context_dir).as_posix()

            if rel_file in expected_symlinks:
                if not file_path.is_symlink():
                    return False
                try:
                    target = file_path.resolve()
                except OSError:
                    return False
                if target != expected_symlinks[rel_file]:
                    return False
                seen_files.add(rel_file)
            elif rel_file in expected_hashes:
                if file_path.is_symlink():
                    return False
                size, digest = expected_hashes[rel_file]
                try:
                    if file_path.stat().st_size != size:
                        return False
                except OSError:
                    return False
                if _hash_file(file_path) != digest:
                    return False
                seen_files.add(rel_file)
            else:
                return False
        for name in dirs:
            dir_path = Path(root) / name
            rel_dir = dir_path.relative_to(context_dir).as_posix()
            if rel_dir not in expected_dirs:
                return False

    if seen_files != all_expected_files:
        return False

    if plan.access == "read-only":
        writable_files = set(expected_hashes.keys())
        if not _all_read_only(context_dir, expected_dirs, writable_files):
            return False

    return True


def clear_context_dir(path: Path) -> None:
    _clear_context_dir(path)


_RANGE_RE = re.compile(r"^\s*L?(\d+)\s*(?:-|:)\s*L?(\d+)\s*$")


def hydrate_manifest(
    manifest_path: str,
    *,
    overrides: HydrateOverrides,
    cwd: str,
) -> HydrateResult:
    plan = build_hydration_plan(
        manifest_path,
        overrides=overrides,
        cwd=cwd,
    )
    return apply_hydration_plan(plan)


def hydrate_manifest_data(
    data: dict[str, Any],
    manifest_cwd: str,
    *,
    manifest_path: str | None = None,
    overrides: HydrateOverrides,
    cwd: str,
) -> HydrateResult:
    plan = build_hydration_plan_data(
        data,
        manifest_cwd,
        manifest_path=manifest_path,
        overrides=overrides,
        cwd=cwd,
    )
    return apply_hydration_plan(plan)


def build_hydration_plan(
    manifest_path: str,
    *,
    overrides: HydrateOverrides,
    cwd: str,
) -> HydratePlan:
    import yaml

    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping with 'config' and 'components'")
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    return build_hydration_plan_data(
        data,
        manifest_dir,
        manifest_path=manifest_path,
        overrides=overrides,
        cwd=cwd,
    )


def build_hydration_plan_data(
    data: dict[str, Any],
    manifest_cwd: str,
    *,
    manifest_path: str | None = None,
    overrides: HydrateOverrides,
    cwd: str,
) -> HydratePlan:
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping with 'config' and 'components'")

    cfg = data.get("config") or {}
    if not isinstance(cfg, dict):
        raise ValueError("'config' must be a mapping")

    components = data.get("components")
    if not isinstance(components, list):
        raise ValueError("'components' must be a list")
    components = normalize_components(components)

    arena_overrides = _resolve_arena_config(cfg)
    atproto_overrides = _resolve_atproto_config(cfg)
    discord_overrides = _resolve_discord_config(cfg)

    base_dir = _resolve_base_dir(cfg, manifest_cwd, manifest_path)
    context_cfg = _resolve_context_config(cfg, overrides, cwd)
    context_dir = context_cfg["dir"]
    has_local_sources = _manifest_has_local_sources(components)
    has_arena_channels = _manifest_has_arena_channels(components)
    has_atproto_sources = _manifest_has_atproto_sources(components)
    has_discord_channels = _manifest_has_discord_channels(components)
    use_external_root = context_cfg["path_strategy"] == "on-disk" and has_local_sources

    used_paths: set[str] = set()
    identity_paths: dict[tuple[Any, ...], Path] = {}
    files_to_write: list[tuple[Path, str]] = []
    files_to_symlink: list[tuple[Path, Path]] = []
    file_timestamps: dict[Path, tuple[float, float]] = {}
    index_components: dict[str, list[dict[str, Any]]] = {}
    normalized_components: list[dict[str, Any]] = []
    arena_pending: dict[tuple[Any, ...], list[_ArenaPendingWrite]] = {}
    arena_seen_counter = 0
    resolved_spec_cache: dict[
        tuple[
            str,
            bool,
            bool,
            str | None,
            tuple[Any, ...] | None,
            tuple[Any, ...] | None,
            tuple[Any, ...] | None,
        ],
        list[ResolvedItem],
    ] = {}

    global_strip_prefix: Path | None = None
    if context_cfg["path_strategy"] == "by-component":
        global_strip_prefix = _find_global_subpath_prefix(components, base_dir)

    if context_cfg["agents_text"] is not None:
        _queue_agent_files(
            files_to_write,
            context_cfg["dir"],
            context_cfg["agents_files"] or ["AGENTS.md"],
            context_cfg["agents_text"],
        )

    for comp in components:
        comp_name = comp["name"]
        comp_files = comp.get("files")
        comp_repos = comp.get("repos")
        comp_text = comp.get("text")
        comp_prefix = comp.get("prefix")
        comp_suffix = comp.get("suffix")
        comp_gitignore_raw = comp.get("gitignore")
        if comp_gitignore_raw is None:
            comp_gitignore = context_cfg["gitignore"]
        elif isinstance(comp_gitignore_raw, bool):
            comp_gitignore = comp_gitignore_raw
        else:
            raise ValueError(f"Component '{comp_name}' gitignore must be a boolean")
        component_root = _build_component_root(
            comp_name,
            comp.get(GROUP_PATH_KEY),
            comp.get(GROUP_BASE_KEY),
            context_cfg["path_strategy"],
        )
        component_arena_overrides = _parse_arena_config_mapping(
            comp.get("arena"), prefix=f"component '{comp_name}'.arena"
        )
        component_atproto_overrides = _parse_atproto_config_mapping(
            comp.get("atproto"),
            prefix=f"component '{comp_name}'.atproto",
        )
        component_discord_overrides = parse_discord_config_mapping(
            comp.get("discord"), prefix=f"component '{comp_name}'.discord"
        )

        if (
            not comp_files
            and not comp_repos
            and comp_text is None
            and comp_prefix is None
            and comp_suffix is None
        ):
            raise ValueError(f"Component '{comp_name}' has no content")

        normalized_comp = _build_normalized_component(comp, comp_name)
        normalized_files: list[Any] = []
        for note_name, note_value in (
            ("text-001.md", comp_text),
            ("prefix.md", comp_prefix),
            ("suffix.md", comp_suffix),
        ):
            if note_value is None:
                continue
            if not isinstance(note_value, str):
                raise ValueError(
                    f"Component '{comp_name}' note '{note_name}' must be a string"
                )
            note_root = component_root or Path(comp_name)
            note_path = context_dir / note_root / "notes" / note_name
            files_to_write.append((note_path, note_value))

        if comp_files is not None and not isinstance(comp_files, list):
            raise ValueError(f"Component '{comp_name}' files must be a list")
        if comp_repos is not None and not isinstance(comp_repos, list):
            raise ValueError(f"Component '{comp_name}' repos must be a list")

        all_specs: list[tuple[Any, bool, str | None]] = []
        if comp_files:
            all_specs.extend((spec, False, None) for spec in comp_files)
        if comp_repos:
            for spec, root in _expand_repo_specs(comp_repos):
                all_specs.append((spec, True, root))
        if all_specs:
            resolved_items: list[
                tuple[
                    ResolvedItem,
                    str,
                    str | None,
                    list[tuple[int, int]] | None,
                    list[str] | None,
                    list[tuple[int, int]] | None,
                    str | None,
                    dict[str, Any],
                    str | None,
                ]
            ] = []
            spec_jobs = get_payload_spec_jobs()
            prepared_specs: list[dict[str, Any]] = []
            pending_by_key: dict[
                tuple[
                    str,
                    bool,
                    bool,
                    str | None,
                    tuple[Any, ...] | None,
                    tuple[Any, ...] | None,
                    tuple[Any, ...] | None,
                ],
                tuple[
                    str,
                    Any | None,
                    bool,
                    dict | None,
                    dict[str, Any] | None,
                    dict | None,
                ],
            ] = {}

            for spec_index, (file_spec, force_git, spec_root) in enumerate(
                all_specs, 1
            ):
                raw_spec, file_opts = coerce_file_spec(file_spec)
                spec_comment = _parse_comment(file_opts.pop("comment", None))
                range_value = file_opts.pop("range", None)
                symbols_value = file_opts.pop("symbols", None)
                range_spec = _parse_range_value(range_value)
                symbols_spec = _parse_symbols_value(symbols_value)
                raw_spec, path_symbols = split_spec_symbols(raw_spec)
                if path_symbols:
                    symbols_spec = _merge_symbols(symbols_spec, path_symbols)
                file_arena_overrides = _parse_arena_config_mapping(
                    file_opts.get("arena"),
                    prefix=f"component '{comp_name}' file[{spec_index}].arena",
                )
                effective_arena_overrides = _merge_arena_overrides(
                    arena_overrides,
                    component_arena_overrides,
                    file_arena_overrides,
                )
                file_atproto_overrides = _parse_atproto_config_mapping(
                    file_opts.get("atproto"),
                    prefix=f"component '{comp_name}' file[{spec_index}].atproto",
                )
                effective_atproto_overrides = _merge_atproto_overrides(
                    atproto_overrides,
                    component_atproto_overrides,
                    file_atproto_overrides,
                )
                file_discord_overrides = parse_discord_config_mapping(
                    file_opts.get("discord"),
                    prefix=f"component '{comp_name}' file[{spec_index}].discord",
                )
                effective_discord_overrides = _merge_discord_overrides(
                    discord_overrides,
                    component_discord_overrides,
                    file_discord_overrides,
                )
                arena_overrides_key = _arena_overrides_cache_key(
                    effective_arena_overrides
                )
                atproto_overrides_key = _atproto_overrides_cache_key(
                    effective_atproto_overrides
                )
                discord_overrides_key = _discord_overrides_cache_key(
                    effective_discord_overrides
                )

                alias_hint = file_opts.get("alias") or file_opts.get("filename")
                cache_alias = alias_hint if isinstance(alias_hint, str) else None
                spec_cache_key = (
                    raw_spec,
                    force_git,
                    comp_gitignore,
                    cache_alias,
                    arena_overrides_key,
                    atproto_overrides_key,
                    discord_overrides_key,
                )
                if (
                    spec_cache_key not in resolved_spec_cache
                    and spec_cache_key not in pending_by_key
                ):
                    pending_by_key[spec_cache_key] = (
                        raw_spec,
                        alias_hint,
                        force_git,
                        effective_arena_overrides,
                        effective_atproto_overrides,
                        effective_discord_overrides,
                    )

                prepared_specs.append(
                    {
                        "spec_cache_key": spec_cache_key,
                        "spec_comment": spec_comment,
                        "range_spec": range_spec,
                        "symbols_spec": symbols_spec,
                        "file_opts": file_opts,
                        "spec_root": spec_root,
                    }
                )

            tasks = [
                (
                    index,
                    (
                        lambda rs=raw_spec, ah=alias_hint, fg=force_git, eao=effective_arena_overrides, eatpo=effective_atproto_overrides, edo=effective_discord_overrides: (
                            _resolve_spec_items(
                                rs,
                                base_dir,
                                component_name=comp_name,
                                alias_hint=ah,
                                gitignore=comp_gitignore,
                                use_cache=context_cfg["use_cache"],
                                cache_ttl=context_cfg["cache_ttl"],
                                refresh_cache=context_cfg["refresh_cache"],
                                force_git=fg,
                                arena_overrides=eao,
                                atproto_overrides=eatpo,
                                discord_overrides=edo,
                            )
                        )
                    ),
                )
                for index, (
                    cache_key,
                    (
                        raw_spec,
                        alias_hint,
                        force_git,
                        effective_arena_overrides,
                        effective_atproto_overrides,
                        effective_discord_overrides,
                    ),
                ) in enumerate(pending_by_key.items())
            ]
            if tasks and any(
                is_http_url(raw_spec) for raw_spec, *_ in pending_by_key.values()
            ):
                from ..references.arena import warmup_arena_network_stack

                warmup_arena_network_stack()
            if tasks and any(
                is_atproto_url(raw_spec) for raw_spec, *_ in pending_by_key.values()
            ):
                from ..references.atproto import warmup_atproto_network_stack

                warmup_atproto_network_stack()
            pending_keys = list(pending_by_key.keys())
            for task_index, result_items in run_indexed_tasks_fail_fast(
                tasks, max_workers=spec_jobs
            ):
                resolved_spec_cache[pending_keys[task_index]] = result_items

            for prepared in prepared_specs:
                spec_cache_key = prepared["spec_cache_key"]
                spec_comment = prepared["spec_comment"]
                range_spec = prepared["range_spec"]
                symbols_spec = prepared["symbols_spec"]
                file_opts = prepared["file_opts"]
                spec_root = prepared["spec_root"]

                cached_items = resolved_spec_cache[spec_cache_key]
                for item in cached_items:
                    ranges = range_spec[:] if range_spec else None
                    symbols = symbols_spec[:] if symbols_spec else None

                    if symbols:
                        ranges, symbols, should_skip = resolve_symbol_ranges(
                            item.context_subpath,
                            symbols,
                            text=item.content,
                            ranges=ranges,
                            warn_label=item.context_subpath,
                            append_to_ranges=True,
                            keep_missing=False,
                            skip_on_missing=True,
                            warn_on_partial=False,
                        )
                        if should_skip:
                            continue

                    content = (
                        extract_ranges(item.content, ranges) if ranges else item.content
                    )
                    suffix = _build_suffix(ranges, symbols)
                    resolved_items.append(
                        (
                            item,
                            content,
                            suffix,
                            ranges,
                            symbols,
                            range_spec,
                            spec_comment,
                            file_opts,
                            spec_root,
                        )
                    )

            comp_strip = comp.get("strip-paths", False)

            if context_cfg["path_strategy"] == "by-component" and comp_strip:
                all_subpaths = [
                    item.context_subpath
                    for item, *_ in resolved_items
                    if item.source_type in ("local", "git")
                ]
                effective_strip_prefix = _find_common_subpath_prefix(all_subpaths)
                skip_external_root = True
            else:
                effective_strip_prefix = global_strip_prefix
                skip_external_root = False

            for (
                item,
                content,
                suffix,
                ranges,
                symbols,
                range_spec,
                spec_comment,
                file_opts,
                spec_root,
            ) in resolved_items:
                rel_path, should_write = _resolve_context_path(
                    comp_name,
                    item,
                    suffix,
                    used_paths,
                    identity_paths,
                    context_cfg["path_strategy"],
                    ranges,
                    symbols,
                    component_root=component_root,
                    use_external_root=use_external_root,
                    strip_prefix=effective_strip_prefix,
                    skip_external_root=skip_external_root,
                    external_root_prefix=spec_root,
                )
                arena_identity = _arena_pending_key(item, ranges, symbols)
                if arena_identity:
                    arena_pending.setdefault(arena_identity, []).append(
                        _ArenaPendingWrite(
                            rel_path=rel_path,
                            content=content,
                            should_write=should_write,
                            item=item,
                            encounter_index=arena_seen_counter,
                        )
                    )
                    arena_seen_counter += 1
                    index_components.setdefault(comp_name, []).append(
                        _build_index_entry(
                            rel_path,
                            item,
                            ranges,
                            symbols,
                            content,
                        )
                    )
                    normalized_files.append(
                        _build_manifest_file_entry(
                            item,
                            range_spec,
                            symbols,
                            spec_comment,
                            file_opts,
                        )
                    )
                    continue
                if should_write:
                    can_symlink = (
                        not context_cfg["copy"]
                        and item.source_type == "local"
                        and item.source_full_path
                        and not ranges
                        and not symbols
                    )
                    if can_symlink:
                        files_to_symlink.append(
                            (context_dir / rel_path, Path(item.source_full_path))
                        )
                    else:
                        files_to_write.append((context_dir / rel_path, content))
                    file_ts = _item_file_ts(item)
                    if file_ts:
                        file_timestamps[context_dir / rel_path] = file_ts
                    dir_ts = _item_dir_ts(item)
                    if dir_ts:
                        file_timestamps.setdefault(
                            (context_dir / rel_path).parent, dir_ts
                        )
                index_components.setdefault(comp_name, []).append(
                    _build_index_entry(
                        rel_path,
                        item,
                        ranges,
                        symbols,
                        content,
                    )
                )
                normalized_files.append(
                    _build_manifest_file_entry(
                        item,
                        range_spec,
                        symbols,
                        spec_comment,
                        file_opts,
                    )
                )

        if normalized_files:
            normalized_comp["files"] = _dedupe_manifest_entries(normalized_files)
        normalized_components.append(normalized_comp)

    if context_cfg["include_meta"]:
        normalized_manifest = {
            "config": _build_normalized_config(
                cfg,
                context_cfg,
                base_dir,
                include_root=has_local_sources,
                include_arena_max_depth=has_arena_channels,
                arena_overrides=arena_overrides,
                include_atproto_defaults=has_atproto_sources,
                atproto_overrides=atproto_overrides,
                include_discord_defaults=has_discord_channels,
                discord_overrides=discord_overrides,
            ),
            "components": normalized_components,
        }
        manifest_text = _dump_manifest(normalized_manifest)
        files_to_write.append((context_dir / "manifest.yaml", manifest_text))

    if context_cfg["include_meta"]:
        index_data = {"version": 1, "components": index_components}
        index_text = _dump_index(index_data)
        files_to_write.append((context_dir / "index.json", index_text))

    _materialize_arena_pending(
        arena_pending,
        context_dir,
        files_to_write,
        files_to_symlink,
        file_timestamps,
    )

    return HydratePlan(
        context_dir=context_dir,
        files_to_write=files_to_write,
        files_to_symlink=files_to_symlink,
        used_paths=used_paths,
        component_count=len(components),
        include_meta=context_cfg["include_meta"],
        access=context_cfg["access"],
        file_timestamps=file_timestamps,
    )


def _resolve_base_dir(
    cfg: dict[str, Any], manifest_cwd: str, manifest_path: str | None
) -> str:
    if "root" in cfg:
        raw_root = cfg.get("root") or "~"
        base_dir = os.path.expanduser(raw_root)
    else:
        base_dir = manifest_cwd if manifest_path or manifest_cwd else os.getcwd()
    return os.path.abspath(base_dir)


def _resolve_context_config(
    cfg: dict[str, Any], overrides: HydrateOverrides, cwd: str
) -> dict[str, Any]:
    context_cfg = cfg.get("context") or {}
    if not isinstance(context_cfg, dict):
        raise ValueError("'config.context' must be a mapping")

    raw_access = overrides.access or context_cfg.get("access") or "writable"
    if not isinstance(raw_access, str):
        raise ValueError("context access must be a string")
    access = raw_access.lower()
    if access not in {"read-only", "writable"}:
        raise ValueError("context access must be 'read-only' or 'writable'")

    raw_dir_value = overrides.context_dir or context_cfg.get("dir")
    if not raw_dir_value:
        manifest_name = cfg.get("name")
        if manifest_name:
            _validate_manifest_name(manifest_name)
            raw_dir_value = f".context/{manifest_name}"
        else:
            raw_dir_value = ".context"
    dir_value = str(raw_dir_value)
    dir_path = Path(os.path.expanduser(dir_value))
    if not dir_path.is_absolute():
        dir_path = Path(os.path.abspath(os.path.join(cwd, dir_path)))

    raw_path_strategy = (
        overrides.path_strategy or context_cfg.get("path-strategy") or "on-disk"
    )
    if not isinstance(raw_path_strategy, str):
        raise ValueError("path-strategy must be a string")
    path_strategy = raw_path_strategy.lower()
    if path_strategy not in {"on-disk", "by-component"}:
        raise ValueError("path-strategy must be 'on-disk' or 'by-component'")

    include_meta_value = context_cfg.get("include-meta", True)
    if not isinstance(include_meta_value, bool):
        raise ValueError("include-meta must be a boolean")
    include_meta = False if overrides.omit_meta else include_meta_value

    agents_text, agents_files = _resolve_agents(context_cfg, overrides)

    gitignore_value = context_cfg.get("gitignore", False)
    if not isinstance(gitignore_value, bool):
        raise ValueError("gitignore must be a boolean")

    if overrides.use_cache is not None:
        use_cache = overrides.use_cache
    else:
        use_cache = True

    if overrides.cache_ttl is not None:
        cache_ttl = overrides.cache_ttl
    else:
        raw_ttl = context_cfg.get("cache-ttl")
        if raw_ttl is not None:
            from ..cache import parse_duration

            if isinstance(raw_ttl, str):
                cache_ttl = parse_duration(raw_ttl)
            elif isinstance(raw_ttl, (int, float)):
                cache_ttl = timedelta(days=raw_ttl)
            else:
                raise ValueError("cache-ttl must be a string or number")
        else:
            cache_ttl = None

    refresh_cache = overrides.refresh_cache

    return {
        "dir": dir_path,
        "dir_value": dir_value,
        "access": access,
        "include_meta": bool(include_meta),
        "path_strategy": path_strategy,
        "agents_text": agents_text,
        "agents_files": agents_files,
        "gitignore": gitignore_value,
        "copy": overrides.copy,
        "use_cache": use_cache,
        "cache_ttl": cache_ttl,
        "refresh_cache": refresh_cache,
    }


def _resolve_agents(
    context_cfg: dict[str, Any], overrides: HydrateOverrides
) -> tuple[str | None, list[str] | None]:
    if overrides.agents_prompt is not None:
        files = list(overrides.agents_filenames) or ["AGENTS.md"]
        return overrides.agents_prompt, files

    agents_cfg = context_cfg.get("agents")
    if not isinstance(agents_cfg, dict):
        return None, None
    text = agents_cfg.get("text")
    if text is None:
        return None, None
    if not isinstance(text, str):
        raise ValueError("context.agents.text must be a string")
    files = agents_cfg.get("files") or ["AGENTS.md"]
    if isinstance(files, str):
        files = [files]
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        raise ValueError("context.agents.files must be a list of strings")
    if not files:
        files = ["AGENTS.md"]
    return text, files


def _parse_arena_config_mapping(raw: Any, *, prefix: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix} must be a mapping")

    from ..references.arena import VALID_SORT_ORDERS

    result: dict[str, Any] = {}

    recurse_depth = raw.get("recurse-depth")
    max_depth_alias = raw.get("max-depth")
    if recurse_depth is not None and max_depth_alias is not None:
        if recurse_depth != max_depth_alias:
            raise ValueError(
                f"{prefix}.recurse-depth and {prefix}.max-depth cannot differ"
            )
    if recurse_depth is None:
        recurse_depth = max_depth_alias
    if recurse_depth is not None:
        if (
            not isinstance(recurse_depth, int)
            or isinstance(recurse_depth, bool)
            or recurse_depth < 0
        ):
            raise ValueError(
                f"{prefix}.recurse-depth (alias: max-depth) must be a non-negative integer"
            )
        result["max_depth"] = recurse_depth

    block_sort = raw.get("block-sort")
    sort_alias = raw.get("sort")
    if block_sort is not None and sort_alias is not None:
        left = str(block_sort).lower().strip()
        right = str(sort_alias).lower().strip()
        if left != right:
            raise ValueError(f"{prefix}.block-sort and {prefix}.sort cannot differ")
    if block_sort is None:
        block_sort = sort_alias
    if block_sort is not None:
        if (
            not isinstance(block_sort, str)
            or block_sort.lower().strip() not in VALID_SORT_ORDERS
        ):
            raise ValueError(
                f"{prefix}.block-sort (alias: sort) must be one of: "
                + ", ".join(sorted(VALID_SORT_ORDERS))
            )
        result["sort_order"] = block_sort.lower().strip()

    max_blocks_per_channel = raw.get("max-blocks-per-channel")
    if max_blocks_per_channel is not None:
        if (
            not isinstance(max_blocks_per_channel, int)
            or isinstance(max_blocks_per_channel, bool)
            or max_blocks_per_channel <= 0
        ):
            raise ValueError(
                f"{prefix}.max-blocks-per-channel must be a positive integer"
            )
        result["max_blocks_per_channel"] = max_blocks_per_channel

    for config_key, result_key in (
        ("connected-after", "connected_after"),
        ("connected-before", "connected_before"),
        ("created-after", "created_after"),
        ("created-before", "created_before"),
    ):
        if config_key not in raw:
            continue
        value = raw[config_key]
        parsed = _parse_iso_timestamp(value)
        if parsed is None:
            raise ValueError(
                f"{prefix}.{config_key} is not a valid timestamp "
                "(ISO, epoch, or relative duration)"
            )
        result[result_key] = parsed

    allowed_keys = {
        "recurse-depth",
        "max-depth",
        "block-sort",
        "sort",
        "max-blocks-per-channel",
        "connected-after",
        "connected-before",
        "created-after",
        "created-before",
        "recurse-users",
        "block",
    }
    unknown_keys = sorted(str(key) for key in raw.keys() if key not in allowed_keys)
    if unknown_keys:
        raise ValueError(f"{prefix} has invalid keys: {', '.join(unknown_keys)}")

    has_connected_window = (
        result.get("connected_after") is not None
        or result.get("connected_before") is not None
    )
    has_created_window = (
        result.get("created_after") is not None
        or result.get("created_before") is not None
    )
    if has_connected_window and has_created_window:
        raise ValueError(f"{prefix} cannot mix connected-* and created-* windows")

    connected_after = result.get("connected_after")
    connected_before = result.get("connected_before")
    created_after = result.get("created_after")
    created_before = result.get("created_before")
    if (
        isinstance(connected_after, datetime)
        and isinstance(connected_before, datetime)
        and connected_after > connected_before
    ):
        raise ValueError(
            f"{prefix}.connected-after must be <= {prefix}.connected-before"
        )
    if (
        isinstance(created_after, datetime)
        and isinstance(created_before, datetime)
        and created_after > created_before
    ):
        raise ValueError(f"{prefix}.created-after must be <= {prefix}.created-before")

    block_cfg = raw.get("block")
    if block_cfg is not None and not isinstance(block_cfg, dict):
        raise ValueError(f"{prefix}.block must be a mapping")
    block_cfg = block_cfg or {}
    allowed_block_keys = {
        "media-desc",
        "link-image-desc",
        "pdf-content",
        "comments",
        "description",
    }
    unknown_block_keys = sorted(
        str(key) for key in block_cfg.keys() if key not in allowed_block_keys
    )
    if unknown_block_keys:
        raise ValueError(
            f"{prefix}.block has invalid keys: {', '.join(unknown_block_keys)}"
        )

    if "description" in block_cfg:
        val = block_cfg["description"]
        if not isinstance(val, bool):
            raise ValueError(f"{prefix}.block.description must be a boolean")
        result["include_descriptions"] = val

    if "comments" in block_cfg:
        val = block_cfg["comments"]
        if not isinstance(val, bool):
            raise ValueError(f"{prefix}.block.comments must be a boolean")
        result["include_comments"] = val

    if "link-image-desc" in block_cfg:
        val = block_cfg["link-image-desc"]
        if not isinstance(val, bool):
            raise ValueError(f"{prefix}.block.link-image-desc must be a boolean")
        result["include_link_image_descriptions"] = val

    if "pdf-content" in block_cfg:
        val = block_cfg["pdf-content"]
        if not isinstance(val, bool):
            raise ValueError(f"{prefix}.block.pdf-content must be a boolean")
        result["include_pdf_content"] = val

    if "media-desc" in block_cfg:
        val = block_cfg["media-desc"]
        if not isinstance(val, bool):
            raise ValueError(f"{prefix}.block.media-desc must be a boolean")
        result["include_media_descriptions"] = val

    if "recurse-users" in raw:
        val = raw["recurse-users"]
        if isinstance(val, str):
            val_lower = val.lower().strip()
            if val_lower == "all":
                result["recurse_users"] = None
            elif val_lower in ("self", "author", "owner"):
                result["recurse_users"] = {val_lower}
            else:
                result["recurse_users"] = {val_lower}
        elif isinstance(val, list):
            result["recurse_users"] = {str(v).strip().lower() for v in val if v}
        else:
            raise ValueError(
                f"{prefix}.recurse-users must be a string or list of strings"
            )

    return result if result else None


def _resolve_arena_config(cfg: dict[str, Any]) -> dict | None:
    return _parse_arena_config_mapping(cfg.get("arena"), prefix="config.arena")


def _merge_arena_overrides(*overrides: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for item in overrides:
        if item:
            merged.update(item)
    return merged or None


def _arena_overrides_cache_key(
    overrides: dict[str, Any] | None,
) -> tuple[Any, ...] | None:
    if not overrides:
        return None

    normalized: list[tuple[str, Any]] = []
    for key, value in sorted(overrides.items()):
        if isinstance(value, set):
            normalized.append((key, tuple(sorted(value))))
        else:
            normalized.append((key, value))
    return tuple(normalized)


def _parse_iso_timestamp(value: Any) -> datetime | None:
    return parse_timestamp_or_duration_value(value)


def _parse_atproto_config_mapping(raw: Any, *, prefix: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix} must be a mapping")

    allowed_keys = {
        "max-items",
        "thread-depth",
        "post-ancestors",
        "quote-depth",
        "max-replies",
        "reply-quote-depth",
        "replies",
        "reposts",
        "likes",
        "created-after",
        "created-before",
        "include-media-descriptions",
        "include-embed-media-descriptions",
        "media-mode",
    }
    unknown_keys = sorted(str(key) for key in raw.keys() if key not in allowed_keys)
    if unknown_keys:
        raise ValueError(f"{prefix} has invalid keys: {', '.join(unknown_keys)}")

    result: dict[str, Any] = {}

    if "max-items" in raw:
        value = raw["max-items"]
        if isinstance(value, int) and not isinstance(value, bool):
            if value <= 0:
                raise ValueError(f"{prefix}.max-items must be >= 1 or 'all'")
            result["max_items"] = value
        elif isinstance(value, str) and value.strip().lower() == "all":
            result["max_items"] = "all"
        else:
            raise ValueError(f"{prefix}.max-items must be a positive integer or 'all'")

    if "thread-depth" in raw:
        value = raw["thread-depth"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{prefix}.thread-depth must be >= 0")
        result["thread_depth"] = value

    if "post-ancestors" in raw:
        value = raw["post-ancestors"]
        if isinstance(value, int) and not isinstance(value, bool):
            if value < 0:
                raise ValueError(f"{prefix}.post-ancestors must be >= 0 or 'all'")
            result["post_ancestors"] = value
        elif isinstance(value, str) and value.strip().lower() == "all":
            result["post_ancestors"] = "all"
        else:
            raise ValueError(
                f"{prefix}.post-ancestors must be a non-negative integer or 'all'"
            )

    if "quote-depth" in raw:
        value = raw["quote-depth"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{prefix}.quote-depth must be >= 0")
        result["quote_depth"] = value

    if "max-replies" in raw:
        value = raw["max-replies"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{prefix}.max-replies must be >= 0")
        result["max_replies"] = value

    if "reply-quote-depth" in raw:
        value = raw["reply-quote-depth"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{prefix}.reply-quote-depth must be >= 0")
        result["reply_quote_depth"] = value

    for config_key, result_key in (
        ("replies", "replies_filter"),
        ("reposts", "reposts_filter"),
        ("likes", "likes_filter"),
    ):
        if config_key not in raw:
            continue
        value = raw[config_key]
        if not isinstance(value, str) or value.strip().lower() not in {
            "include",
            "exclude",
            "only",
        }:
            raise ValueError(
                f"{prefix}.{config_key} must be one of: include, exclude, only"
            )
        result[result_key] = value.strip().lower()
    only_modes = [
        mode
        for mode in (
            result.get("replies_filter"),
            result.get("reposts_filter"),
            result.get("likes_filter"),
        )
        if mode == "only"
    ]
    if len(only_modes) > 1:
        raise ValueError(
            f"{prefix} can set at most one of replies/reposts/likes to 'only'"
        )

    for config_key, result_key in (
        ("created-after", "created_after"),
        ("created-before", "created_before"),
    ):
        if config_key not in raw:
            continue
        value = raw[config_key]
        parsed = _parse_iso_timestamp(value)
        if parsed is None:
            raise ValueError(
                f"{prefix}.{config_key} is not a valid timestamp "
                "(ISO, epoch, or relative duration)"
            )
        result[result_key] = parsed

    created_after = result.get("created_after")
    created_before = result.get("created_before")
    if (
        isinstance(created_after, datetime)
        and isinstance(created_before, datetime)
        and created_after > created_before
    ):
        raise ValueError(f"{prefix}.created-after must be <= {prefix}.created-before")

    if "include-media-descriptions" in raw:
        value = raw["include-media-descriptions"]
        if not isinstance(value, bool):
            raise ValueError(f"{prefix}.include-media-descriptions must be a boolean")
        result["include_media_descriptions"] = value

    if "include-embed-media-descriptions" in raw:
        value = raw["include-embed-media-descriptions"]
        if not isinstance(value, bool):
            raise ValueError(
                f"{prefix}.include-embed-media-descriptions must be a boolean"
            )
        result["include_embed_media_descriptions"] = value

    if "media-mode" in raw:
        value = raw["media-mode"]
        if not isinstance(value, str) or value.strip().lower() not in {
            "describe",
            "transcribe",
        }:
            raise ValueError(
                f"{prefix}.media-mode must be one of: describe, transcribe"
            )
        result["media_mode"] = value.strip().lower()

    return result or None


def _resolve_atproto_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    return _parse_atproto_config_mapping(cfg.get("atproto"), prefix="config.atproto")


def _merge_atproto_overrides(
    *overrides: dict[str, Any] | None,
) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for item in overrides:
        if item:
            merged.update(item)
    return merged or None


def _atproto_overrides_cache_key(
    overrides: dict[str, Any] | None,
) -> tuple[Any, ...] | None:
    if not overrides:
        return None
    return tuple((key, value) for key, value in sorted(overrides.items()))


def _resolve_discord_config(cfg: dict[str, Any]) -> dict | None:
    return parse_discord_config_mapping(cfg.get("discord"), prefix="config.discord")


def _merge_discord_overrides(
    *overrides: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return merge_discord_overrides(*overrides)


def _discord_overrides_cache_key(
    overrides: dict[str, Any] | None,
) -> tuple[Any, ...] | None:
    return discord_overrides_cache_key(overrides)


def _discord_settings_cache_key(settings: Any) -> tuple[Any, ...]:
    return discord_settings_cache_key(settings)


def _atproto_settings_cache_key(settings: Any) -> tuple[Any, ...]:
    return atproto_settings_cache_key(settings)


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


def _queue_agent_files(
    files: list[tuple[Path, str]],
    context_dir: Path,
    filenames: list[str],
    content: str,
) -> None:
    for filename in filenames:
        _validate_agent_filename(filename)
        files.append((context_dir / filename, content))


def _validate_agent_filename(name: str) -> None:
    if not name or name in {".", ".."}:
        raise ValueError("Agent filename must be a non-empty name")
    if "/" in name or "\\" in name:
        raise ValueError("Agent filename must not contain path separators")


def _validate_manifest_name(name: Any) -> None:
    if not isinstance(name, str):
        raise ValueError("config.name must be a string")
    if not name or name in {".", ".."}:
        raise ValueError("config.name must be a non-empty name")
    if "/" in name or "\\" in name:
        raise ValueError("config.name must not contain path separators")


def _assign_component_names(components: list[dict[str, Any]]) -> None:
    used = set()
    for comp in components:
        if not isinstance(comp, dict):
            raise ValueError("Components must be mappings")
        name = comp.get("name")
        if name is None:
            continue
        if not isinstance(name, str):
            raise ValueError("Component name must be a string")
        _validate_component_name(name)
        if name in used:
            raise ValueError(f"Duplicate component name: {name}")
        used.add(name)

    counter = 1
    for comp in components:
        if comp.get("name"):
            continue
        while True:
            candidate = f"component-{counter:03d}"
            counter += 1
            if candidate not in used:
                break
        _validate_component_name(candidate)
        comp["name"] = candidate
        used.add(candidate)


def _validate_component_name(name: str) -> None:
    parts = Path(name).parts
    if len(parts) != 1 or name in {".", ".."}:
        raise ValueError(f"Invalid component name: {name}")
    if "/" in name or "\\" in name:
        raise ValueError(f"Invalid component name: {name}")


def _build_normalized_component(comp: dict[str, Any], name: str) -> dict[str, Any]:
    normalized = {
        k: v
        for k, v in comp.items()
        if k not in {"files", "repos", "name", GROUP_PATH_KEY, GROUP_BASE_KEY}
        and v is not None
    }
    normalized["name"] = name
    return normalized


def _expand_repo_specs(
    repos: list[Any],
) -> list[tuple[Any, str | None]]:
    result: list[tuple[Any, str | None]] = []
    for repo_spec in repos:
        if isinstance(repo_spec, dict) and "items" in repo_spec:
            root = repo_spec.get("root")
            for item in repo_spec["items"]:
                result.append((item, root))
        else:
            result.append((repo_spec, None))
    return result


def _find_common_subpath_prefix(subpaths: list[str]) -> Path | None:
    if not subpaths:
        return None

    parents = [Path(sp).parent for sp in subpaths]
    if any(p == Path(".") or not p.parts for p in parents):
        return None

    if len(parents) == 1:
        return parents[0] if parents[0].parts else None

    first = parents[0].parts
    common_parts: list[str] = []
    for i, part in enumerate(first):
        if all(len(p.parts) > i and p.parts[i] == part for p in parents):
            common_parts.append(part)
        else:
            break

    if not common_parts:
        return None

    return Path(*common_parts)


def _find_global_subpath_prefix(
    components: list[dict[str, Any]], base_dir: str
) -> Path | None:
    all_subpaths: list[str] = []
    for comp in components:
        comp_name = comp["name"]
        comp_files = comp.get("files") or []
        comp_repos = comp.get("repos") or []
        all_specs: list[tuple[Any, bool, str | None]] = [
            *((s, False, None) for s in comp_files),
            *((s, True, root) for s, root in _expand_repo_specs(comp_repos)),
        ]
        if not all_specs:
            continue
        for file_spec, force_git, _root in all_specs:
            raw_spec, file_opts = coerce_file_spec(file_spec)
            raw_spec, _ = split_spec_symbols(raw_spec)
            if is_http_url(raw_spec) or is_atproto_url(os.path.expanduser(raw_spec)):
                continue
            try:
                alias_hint = file_opts.get("alias") or file_opts.get("filename")
                for item in _resolve_spec_items(
                    raw_spec,
                    base_dir,
                    component_name=comp_name,
                    alias_hint=alias_hint,
                    force_git=force_git,
                ):
                    all_subpaths.append(item.context_subpath)
            except (FileNotFoundError, ValueError):
                pass
    return _find_common_subpath_prefix(all_subpaths)


def _manifest_has_local_sources(components: list[dict[str, Any]]) -> bool:
    for comp in components:
        files = comp.get("files")
        if not files or not isinstance(files, list):
            continue
        for file_spec in files:
            raw_spec, _ = coerce_file_spec(file_spec)
            if not _is_external_spec(raw_spec):
                return True
    return False


def _manifest_has_arena_channels(components: list[dict[str, Any]]) -> bool:
    for comp in components:
        files = comp.get("files")
        if not files or not isinstance(files, list):
            continue
        for file_spec in files:
            raw_spec, _ = coerce_file_spec(file_spec)
            spec = os.path.expanduser(raw_spec)
            if not is_http_url(spec):
                continue
            opts = parse_target_spec(spec)
            target = opts.get("target", spec)
            if is_arena_channel_url(target):
                return True
    return False


def _manifest_has_atproto_sources(components: list[dict[str, Any]]) -> bool:
    for comp in components:
        files = comp.get("files")
        if not files or not isinstance(files, list):
            continue
        for file_spec in files:
            raw_spec, _ = coerce_file_spec(file_spec)
            spec = os.path.expanduser(raw_spec)
            if is_atproto_url(spec):
                return True
            if not is_http_url(spec):
                continue
            opts = parse_target_spec(spec)
            target = opts.get("target", spec)
            if is_atproto_url(target):
                return True
    return False


def _manifest_has_discord_channels(components: list[dict[str, Any]]) -> bool:
    for comp in components:
        files = comp.get("files")
        if not files or not isinstance(files, list):
            continue
        for file_spec in files:
            raw_spec, _ = coerce_file_spec(file_spec)
            spec = os.path.expanduser(raw_spec)
            if not is_http_url(spec):
                continue
            opts = parse_target_spec(spec)
            target = opts.get("target", spec)
            if is_discord_url(target):
                return True
    return False


def _is_external_spec(raw_spec: str) -> bool:
    spec = os.path.expanduser(raw_spec)
    if is_atproto_url(spec):
        return True
    if is_http_url(spec):
        return True
    return parse_git_target(spec) is not None


def _parse_comment(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("File comment must be a string")
    return value


def _parse_range_value(value: Any) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("Range lines must be >= 1")
        return [(value, value)]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        ranges = [_parse_range_part(part) for part in parts]
        return ranges or None
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(v, int) for v in value):
            return [_parse_range_tuple(value[0], value[1])]
        ranges: list[tuple[int, int]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Range must be a pair of integers")
            ranges.append(_parse_range_tuple(item[0], item[1]))
        return ranges or None
    raise ValueError("Range must be a string or list")


def _parse_range_part(part: str) -> tuple[int, int]:
    match = _RANGE_RE.match(part)
    if match:
        return _parse_range_tuple(int(match.group(1)), int(match.group(2)))
    if part.isdigit():
        return _parse_range_tuple(int(part), int(part))
    raise ValueError(f"Invalid range value: {part}")


def _parse_range_tuple(start: int, end: int) -> tuple[int, int]:
    if start <= 0 or end <= 0:
        raise ValueError("Range lines must be >= 1")
    if start > end:
        raise ValueError("Range start must be <= end")
    return start, end


def _parse_symbols_value(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        symbols = [part.strip() for part in value.split(",") if part.strip()]
        return symbols or None
    if isinstance(value, (list, tuple)):
        symbols = [str(item).strip() for item in value if str(item).strip()]
        return symbols or None
    raise ValueError("Symbols must be a string or list")


def _merge_symbols(primary: list[str] | None, extra: list[str]) -> list[str] | None:
    merged = list(primary or [])
    for item in extra:
        if item not in merged:
            merged.append(item)
    return merged or None


def _resolve_spec_items(
    raw_spec: str,
    base_dir: str,
    *,
    component_name: str,
    alias_hint: Any | None,
    gitignore: bool = False,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    force_git: bool = False,
    arena_overrides: dict | None = None,
    atproto_overrides: dict[str, Any] | None = None,
    discord_overrides: dict | None = None,
) -> list[ResolvedItem]:
    spec = os.path.expanduser(raw_spec)

    if is_atproto_url(spec):
        alias = alias_hint if isinstance(alias_hint, str) else None
        return _resolve_atproto_items(
            spec,
            alias,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            atproto_overrides=atproto_overrides,
        )

    if is_http_url(spec):
        opts = parse_target_spec(spec)
        url = opts.get("target", spec)
        alias = alias_hint or opts.get("filename")

        if force_git:
            tgt = parse_git_target(url)
        else:
            tgt = parse_git_url_target(url)
        if tgt:
            return _resolve_git_items(
                tgt, component_name, alias=alias, gitignore=gitignore
            )

        gist_id = parse_gist_url(url)
        if gist_id:
            gist_files = fetch_gist_files(gist_id)
            if gist_files:
                if alias and len(gist_files) == 1:
                    return [
                        _resolve_gist_item(
                            gist_files[0][1],
                            alias,
                            use_cache=use_cache,
                            cache_ttl=cache_ttl,
                            refresh_cache=refresh_cache,
                        )
                    ]
                elif alias:
                    return [
                        _resolve_gist_item(
                            raw_url,
                            f"{alias}-{fname}",
                            use_cache=use_cache,
                            cache_ttl=cache_ttl,
                            refresh_cache=refresh_cache,
                        )
                        for fname, raw_url in gist_files
                    ]
                return [
                    _resolve_gist_item(
                        raw_url,
                        fname,
                        use_cache=use_cache,
                        cache_ttl=cache_ttl,
                        refresh_cache=refresh_cache,
                    )
                    for fname, raw_url in gist_files
                ]

        if is_youtube_url(url):
            return [
                _resolve_youtube_item(
                    url,
                    alias,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                )
            ]

        if is_arena_url(url):
            return _resolve_arena_items(
                url,
                alias,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
                arena_overrides=arena_overrides,
            )

        if is_discord_url(url):
            return _resolve_discord_items(
                url,
                alias,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
                discord_overrides=discord_overrides,
            )

        if is_atproto_url(url):
            return _resolve_atproto_items(
                url,
                alias,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
                atproto_overrides=atproto_overrides,
            )

        return [
            _resolve_http_item(
                url,
                alias,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
            )
        ]

    tgt = parse_git_target(spec)
    if tgt:
        return _resolve_git_items(
            tgt, component_name, alias=alias_hint, gitignore=gitignore
        )

    base = "" if os.path.isabs(spec) else base_dir
    paths = expand_git_paths(base, spec)
    resolved: list[ResolvedItem] = []

    ignore_cache: dict[str, list[str] | None] = {}

    for full in paths:
        if not os.path.exists(full):
            raise FileNotFoundError(
                f"Component '{component_name}' path not found: {full}"
            )

        ignore_patterns = None
        if gitignore:
            repo_root = get_repo_root(full)
            cache_key = repo_root or ""
            if cache_key not in ignore_cache:
                ignore_cache[cache_key] = read_gitignore_patterns(repo_root)
            ignore_patterns = ignore_cache[cache_key]

        refs = create_file_references(
            [full], ignore_patterns=ignore_patterns, format="raw", text_only=True
        )["refs"]
        for ref in refs:
            rel_path = _relative_path(ref.path, base_dir)
            resolved.append(
                ResolvedItem(
                    source_type="local",
                    source_ref=base_dir,
                    source_rev=None,
                    source_path=rel_path,
                    context_subpath=rel_path,
                    content=ref.file_content,
                    manifest_spec=rel_path,
                    source_full_path=ref.path,
                )
            )
    return resolved


def _resolve_git_items(
    tgt, component_name: str, *, alias: Any | None = None, gitignore: bool = False
) -> list[ResolvedItem]:
    repo_dir = ensure_repo(tgt)
    if tgt.path:
        paths = expand_git_paths(repo_dir, tgt.path)
    else:
        paths = [repo_dir]
    resolved: list[ResolvedItem] = []

    ignore_patterns = None
    if gitignore:
        ignore_patterns = read_gitignore_patterns(repo_dir)

    for full in paths:
        if not os.path.exists(full):
            raise FileNotFoundError(
                f"Component '{component_name}' path not found: {full}"
            )
        refs = create_file_references(
            [full], ignore_patterns=ignore_patterns, format="raw", text_only=True
        )["refs"]
        for ref in refs:
            rel_path = _relative_path(ref.path, repo_dir)
            manifest_spec = _format_git_spec(tgt.repo_url, tgt.rev, rel_path)
            resolved.append(
                ResolvedItem(
                    source_type="git",
                    source_ref=tgt.repo_url,
                    source_rev=tgt.rev,
                    source_path=rel_path,
                    context_subpath=rel_path,
                    content=ref.file_content,
                    manifest_spec=manifest_spec,
                    alias=alias if isinstance(alias, str) else None,
                )
            )
    return resolved


def _resolve_http_item(
    url: str,
    alias: Any | None,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> ResolvedItem:
    url_ref = URLReference(
        url,
        format="raw",
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    origin, url_path = _split_url_path(url)
    context_path = _apply_filename_hint(url_path, alias)
    return ResolvedItem(
        source_type="http",
        source_ref=origin,
        source_rev=None,
        source_path=url_path,
        context_subpath=context_path,
        content=url_ref.file_content,
        manifest_spec=url,
        alias=alias if isinstance(alias, str) else None,
    )


def _resolve_youtube_item(
    url: str,
    alias: Any | None,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> ResolvedItem:
    video_id = extract_video_id(url)
    yt_ref = YouTubeReference(
        url,
        format="raw",
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    filename = f"youtube-{video_id}.md"
    if alias and isinstance(alias, str):
        filename = alias if alias.endswith(".md") else f"{alias}.md"
    return ResolvedItem(
        source_type="youtube",
        source_ref="youtube.com",
        source_rev=None,
        source_path=video_id or url,
        context_subpath=filename,
        content=yt_ref.file_content,
        manifest_spec=url,
        alias=alias if isinstance(alias, str) else None,
    )


def _resolve_atproto_items(
    url: str,
    alias: Any | None,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    atproto_overrides: dict[str, Any] | None = None,
) -> list[ResolvedItem]:
    settings = build_atproto_settings(atproto_overrides)
    settings_key = _atproto_settings_cache_key(settings)
    documents = resolve_atproto_url(
        url,
        settings=settings,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )

    alias_prefix: str | None = None
    if isinstance(alias, str) and alias.strip():
        raw_alias = alias.strip()
        if raw_alias.endswith(".md"):
            raw_alias = raw_alias[:-3]
        alias_prefix = _sanitize_path_segment(raw_alias, fallback="atproto")

    items: list[ResolvedItem] = []
    for document in documents:
        context_subpath = document.context_subpath
        if alias_prefix:
            context_subpath = f"{alias_prefix}/{context_subpath}"
        uri = document.uri
        source_path = uri.replace("at://", "", 1) if uri.startswith("at://") else uri
        items.append(
            ResolvedItem(
                source_type="atproto",
                source_ref="atproto",
                source_rev=None,
                source_path=source_path or document.trace_path,
                context_subpath=context_subpath,
                content=document.rendered,
                manifest_spec=url,
                alias=alias if isinstance(alias, str) else None,
                source_created=document.source_created,
                source_modified=document.source_modified,
                atproto_kind=document.kind,
                atproto_settings_key=settings_key,
            )
        )
    return items


def _resolve_arena_items(
    url: str,
    alias: Any | None,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    arena_overrides: dict | None = None,
) -> list[ResolvedItem]:
    from ..cache.arena import get_cached_block_render
    from ..references.arena import (
        _attachment_media_kind,
        _fetch_block,
        _render_block,
        _render_channel_stub,
        build_arena_settings,
        extract_block_id,
        extract_channel_slug,
        resolve_channel,
        warmup_arena_network_stack,
    )
    from ..references.arena import (
        _log as _arena_log,
    )

    settings = build_arena_settings(arena_overrides)
    settings_key = _arena_settings_cache_key(settings)

    block_id = extract_block_id(url)
    if block_id is not None:
        block = _fetch_block(block_id)
        text = (
            _render_block(
                block,
                include_descriptions=settings.include_descriptions,
                include_comments=settings.include_comments,
                include_link_image_descriptions=settings.include_link_image_descriptions,
                include_pdf_content=settings.include_pdf_content,
                include_media_descriptions=settings.include_media_descriptions,
            )
            or ""
        )
        filename = f"arena-block-{block_id}.md"
        if alias and isinstance(alias, str):
            filename = alias if alias.endswith(".md") else f"{alias}.md"
        return [
            ResolvedItem(
                source_type="arena",
                source_ref="are.na",
                source_rev=None,
                source_path=str(block_id),
                context_subpath=filename,
                content=text,
                manifest_spec=url,
                alias=alias if isinstance(alias, str) else None,
                source_created=block.get("connected_at") or block.get("created_at"),
                source_modified=block.get("updated_at"),
                arena_kind="block",
                arena_depth=0,
                arena_settings_key=settings_key,
            )
        ]

    slug = extract_channel_slug(url)
    if not slug:
        raise ValueError(f"Could not parse Are.na URL: {url}")

    warmup_arena_network_stack()
    metadata, flat = resolve_channel(
        slug,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
        settings=settings,
    )

    items: list[ResolvedItem] = []
    dir_name = alias if isinstance(alias, str) else slug
    ch_created = metadata.get("created_at")
    ch_updated = metadata.get("updated_at")

    total = len(flat)
    media_jobs = get_payload_media_jobs()

    def _render_channel_item(
        idx: int, channel_path: str, block: dict[str, Any]
    ) -> ResolvedItem | None:
        block_type = block.get("type", "")
        is_channel = block_type == "Channel" or block.get("base_type") == "Channel"

        if is_channel:
            rendered = _render_channel_stub(block)
        else:
            if block_type in ("Image", "Attachment"):
                from ..runtime import (
                    get_refresh_audio,
                    get_refresh_images,
                    get_refresh_media,
                    get_refresh_videos,
                )

                block_id = block.get("id")
                updated_at = block.get("updated_at") or ""
                refresh_for_block = False
                if block_type == "Image":
                    refresh_for_block = get_refresh_images() or get_refresh_media()
                elif block_type == "Attachment":
                    attachment = block.get("attachment") or {}
                    filename = attachment.get("filename") or ""
                    content_type = attachment.get("content_type") or ""
                    extension = attachment.get("file_extension") or ""
                    media_kind = _attachment_media_kind(
                        filename=filename,
                        extension=extension,
                        content_type=content_type,
                    )
                    refresh_for_block = get_refresh_media() or (
                        media_kind == "image"
                        and get_refresh_images()
                        or media_kind == "video"
                        and get_refresh_videos()
                        or media_kind == "audio"
                        and get_refresh_audio()
                    )
                if not (
                    block_id
                    and updated_at
                    and get_cached_block_render(block_id, updated_at) is not None
                    and not refresh_for_block
                ):
                    block_title = block.get("title") or f"block-{block_id}"
                    _arena_log(
                        f"  rendering {block_type.lower()} ({idx}/{total}): {block_title[:60]}"
                    )
            rendered = _render_block(
                block,
                include_descriptions=settings.include_descriptions,
                include_comments=settings.include_comments,
                include_link_image_descriptions=settings.include_link_image_descriptions,
                include_pdf_content=settings.include_pdf_content,
                include_media_descriptions=settings.include_media_descriptions,
            )
        if rendered is None:
            return None

        bid = block.get("id", "unknown")
        ch_slug = block.get("slug", "")
        label = ch_slug or str(bid)
        channel_parts = [part for part in channel_path.split("/") if part]
        if channel_parts and channel_parts[0] == slug:
            channel_parts = channel_parts[1:]
        arena_depth = len(channel_parts)
        channel_subdir = "/".join(channel_parts)
        context_subpath = (
            f"{dir_name}/{channel_subdir}/{label}.md"
            if channel_subdir
            else f"{dir_name}/{label}.md"
        )
        channel_id = None
        if is_channel:
            raw_id = block.get("id")
            if raw_id is not None:
                channel_id = str(raw_id)
            elif ch_slug:
                channel_id = ch_slug
        return ResolvedItem(
            source_type="arena",
            source_ref="are.na",
            source_rev=None,
            source_path=f"{slug}/{label}",
            context_subpath=context_subpath,
            content=rendered,
            manifest_spec=url,
            alias=alias if isinstance(alias, str) else None,
            source_created=block.get("connected_at") or block.get("created_at"),
            source_modified=block.get("updated_at"),
            dir_created=ch_created,
            dir_modified=ch_updated,
            arena_kind="channel" if is_channel else "block",
            arena_channel_id=channel_id,
            arena_depth=arena_depth,
            arena_settings_key=settings_key,
        )

    tasks = [
        (
            idx - 1,
            (lambda i=idx, cp=channel_path, b=block: _render_channel_item(i, cp, b)),
        )
        for idx, (channel_path, block) in enumerate(flat, 1)
    ]
    for _, rendered_item in run_indexed_tasks_fail_fast(tasks, max_workers=media_jobs):
        if rendered_item is not None:
            items.append(rendered_item)

    return items


def _sanitize_discord_segment(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    cleaned = cleaned.strip("-._")
    return cleaned or fallback


def _discord_scope_parts(
    documents: list[Any],
    *,
    alias: Any | None,
    ext: str,
) -> dict[int, list[str]]:
    alias_prefix: str | None = None
    if isinstance(alias, str) and alias.strip():
        alias_raw = alias.strip()
        if alias_raw.endswith(ext):
            alias_raw = alias_raw[: -len(ext)]
        alias_prefix = _sanitize_discord_segment(alias_raw, "discord")

    channel_index = next(
        (idx for idx, document in enumerate(documents) if document.kind == "channel"),
        None,
    )
    channel_doc = documents[channel_index] if channel_index is not None else None
    channel_slug = (
        _sanitize_discord_segment(
            str(channel_doc.channel_name or channel_doc.channel_id),
            "channel",
        )
        if channel_doc is not None
        else None
    )

    candidates: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        if channel_doc is not None and index == channel_index:
            parts = [channel_slug or "channel"]
        elif (
            channel_doc is not None
            and document.thread_id
            and document.parent_channel_id == channel_doc.channel_id
        ):
            parts = [
                channel_slug or "channel",
                _sanitize_discord_segment(
                    str(document.thread_name or document.thread_id),
                    "thread",
                ),
            ]
        elif document.thread_id:
            parts = [
                _sanitize_discord_segment(
                    str(document.thread_name or document.thread_id),
                    "thread",
                )
            ]
        else:
            parts = [
                _sanitize_discord_segment(
                    str(document.channel_name or document.channel_id),
                    "channel",
                )
            ]

        if alias_prefix:
            parts = [alias_prefix, *parts]

        candidates.append(
            {
                "index": index,
                "parts": parts,
                "fallback_id": str(document.thread_id or document.channel_id or index),
            }
        )

    counts: dict[tuple[tuple[str, ...], str], int] = {}
    for candidate in candidates:
        parent = tuple(candidate["parts"][:-1])
        slug = candidate["parts"][-1]
        key = (parent, slug)
        counts[key] = counts.get(key, 0) + 1

    used: dict[tuple[str, ...], set[str]] = {}
    resolved: dict[int, list[str]] = {}
    for candidate in candidates:
        base_parts = list(candidate["parts"])
        parent = tuple(base_parts[:-1])
        slug = base_parts[-1]
        key = (parent, slug)
        if counts.get(key, 0) > 1:
            slug = _sanitize_discord_segment(
                f"{slug}-{candidate['fallback_id']}",
                slug,
            )
        parent_used = used.setdefault(parent, set())
        chosen = slug
        counter = 2
        while chosen in parent_used:
            chosen = f"{slug}__{counter}"
            counter += 1
        parent_used.add(chosen)
        resolved[int(candidate["index"])] = [*base_parts[:-1], chosen]

    return resolved


def _resolve_discord_items(
    url: str,
    alias: Any | None,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    discord_overrides: dict | None = None,
) -> list[ResolvedItem]:
    settings = build_discord_settings(discord_overrides)
    settings_key = _discord_settings_cache_key(settings)
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
            return []
        raise

    ext = ".yaml" if settings.format == "yaml" else ".md"
    items: list[ResolvedItem] = []
    parsed = parse_discord_url(url)
    target_message_id = (
        parsed.get("message_id") if parsed and parsed.get("kind") == "message" else None
    )
    scope_paths = _discord_scope_parts(documents, alias=alias, ext=ext)

    for index, document in enumerate(documents):
        day_documents = split_discord_document_by_utc_day(document, settings=settings)
        scope_parts = scope_paths.get(index) or ["discord"]
        scope_path = "/".join(scope_parts)

        for day_document in day_documents:
            source_created, source_modified = discord_document_timestamps(day_document)
            date_utc = source_created[:10] if source_created else None
            day_slug = date_utc or "undated"
            context_subpath = f"{scope_path}/{day_slug}{ext}"

            source_path_parts = [day_document.guild_id, day_document.channel_id]
            if day_document.thread_id:
                source_path_parts.append(day_document.thread_id)
            if target_message_id:
                source_path_parts.append(f"message-{target_message_id}")
            source_path_parts.append(f"day-{day_slug}")
            source_path = "/".join(source_path_parts)

            content = render_discord_document_with_metadata(
                day_document,
                settings=settings,
                source_url=url,
            )

            scope_id = day_document.thread_id or day_document.channel_id
            depth = 1 if day_document.thread_id else 0

            items.append(
                ResolvedItem(
                    source_type="discord",
                    source_ref="discord.com",
                    source_rev=None,
                    source_path=source_path,
                    context_subpath=context_subpath,
                    content=content,
                    manifest_spec=url,
                    alias=alias if isinstance(alias, str) else None,
                    source_created=source_created,
                    source_modified=source_modified,
                    discord_kind=day_document.kind,
                    discord_scope_id=scope_id,
                    discord_depth=depth,
                    discord_settings_key=settings_key,
                )
            )

    return items


def _resolve_gist_item(
    raw_url: str,
    filename: str,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> ResolvedItem:
    url_ref = URLReference(
        raw_url,
        format="raw",
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    origin, url_path = _split_url_path(raw_url)
    return ResolvedItem(
        source_type="http",
        source_ref=origin,
        source_rev=None,
        source_path=url_path,
        context_subpath=filename,
        content=url_ref.file_content,
        manifest_spec=raw_url,
    )


def _split_url_path(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    raw_path = unquote(parsed.path or "")
    if not raw_path or raw_path == "/":
        path = "index"
    elif raw_path.endswith("/"):
        path = f"{raw_path}index"
    else:
        path = raw_path
    path = path.lstrip("/")
    return origin, path or "index"


def _apply_filename_hint(path: str, filename_hint: Any | None) -> str:
    if filename_hint is None:
        return path
    if not isinstance(filename_hint, str) or not filename_hint.strip():
        raise ValueError("filename must be a non-empty string")
    if "/" in filename_hint or "\\" in filename_hint:
        raise ValueError("filename must not contain path separators")
    parent = os.path.dirname(path)
    return f"{parent}/{filename_hint}" if parent else filename_hint


def _relative_path(path: str, root: str) -> str:
    path_obj = Path(os.path.abspath(path))
    root_obj = Path(os.path.abspath(root))
    try:
        rel = path_obj.relative_to(root_obj)
    except ValueError as exc:
        raise ValueError(f"Path outside root: {path}") from exc
    rel_str = rel.as_posix()
    if rel_str.startswith("../") or rel_str == "..":
        raise ValueError(f"Path outside root: {path}")
    return rel_str


def _format_git_spec(repo_url: str, rev: str | None, path: str) -> str:
    base = f"{repo_url}@{rev}" if rev else repo_url
    return f"{base}:{path}" if path else base


def _build_suffix(
    ranges: list[tuple[int, int]] | None, symbols: list[str] | None
) -> str | None:
    if symbols:
        safe_symbols = [_sanitize_symbol(sym) for sym in symbols]
        return "__S-" + "-".join(safe_symbols)
    if ranges:
        start, end = ranges[0]
        return f"__L{start}-L{end}"
    return None


def _sanitize_symbol(symbol: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", symbol.strip())
    return safe.strip("-") or "sym"


def _build_component_root(
    component_name: str,
    group_path: Any | None,
    base_name: Any | None,
    path_strategy: str,
) -> Path | None:
    if path_strategy != "by-component":
        return None
    if group_path:
        parts = [group_path] if isinstance(group_path, str) else list(group_path)
        name = base_name if isinstance(base_name, str) and base_name else component_name
        return Path(*parts, name)
    return Path(component_name)


def _resolve_context_path(
    component_name: str,
    item: ResolvedItem,
    suffix: str | None,
    used_paths: set[str],
    identity_paths: dict[tuple[Any, ...], Path],
    path_strategy: str,
    ranges: list[tuple[int, int]] | None,
    symbols: list[str] | None,
    *,
    component_root: Path | None = None,
    use_external_root: bool,
    strip_prefix: Path | None = None,
    skip_external_root: bool = False,
    external_root_prefix: str | None = None,
) -> tuple[Path, bool]:
    rel_path = _build_base_context_path(
        component_name,
        item,
        suffix,
        path_strategy,
        component_root,
        use_external_root,
        strip_prefix,
        skip_external_root,
        external_root_prefix,
    )
    _ensure_relative(rel_path)

    if path_strategy == "on-disk":
        identity = _build_identity_key(item, ranges, symbols)
        existing = identity_paths.get(identity)
        if existing:
            return existing, False
        rel_path = _dedupe_path(rel_path, used_paths)
        identity_paths[identity] = rel_path
        return rel_path, True

    rel_path = _dedupe_path(rel_path, used_paths)
    return rel_path, True


def _strip_subpath_prefix(subpath: Path, prefix: Path | None) -> Path:
    if prefix is None:
        return subpath
    try:
        return subpath.relative_to(prefix)
    except ValueError:
        return subpath


def _build_base_context_path(
    component_name: str,
    item: ResolvedItem,
    suffix: str | None,
    path_strategy: str,
    component_root: Path | None = None,
    use_external_root: bool = True,
    strip_prefix: Path | None = None,
    skip_external_root: bool = False,
    external_root_prefix: str | None = None,
) -> Path:
    if item.source_type == "local":
        subpath = _split_subpath(item.context_subpath)
        if path_strategy == "by-component":
            subpath = _strip_subpath_prefix(subpath, strip_prefix)
        rel_path = (
            subpath
            if path_strategy == "on-disk"
            else (component_root or Path(component_name)) / subpath
        )
    else:
        if item.source_type in ("http", "youtube"):
            ext_path = Path(item.context_subpath)
            if use_external_root:
                ext_path = Path("external") / ext_path
        else:
            subpath = _split_subpath(item.context_subpath)
            if path_strategy == "by-component":
                subpath = _strip_subpath_prefix(subpath, strip_prefix)
            if skip_external_root:
                ext_path = subpath
            else:
                ext_path = _build_external_path(
                    item, subpath, use_external_root, external_root_prefix
                )
        rel_path = (
            ext_path
            if path_strategy == "on-disk"
            else (component_root or Path(component_name)) / ext_path
        )

    if suffix:
        rel_path = _apply_suffix(rel_path, suffix)
    return rel_path


def _build_identity_key(
    item: ResolvedItem,
    ranges: list[tuple[int, int]] | None,
    symbols: list[str] | None,
) -> tuple[Any, ...]:
    ranges_key = tuple(tuple(r) for r in ranges) if ranges else None
    symbols_key = tuple(symbols) if symbols else None
    return (
        item.source_type,
        item.source_ref,
        item.source_rev,
        item.source_path,
        item.arena_settings_key if item.source_type == "arena" else None,
        item.atproto_settings_key if item.source_type == "atproto" else None,
        item.discord_settings_key if item.source_type == "discord" else None,
        ranges_key,
        symbols_key,
    )


def _split_subpath(subpath: str) -> Path:
    parts = [p for p in subpath.split("/") if p]
    return Path(*parts) if parts else Path("index")


def _apply_suffix(path: Path, suffix: str) -> Path:
    if path.suffix:
        name = f"{path.stem}{suffix}{path.suffix}"
    else:
        name = f"{path.name}{suffix}"
    return path.with_name(name)


def _dedupe_path(path: Path, used_paths: set[str]) -> Path:
    candidate = path
    counter = 2
    while candidate.as_posix() in used_paths:
        candidate = _append_disambiguator(path, counter)
        counter += 1
    used_paths.add(candidate.as_posix())
    return candidate


def _append_disambiguator(path: Path, counter: int) -> Path:
    suffix = f"__{counter}"
    if path.suffix:
        name = f"{path.stem}{suffix}{path.suffix}"
    else:
        name = f"{path.name}{suffix}"
    return path.with_name(name)


def _ensure_relative(path: Path) -> None:
    if path.is_absolute():
        raise ValueError(f"Invalid context path: {path}")
    if any(part in {"..", ""} for part in path.parts):
        raise ValueError(f"Invalid context path: {path}")


def _build_external_path(
    item: ResolvedItem,
    subpath: Path,
    use_external_root: bool,
    external_root_prefix: str | None = None,
) -> Path:
    prefix = Path("external") if use_external_root else Path()
    if external_root_prefix:
        prefix = prefix / external_root_prefix
    if item.source_type == "git":
        return (
            prefix
            / _build_git_external_root(item.source_ref, item.source_rev, item.alias)
            / subpath
        )
    if item.source_type == "arena":
        return prefix / subpath if prefix.parts else subpath
    if item.source_type == "atproto":
        return prefix / subpath if prefix.parts else subpath
    if item.source_type == "discord":
        return prefix / subpath if prefix.parts else subpath
    return (
        prefix
        / _build_generic_external_root(item.source_type, item.source_ref)
        / subpath
    )


def _build_http_external_path(item: ResolvedItem) -> Path:
    context_leaf = Path(item.context_subpath).name
    source_leaf = Path(item.source_path).name
    if context_leaf != source_leaf:
        return Path(context_leaf)
    host = _parse_http_host(item.source_ref)
    leaf = _pick_http_leaf(item.context_subpath)
    slug = f"{host}-{leaf}" if host else leaf
    slug = _sanitize_path_segment(slug, fallback="external")
    return Path(slug)


def _parse_http_host(source_ref: str) -> str:
    parsed = urlparse(source_ref)
    host = parsed.hostname or parsed.netloc or source_ref
    if parsed.port:
        host = f"{host}-{parsed.port}"
    return host


def _pick_http_leaf(source_path: str) -> str:
    path = Path(source_path)
    name = path.name
    if not name or name == "index":
        parent = path.parent.name
        if parent and parent != ".":
            return parent
        return "index"
    return name


def _build_git_external_root(
    source_ref: str, source_rev: str | None, alias: str | None = None
) -> Path:
    if alias:
        slug = alias
        if source_rev:
            slug = f"{slug}@{_format_rev_for_path(source_rev)}"
        slug = _sanitize_path_segment(slug, fallback="repo")
        return Path(slug)
    _, host, repo_path = _parse_git_source_ref(source_ref)
    repo_parts = [part for part in repo_path.split("/") if part]
    if repo_parts:
        slug = repo_parts[-1]
    else:
        slug = host or "repo"
    if source_rev:
        slug = f"{slug}@{_format_rev_for_path(source_rev)}"
    slug = _sanitize_path_segment(slug, fallback="repo")
    return Path(slug)


def _build_generic_external_root(source_type: str, source_ref: str) -> Path:
    safe_type = _sanitize_path_segment(source_type, fallback="external")
    safe_ref = _sanitize_path_segment(source_ref, fallback="source")
    return Path(f"{safe_type}-{safe_ref}")


def _parse_git_source_ref(source_ref: str) -> tuple[str, str, str]:
    if source_ref.startswith("git@"):
        host_path = source_ref[4:]
        host, _, path = host_path.partition(":")
        scheme = "ssh"
    else:
        parsed = urlparse(source_ref)
        scheme = (parsed.scheme or "https").lower()
        host = parsed.hostname or parsed.netloc or ""
        path = parsed.path.lstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if not host:
        host = source_ref
    return scheme, host, path


def _format_rev_for_path(value: str) -> str:
    rev = value.strip()
    if re.fullmatch(r"[0-9a-f]{7,40}", rev):
        return rev[:8]
    return rev


def _sanitize_path_segment(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _build_index_entry(
    rel_path: Path,
    item: ResolvedItem,
    ranges: list[tuple[int, int]] | None,
    symbols: list[str] | None,
    content: str,
) -> dict[str, Any]:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "context_path": rel_path.as_posix(),
        "source_type": item.source_type,
        "source_ref": item.source_ref,
        "source_path": item.source_path,
        "source_rev": item.source_rev,
        "range": ranges,
        "symbols": symbols,
        "hash": f"sha256:{digest}",
    }


def _build_manifest_file_entry(
    item: ResolvedItem,
    range_spec: list[tuple[int, int]] | None,
    symbols: list[str] | None,
    comment: str | None,
    extras: dict[str, Any],
) -> dict[str, Any] | str:
    key = "url" if item.source_type in {"http", "atproto"} else "path"
    entry = {key: item.manifest_spec}
    entry.update({k: v for k, v in extras.items() if k not in {"range", "symbols"}})
    if range_spec:
        entry["range"] = _format_ranges(range_spec)
    if symbols:
        entry["symbols"] = symbols
    if comment is not None:
        entry["comment"] = comment
    if len(entry) == 1 and key == "path":
        return item.manifest_spec
    return entry


def _arena_identity(item: ResolvedItem) -> tuple[Any, ...] | None:
    if item.source_type != "arena":
        return None
    settings_key = item.arena_settings_key or ()
    if item.arena_kind == "channel" and item.arena_channel_id:
        return ("arena-channel", item.arena_channel_id, settings_key)
    leaf = item.source_path.rsplit("/", 1)[-1]
    if leaf.isdigit():
        return ("arena-block", leaf, settings_key)
    return None


def _arena_pending_key(
    item: ResolvedItem,
    ranges: list[tuple[int, int]] | None,
    symbols: list[str] | None,
) -> tuple[Any, ...] | None:
    arena_identity = _arena_identity(item)
    if arena_identity is None:
        return None
    ranges_key = tuple(tuple(r) for r in ranges) if ranges else None
    symbols_key = tuple(symbols) if symbols else None
    return (*arena_identity, ranges_key, symbols_key)


def _arena_rank(item: ResolvedItem, encounter_index: int) -> tuple[int, int]:
    depth = item.arena_depth if item.arena_depth is not None else 10**9
    return (depth, encounter_index)


def _materialize_arena_pending(
    arena_pending: dict[tuple[Any, ...], list[_ArenaPendingWrite]],
    context_dir: Path,
    files_to_write: list[tuple[Path, str]],
    files_to_symlink: list[tuple[Path, Path]],
    file_timestamps: dict[Path, tuple[float, float]],
) -> None:
    for occurrences in arena_pending.values():
        canonical = min(
            occurrences,
            key=lambda occ: _arena_rank(occ.item, occ.encounter_index),
        )
        canonical_path = context_dir / canonical.rel_path

        writer = canonical
        if not writer.should_write:
            writer = next(
                (
                    occ
                    for occ in occurrences
                    if occ.rel_path == canonical.rel_path and occ.should_write
                ),
                writer,
            )

        if writer.should_write:
            files_to_write.append((canonical_path, writer.content))
            file_ts = _item_file_ts(writer.item)
            if file_ts:
                file_timestamps[canonical_path] = file_ts
            dir_ts = _item_dir_ts(writer.item)
            if dir_ts:
                file_timestamps.setdefault(canonical_path.parent, dir_ts)

        for occ in occurrences:
            if occ.rel_path == canonical.rel_path:
                continue
            files_to_symlink.append((context_dir / occ.rel_path, canonical_path))


def _dedupe_manifest_entries(files: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()
    for item in files:
        if isinstance(item, dict):
            key = "d:" + json.dumps(item, sort_keys=True, separators=(",", ":"))
        else:
            key = "s:" + str(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _format_ranges(ranges: list[tuple[int, int]]) -> str | list[str]:
    formatted = [f"{start}-{end}" for start, end in ranges]
    if len(formatted) == 1:
        return formatted[0]
    return formatted


def _format_duration_for_manifest(value: timedelta) -> str:
    total = int(value.total_seconds())
    if total % 86400 == 0:
        return f"{total // 86400}d"
    if total % 3600 == 0:
        return f"{total // 3600}h"
    if total % 60 == 0:
        return f"{total // 60}i"
    return f"{total}s"


def _build_normalized_config(
    cfg: dict[str, Any],
    context_cfg: dict[str, Any],
    base_dir: str,
    *,
    include_root: bool,
    include_arena_max_depth: bool,
    arena_overrides: dict[str, Any] | None,
    include_atproto_defaults: bool,
    atproto_overrides: dict[str, Any] | None,
    include_discord_defaults: bool,
    discord_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = dict(cfg)
    if include_root:
        normalized["root"] = base_dir
    else:
        normalized.pop("root", None)
    context = dict(cfg.get("context") or {})
    context.pop("dir", None)
    context.pop("access", None)
    if "path-strategy" in context:
        context["path-strategy"] = context_cfg["path_strategy"]
    if "include-meta" in context:
        context["include-meta"] = context_cfg["include_meta"]
    if context_cfg["agents_text"] is not None:
        context["agents"] = {
            "files": context_cfg["agents_files"] or ["AGENTS.md"],
            "text": context_cfg["agents_text"],
        }
    else:
        context.pop("agents", None)
    if context:
        normalized["context"] = context
    else:
        normalized.pop("context", None)
    if include_arena_max_depth:
        from ..references.arena import build_arena_settings

        arena_settings = build_arena_settings(arena_overrides)
        arena = dict(cfg.get("arena") or {})
        arena.pop("max-depth", None)
        arena.pop("sort", None)
        arena["recurse-depth"] = arena_settings.max_depth
        arena["block-sort"] = arena_settings.sort_order
        if arena_settings.max_blocks_per_channel is None:
            arena.pop("max-blocks-per-channel", None)
        else:
            arena["max-blocks-per-channel"] = arena_settings.max_blocks_per_channel
        if arena_settings.connected_after:
            arena["connected-after"] = (
                arena_settings.connected_after.isoformat().replace("+00:00", "Z")
            )
        else:
            arena.pop("connected-after", None)
        if arena_settings.connected_before:
            arena["connected-before"] = (
                arena_settings.connected_before.isoformat().replace("+00:00", "Z")
            )
        else:
            arena.pop("connected-before", None)
        if arena_settings.created_after:
            arena["created-after"] = arena_settings.created_after.isoformat().replace(
                "+00:00", "Z"
            )
        else:
            arena.pop("created-after", None)
        if arena_settings.created_before:
            arena["created-before"] = arena_settings.created_before.isoformat().replace(
                "+00:00", "Z"
            )
        else:
            arena.pop("created-before", None)
        normalized["arena"] = arena
    if include_atproto_defaults:
        atproto_settings = build_atproto_settings(atproto_overrides)
        atproto = dict(cfg.get("atproto") or {})
        atproto["max-items"] = (
            "all" if atproto_settings.max_items is None else atproto_settings.max_items
        )
        atproto["post-ancestors"] = (
            "all"
            if atproto_settings.post_ancestors is None
            else atproto_settings.post_ancestors
        )
        atproto["thread-depth"] = atproto_settings.thread_depth
        atproto["quote-depth"] = atproto_settings.quote_depth
        atproto["max-replies"] = atproto_settings.max_replies
        atproto["reply-quote-depth"] = atproto_settings.reply_quote_depth
        atproto["replies"] = atproto_settings.replies_filter
        atproto["reposts"] = atproto_settings.reposts_filter
        atproto["likes"] = atproto_settings.likes_filter
        if atproto_settings.created_after:
            atproto["created-after"] = (
                atproto_settings.created_after.isoformat().replace("+00:00", "Z")
            )
        else:
            atproto.pop("created-after", None)
        if atproto_settings.created_before:
            atproto["created-before"] = (
                atproto_settings.created_before.isoformat().replace("+00:00", "Z")
            )
        else:
            atproto.pop("created-before", None)
        atproto["include-media-descriptions"] = (
            atproto_settings.include_media_descriptions
        )
        atproto["include-embed-media-descriptions"] = (
            atproto_settings.include_embed_media_descriptions
        )
        atproto["media-mode"] = atproto_settings.media_mode
        normalized["atproto"] = atproto
    if include_discord_defaults:
        discord_settings = build_discord_settings(discord_overrides)
        discord = dict(cfg.get("discord") or {})
        window = dict(discord.get("window") or {})
        media = dict(discord.get("media") or {})

        discord["format"] = discord_settings.format
        discord["include-system"] = discord_settings.include_system
        discord["include-thread-starters"] = discord_settings.include_thread_starters
        discord["expand-threads"] = discord_settings.expand_threads
        discord["gap-threshold"] = _format_duration_for_manifest(
            discord_settings.gap_threshold
        )

        window["before-messages"] = discord_settings.before_messages
        window["after-messages"] = discord_settings.after_messages
        window["around-messages"] = discord_settings.around_messages
        window["message-context"] = discord_settings.message_context
        window["channel-limit"] = discord_settings.channel_limit
        if discord_settings.start:
            window["start"] = discord_settings.start.isoformat().replace("+00:00", "Z")
        if discord_settings.end:
            window["end"] = discord_settings.end.isoformat().replace("+00:00", "Z")
        if discord_settings.start_message_id:
            window["start-message"] = discord_settings.start_message_id
        if discord_settings.end_message_id:
            window["end-message"] = discord_settings.end_message_id
        if discord_settings.before_duration:
            window["before-duration"] = _format_duration_for_manifest(
                discord_settings.before_duration
            )
        if discord_settings.after_duration:
            window["after-duration"] = _format_duration_for_manifest(
                discord_settings.after_duration
            )
        if discord_settings.around_duration:
            window["around-duration"] = _format_duration_for_manifest(
                discord_settings.around_duration
            )
        discord["window"] = {k: v for k, v in window.items() if v is not None}

        media["describe"] = discord_settings.include_media_descriptions
        media["embed-media-describe"] = (
            discord_settings.include_embed_media_descriptions
        )
        media["file-content"] = discord_settings.include_file_content
        media["mode"] = discord_settings.media_mode
        discord["media"] = media
        normalized["discord"] = discord
    return normalized


def _dump_manifest(data: dict[str, Any]) -> str:
    import yaml

    class _LiteralDumper(yaml.SafeDumper):
        pass

    def _repr_str(dumper, value):
        if "\n" in value:
            return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", value)

    _LiteralDumper.add_representer(str, _repr_str)
    return yaml.dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        Dumper=_LiteralDumper,
    )


def _dump_index(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, indent=2, sort_keys=True)


def _parse_arena_timestamp(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(iso).timestamp()
    except (ValueError, TypeError):
        return None


def _item_file_ts(item: ResolvedItem) -> tuple[float, float] | None:
    atime = _parse_arena_timestamp(item.source_created)
    mtime = _parse_arena_timestamp(item.source_modified)
    if atime is not None and mtime is not None:
        return (atime, mtime)
    if mtime is not None:
        return (mtime, mtime)
    if atime is not None:
        return (atime, atime)
    return None


def _item_dir_ts(item: ResolvedItem) -> tuple[float, float] | None:
    atime = _parse_arena_timestamp(item.dir_created)
    mtime = _parse_arena_timestamp(item.dir_modified)
    if atime is not None and mtime is not None:
        return (atime, mtime)
    if mtime is not None:
        return (mtime, mtime)
    if atime is not None:
        return (atime, atime)
    return None


def _apply_timestamps(timestamps: dict[Path, tuple[float, float]]) -> None:
    files = []
    dirs = []
    for path, times in timestamps.items():
        if path.is_dir():
            dirs.append((path, times))
        else:
            files.append((path, times))
    for path, (atime, mtime) in files:
        try:
            os.utime(path, (atime, mtime))
        except OSError:
            pass
    for path, (atime, mtime) in dirs:
        try:
            os.utime(path, (atime, mtime))
        except OSError:
            pass


def _write_files(files: list[tuple[Path, str]]) -> None:
    for path, content in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _create_symlinks(links: list[tuple[Path, Path]]) -> None:
    for dest, source in links:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(source.resolve())


def _apply_read_only(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            _chmod(Path(dirpath) / name, 0o444)
        for name in dirnames:
            _chmod(Path(dirpath) / name, 0o555)
    _chmod(root, 0o555)


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        return


def _collect_parent_dirs(rel_path: str, target: set[str]) -> None:
    path = Path(rel_path)
    parent = path.parent
    while parent and parent.as_posix() != ".":
        target.add(parent.as_posix())
        parent = parent.parent


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _all_read_only(
    context_dir: Path, expected_dirs: set[str], expected_files: list[str] | set[str]
) -> bool:
    for rel in expected_files:
        path = context_dir / rel
        if not _is_read_only(path):
            return False
    for rel in expected_dirs:
        path = context_dir / rel
        if not _is_read_only(path):
            return False
    return True


def _is_read_only(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    return (
        (mode & stat.S_IWUSR) == 0
        and (mode & stat.S_IWGRP) == 0
        and (mode & stat.S_IWOTH) == 0
    )


def _clear_context_dir(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        raise ValueError(f"Context dir exists and is not a directory: {path}")
    if not path.exists():
        return
    _make_writable(path)
    for item in path.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


def find_untracked_files(path: Path) -> list[str]:
    if not path.exists() or not path.is_dir():
        return []

    index_path = path / "index.json"
    tracked_paths: set[str] = set()

    if index_path.exists():
        try:
            import json

            with open(index_path, "r", encoding="utf-8") as fh:
                index_data = json.load(fh)
            components = index_data.get("components", {})
            for entries in components.values():
                for entry in entries:
                    ctx_path = entry.get("context_path")
                    if ctx_path:
                        tracked_paths.add(ctx_path)
            tracked_paths.add("manifest.yaml")
            tracked_paths.add("index.json")
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass

    untracked: list[str] = []
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                rel_path = file_path.relative_to(path).as_posix()
            except ValueError:
                continue
            if rel_path not in tracked_paths:
                untracked.append(rel_path)

    return untracked


def _make_writable(path: Path) -> None:
    for root, dirs, files in os.walk(path, topdown=False):
        root_path = Path(root)
        for name in files:
            target = root_path / name
            if target.is_symlink():
                continue
            _chmod(target, 0o666)
        for name in dirs:
            target = root_path / name
            if target.is_symlink():
                continue
            _chmod(target, 0o777)
    if not path.is_symlink():
        _chmod(path, 0o777)
