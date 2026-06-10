"""Manifest payload rendering - public API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from .build import build_payload_impl
from .manifest import normalize_components
from .source import load_manifest_source


@dataclass
class PayloadResult:
    payload: str
    input_refs: list[Any]
    trace_items: list[Any]
    base_dir: str
    skipped_paths: list[str]
    skip_impact: dict[str, Any]


def build_payload(
    components: list[dict[str, Any]],
    base_dir: str,
    *,
    inject: bool = False,
    depth: int = 5,
    link_depth: int = 0,
    link_scope: str = "all",
    link_skip: list[str] | None = None,
    embedded_resolution: bool = False,
    exclude_keys: list[str] | None = None,
    map_mode: bool = False,
    map_keys: list[str] | None = None,
    token_target: str = "cl100k_base",
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    plugin_overrides: dict[str, Any] | None = None,
) -> PayloadResult:
    payload, input_refs, trace_items, base, skipped, impact = build_payload_impl(
        components,
        base_dir,
        inject=inject,
        depth=depth,
        link_depth_default=link_depth,
        link_scope_default=link_scope,
        link_skip_default=link_skip or [],
        embedded_resolution_default=embedded_resolution,
        exclude_keys=exclude_keys,
        map_mode=map_mode,
        map_keys=map_keys,
        token_target=token_target,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
        plugin_overrides=plugin_overrides,
    )
    return PayloadResult(payload, input_refs, trace_items, base, skipped, impact)


def _prepare_manifest_payload(
    data: dict[str, Any], base_dir: str
) -> tuple[
    list[dict[str, Any]],
    str,
    int,
    str,
    list[str],
    bool,
    timedelta | None,
]:
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping with 'config' and 'components'")

    cfg = data.get("config", {})
    if "root" in cfg:
        base_dir = os.path.expanduser(cfg.get("root") or "~")

    comps = data.get("components")
    if not isinstance(comps, list):
        raise ValueError("'components' must be a list")
    comps = normalize_components(comps)

    link_depth_default = int(cfg.get("link-depth", 0) or 0)
    link_scope_default = (cfg.get("link-scope", "all") or "all").lower()
    link_skip_default = cfg.get("link-skip", [])
    if link_skip_default is None:
        link_skip_default = []
    elif isinstance(link_skip_default, str):
        link_skip_default = [link_skip_default]

    removed = [
        key for key in ("target-depth", "target-scope", "include-parent") if key in cfg
    ]
    if removed:
        keys = ", ".join(removed)
        raise ValueError(
            f"config uses removed embedded traversal key(s): {keys}. Use embedded-resolution instead."
        )
    embedded_resolution_default = cfg.get("embedded-resolution", False)
    if not isinstance(embedded_resolution_default, bool):
        raise ValueError("embedded-resolution must be a boolean")

    context_cfg = cfg.get("context", {})
    raw_ttl = context_cfg.get("cache-ttl") if isinstance(context_cfg, dict) else None
    manifest_cache_ttl = None
    if raw_ttl is not None:
        from ..cache import parse_duration

        if isinstance(raw_ttl, str):
            manifest_cache_ttl = parse_duration(raw_ttl)
        elif isinstance(raw_ttl, (int, float)):
            manifest_cache_ttl = timedelta(days=raw_ttl)

    return (
        comps,
        base_dir,
        link_depth_default,
        link_scope_default,
        link_skip_default,
        embedded_resolution_default,
        manifest_cache_ttl,
    )


def render_manifest(
    manifest_path: str,
    *,
    inject: bool = False,
    depth: int = 5,
    exclude_keys: list[str] | None = None,
    map_mode: bool = False,
    map_keys: list[str] | None = None,
    token_target: str = "cl100k_base",
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    plugin_overrides: dict[str, Any] | None = None,
) -> PayloadResult:
    """
    Load YAML and assemble payload with mdlinks.
    Respects top-level config keys:
      - root: base directory for relative paths
      - link-depth: default depth for Markdown link traversal
      - link-scope: "first" or "all" (default: all)
      - link-skip: list of paths to skip when resolving Markdown links
      - embedded-resolution: resolve plugin targets attached to collected items
      - context.cache-ttl: default cache TTL for URL content
    """
    source = load_manifest_source(manifest_path)
    data = source.data
    base_dir_default = source.manifest_cwd
    (
        comps,
        base_dir,
        link_depth_default,
        link_scope_default,
        link_skip_default,
        embedded_resolution_default,
        manifest_cache_ttl,
    ) = _prepare_manifest_payload(data, base_dir_default)

    effective_ttl = cache_ttl if cache_ttl is not None else manifest_cache_ttl

    return build_payload(
        comps,
        base_dir,
        inject=inject,
        depth=depth,
        link_depth=link_depth_default,
        link_scope=link_scope_default,
        link_skip=link_skip_default,
        embedded_resolution=embedded_resolution_default,
        exclude_keys=exclude_keys,
        map_mode=map_mode,
        map_keys=map_keys,
        token_target=token_target,
        use_cache=use_cache,
        cache_ttl=effective_ttl,
        refresh_cache=refresh_cache,
        plugin_overrides=plugin_overrides,
    )


def render_manifest_data(
    data: dict[str, Any],
    manifest_cwd: str,
    *,
    inject: bool = False,
    depth: int = 5,
    exclude_keys: list[str] | None = None,
    map_mode: bool = False,
    map_keys: list[str] | None = None,
    token_target: str = "cl100k_base",
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    plugin_overrides: dict[str, Any] | None = None,
) -> PayloadResult:
    """
    Assemble from an already-parsed YAML mapping (used for stdin case).
    """
    (
        comps,
        base_dir,
        link_depth_default,
        link_scope_default,
        link_skip_default,
        embedded_resolution_default,
        manifest_cache_ttl,
    ) = _prepare_manifest_payload(data, manifest_cwd)

    effective_ttl = cache_ttl if cache_ttl is not None else manifest_cache_ttl

    return build_payload(
        comps,
        base_dir,
        inject=inject,
        depth=depth,
        link_depth=link_depth_default,
        link_scope=link_scope_default,
        link_skip=link_skip_default,
        embedded_resolution=embedded_resolution_default,
        exclude_keys=exclude_keys,
        map_mode=map_mode,
        map_keys=map_keys,
        token_target=token_target,
        use_cache=use_cache,
        cache_ttl=effective_ttl,
        refresh_cache=refresh_cache,
        plugin_overrides=plugin_overrides,
    )
