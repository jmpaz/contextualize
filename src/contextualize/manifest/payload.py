"""Manifest payload rendering - public API."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

from .build import build_payload_impl
from .manifest import normalize_components


@dataclass
class PayloadResult:
    payload: str
    input_refs: List[Any]
    trace_items: List[Any]
    base_dir: str
    skipped_paths: List[str]
    skip_impact: Dict[str, Any]


def assemble_payload(
    components: List[Dict[str, Any]],
    base_dir: str,
    *,
    inject: bool = False,
    depth: int = 5,
) -> str:
    normalized_components = normalize_components(components)
    return build_payload(
        normalized_components,
        base_dir,
        inject=inject,
        depth=depth,
        link_depth=0,
        link_scope="all",
        link_skip=None,
    ).payload


def build_payload(
    components: List[Dict[str, Any]],
    base_dir: str,
    *,
    inject: bool = False,
    depth: int = 5,
    link_depth: int = 0,
    link_scope: str = "all",
    link_skip: Optional[List[str]] = None,
    exclude_keys: list[str] | None = None,
    map_mode: bool = False,
    map_keys: list[str] | None = None,
    token_target: str = "cl100k_base",
) -> PayloadResult:
    payload, input_refs, trace_items, base, skipped, impact = build_payload_impl(
        components,
        base_dir,
        inject=inject,
        depth=depth,
        link_depth_default=link_depth,
        link_scope_default=link_scope,
        link_skip_default=link_skip or [],
        exclude_keys=exclude_keys,
        map_mode=map_mode,
        map_keys=map_keys,
        token_target=token_target,
    )
    return PayloadResult(payload, input_refs, trace_items, base, skipped, impact)


def render_from_yaml(
    manifest_path: str,
    *,
    inject: bool = False,
    depth: int = 5,
) -> str:
    """Render manifest to payload text."""
    return render_manifest(manifest_path, inject=inject, depth=depth).payload


def render_manifest(
    manifest_path: str,
    *,
    inject: bool = False,
    depth: int = 5,
    exclude_keys: list[str] | None = None,
    map_mode: bool = False,
    map_keys: list[str] | None = None,
    token_target: str = "cl100k_base",
) -> PayloadResult:
    """
    Load YAML and assemble payload with mdlinks.
    Respects top-level config keys:
      - root: base directory for relative paths
      - link-depth: default depth for Markdown link traversal
      - link-scope: "first" or "all" (default: all)
      - link-skip: list of paths to skip when resolving Markdown links
    """
    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping with 'config' and 'components'")

    cfg = data.get("config", {})
    if "root" in cfg:
        raw = cfg.get("root") or "~"
        base_dir = os.path.expanduser(raw)
    else:
        base_dir = os.path.dirname(os.path.abspath(manifest_path))

    comps = data.get("components")
    if not isinstance(comps, list):
        raise ValueError("'components' must be a list")
    comps = normalize_components(comps)

    link_depth_default = int(cfg.get("link-depth", 0) or 0)
    link_scope_default = (cfg.get("link-scope", "all") or "all").lower()
    link_skip_default = cfg.get("link-skip", [])
    if isinstance(link_skip_default, str):
        link_skip_default = [link_skip_default]

    return build_payload(
        comps,
        base_dir,
        inject=inject,
        depth=depth,
        link_depth=link_depth_default,
        link_scope=link_scope_default,
        link_skip=link_skip_default,
        exclude_keys=exclude_keys,
        map_mode=map_mode,
        map_keys=map_keys,
        token_target=token_target,
    )


def render_manifest_data(
    data: Dict[str, Any],
    manifest_cwd: str,
    *,
    inject: bool = False,
    depth: int = 5,
    exclude_keys: list[str] | None = None,
    map_mode: bool = False,
    map_keys: list[str] | None = None,
    token_target: str = "cl100k_base",
) -> PayloadResult:
    """
    Assemble from an already-parsed YAML mapping (used for stdin case).
    """
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping with 'config' and 'components'")

    cfg = data.get("config", {})
    if "root" in cfg:
        base_dir = os.path.expanduser(cfg.get("root") or "~")
    else:
        base_dir = manifest_cwd

    comps = data.get("components")
    if not isinstance(comps, list):
        raise ValueError("'components' must be a list")
    comps = normalize_components(comps)

    link_depth_default = int(cfg.get("link-depth", 0) or 0)
    link_scope_default = (cfg.get("link-scope", "all") or "all").lower()
    link_skip_default = cfg.get("link-skip", [])
    if isinstance(link_skip_default, str):
        link_skip_default = [link_skip_default]

    return build_payload(
        comps,
        base_dir,
        inject=inject,
        depth=depth,
        link_depth=link_depth_default,
        link_scope=link_scope_default,
        link_skip=link_skip_default,
        exclude_keys=exclude_keys,
        map_mode=map_mode,
        map_keys=map_keys,
        token_target=token_target,
    )
