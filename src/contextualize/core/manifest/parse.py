"""Manifest parsing utilities."""

import os
from io import StringIO
from pathlib import Path
from typing import Any, IO, Union

import yaml

from .types import (
    Component,
    ContextConfig,
    FileSpec,
    Manifest,
    ManifestConfig,
)
from .normalize import normalize_components, extract_groups


def parse_manifest(source: Union[str, Path, IO]) -> Manifest:
    """Parse a manifest from a file path, string, or file object.

    Args:
        source: Path to manifest file, YAML string, or file object

    Returns:
        Parsed Manifest object

    Raises:
        ValueError: If the manifest structure is invalid
    """
    # Load the raw YAML data
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            base_dir = str(path.parent.absolute())
        else:
            # Treat as YAML string
            data = yaml.safe_load(source)
            base_dir = os.getcwd()
    else:
        # File object
        data = yaml.safe_load(source)
        base_dir = os.getcwd()

    return parse_manifest_data(data, base_dir)


def parse_manifest_data(data: dict[str, Any], base_dir: str) -> Manifest:
    """Parse a manifest from already-loaded data.

    Args:
        data: Parsed YAML data
        base_dir: Base directory for resolving relative paths

    Returns:
        Parsed Manifest object
    """
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping with 'config' and 'components'")

    # Parse config
    config = _parse_config(data.get("config", {}), base_dir)

    # Parse components
    raw_components = data.get("components", [])
    if not isinstance(raw_components, list):
        raise ValueError("'components' must be a list")

    normalized = normalize_components(raw_components)
    components = [_dict_to_component(c) for c in normalized]

    # Extract groups
    groups = extract_groups(normalized)

    return Manifest(
        config=config,
        components=components,
        groups=groups,
        raw_data=data,
    )


def _parse_config(cfg: dict[str, Any], base_dir: str) -> ManifestConfig:
    """Parse the config section of a manifest."""
    # Handle root directory
    root = cfg.get("root")
    if root:
        root = os.path.expanduser(root)
        if not os.path.isabs(root):
            root = os.path.join(base_dir, root)
    else:
        root = base_dir

    # Handle context config
    ctx_cfg = cfg.get("context", {})
    context = ContextConfig(
        dir=ctx_cfg.get("dir", ".context"),
        access=ctx_cfg.get("access", "writable"),
        path_strategy=ctx_cfg.get("path-strategy", "on-disk"),
        include_meta=ctx_cfg.get("include-meta", True),
        agents=ctx_cfg.get("agents", {}),
    )

    # Handle link-skip
    link_skip = cfg.get("link-skip", [])
    if isinstance(link_skip, str):
        link_skip = [link_skip]

    return ManifestConfig(
        root=root,
        link_depth=int(cfg.get("link-depth", 0) or 0),
        link_scope=str(cfg.get("link-scope", "all") or "all").lower(),
        link_skip=link_skip,
        context=context,
    )


def _parse_file_spec(spec: Any) -> FileSpec:
    """Parse a file specification into a FileSpec object."""
    if isinstance(spec, str):
        return FileSpec(path=spec)

    if isinstance(spec, dict):
        path = spec.get("path") or spec.get("target") or spec.get("url")
        if not path or not isinstance(path, str):
            raise ValueError(f"Invalid file spec: expected 'path' string, got {spec}")

        options = {k: v for k, v in spec.items() if k not in ("path", "target", "url")}
        return FileSpec(path=path, options=options)

    raise ValueError(f"Invalid file spec: expected string or mapping, got {type(spec)}")


def _dict_to_component(comp: dict[str, Any]) -> Component:
    """Convert a normalized component dict to a Component object."""
    # Parse files
    files = []
    raw_files = comp.get("files", [])
    if raw_files:
        for spec in raw_files:
            files.append(_parse_file_spec(spec))

    # Handle link-skip
    link_skip = comp.get("link-skip", [])
    if isinstance(link_skip, str):
        link_skip = [link_skip]

    # Get group path
    group_path = comp.get("__group_path")
    if group_path and not isinstance(group_path, tuple):
        group_path = tuple(group_path)

    return Component(
        name=comp.get("name", ""),
        files=files,
        text=comp.get("text"),
        prefix=comp.get("prefix"),
        suffix=comp.get("suffix"),
        wrap=comp.get("wrap"),
        comment=comp.get("comment"),
        link_depth=int(comp.get("link-depth", 0) or 0),
        link_scope=str(comp.get("link-scope", "all") or "all").lower(),
        link_skip=link_skip,
        group_path=group_path,
        group_base=comp.get("__group_base"),
    )
