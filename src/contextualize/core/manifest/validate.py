"""Manifest validation utilities."""

import os
from typing import Any

from .types import Component, FileSpec, Manifest


def validate_manifest(manifest: Manifest) -> list[str]:
    """Validate a manifest and return a list of errors.

    Args:
        manifest: Parsed Manifest object

    Returns:
        List of error messages (empty if valid)
    """
    errors: list[str] = []

    # Validate config
    errors.extend(_validate_config(manifest.config))

    # Validate components
    seen_names: set[str] = set()
    for comp in manifest.components:
        comp_errors = _validate_component(comp, seen_names)
        errors.extend(comp_errors)
        seen_names.add(comp.name)

    # Validate groups reference existing components
    all_names = {c.name for c in manifest.components}
    for group_name, component_names in manifest.groups.items():
        for name in component_names:
            if name not in all_names:
                errors.append(
                    f"Group '{group_name}' references unknown component: {name}"
                )

    return errors


def _validate_config(config) -> list[str]:
    """Validate manifest configuration."""
    errors: list[str] = []

    # Validate root directory
    if config.root and not os.path.isdir(config.root):
        errors.append(f"Config root directory does not exist: {config.root}")

    # Validate context settings
    ctx = config.context
    if ctx.access not in ("writable", "read-only"):
        errors.append(f"Invalid context.access: {ctx.access} (must be 'writable' or 'read-only')")

    if ctx.path_strategy not in ("on-disk", "by-component"):
        errors.append(f"Invalid context.path-strategy: {ctx.path_strategy}")

    # Validate link settings
    if config.link_depth < 0:
        errors.append(f"link-depth must be non-negative: {config.link_depth}")

    if config.link_scope not in ("all", "first"):
        errors.append(f"Invalid link-scope: {config.link_scope} (must be 'all' or 'first')")

    return errors


def _validate_component(comp: Component, seen_names: set[str]) -> list[str]:
    """Validate a single component."""
    errors: list[str] = []

    # Name validation
    if not comp.name:
        errors.append("Component must have a name")
    elif comp.name in seen_names:
        errors.append(f"Duplicate component name: {comp.name}")

    # Must have either files or text
    if not comp.is_text_only and not comp.files:
        errors.append(f"Component '{comp.name}' must have either 'files' or 'text'")

    # Validate files
    for i, file_spec in enumerate(comp.files):
        file_errors = _validate_file_spec(file_spec, comp.name, i)
        errors.extend(file_errors)

    # Validate link settings
    if comp.link_depth < 0:
        errors.append(f"Component '{comp.name}' link-depth must be non-negative")

    if comp.link_scope not in ("all", "first"):
        errors.append(
            f"Component '{comp.name}' invalid link-scope: {comp.link_scope}"
        )

    return errors


def _validate_file_spec(spec: FileSpec, comp_name: str, index: int) -> list[str]:
    """Validate a file specification."""
    errors: list[str] = []

    if not spec.path:
        errors.append(f"Component '{comp_name}' file[{index}] has empty path")

    # Check for conflicting options
    if spec.options.get("filename") and spec.options.get("wrap"):
        # This is fine, just informational
        pass

    return errors


def validate_component_dict(comp: dict[str, Any]) -> list[str]:
    """Validate a raw component dict (before normalization).

    Args:
        comp: Raw component dict

    Returns:
        List of error messages
    """
    errors: list[str] = []

    if not isinstance(comp, dict):
        return ["Component must be a mapping"]

    # Check for required keys
    has_group = "group" in comp
    has_text = "text" in comp
    has_name = "name" in comp
    has_files = "files" in comp

    if has_group:
        # Group validation
        if "components" not in comp:
            errors.append(f"Group '{comp.get('group')}' must have 'components'")
    else:
        # Component validation
        if not has_text and not has_files:
            errors.append(
                f"Component '{comp.get('name', '<unnamed>')}' must have either 'text' or 'files'"
            )

    return errors
