"""
Manifest parsing, normalization, and validation.

This package provides types and utilities for working with
contextualize manifest files.

Public API:
    parse_manifest(source) -> Manifest
        Parse a manifest from file, string, or IO.

    normalize_components(data) -> list[dict]
        Normalize component definitions.

    validate_manifest(manifest) -> list[str]
        Validate a manifest, returning errors.

Types:
    Manifest - Top-level manifest structure
    Component - A component in a manifest
    FileSpec - A file specification
    ManifestConfig - Manifest configuration
    ContextConfig - Context/hydration configuration
"""

from .types import (
    Component,
    ContextConfig,
    FileSpec,
    Manifest,
    ManifestConfig,
)

from .parse import (
    parse_manifest,
    parse_manifest_data,
)

from .normalize import (
    normalize_components,
    extract_groups,
    coerce_file_spec,
    component_selectors,
    GROUP_DELIMITER,
    GROUP_PATH_KEY,
    GROUP_BASE_KEY,
)

from .validate import (
    validate_manifest,
    validate_component_dict,
)

__all__ = [
    # Types
    "Component",
    "ContextConfig",
    "FileSpec",
    "Manifest",
    "ManifestConfig",
    # Parsing
    "parse_manifest",
    "parse_manifest_data",
    # Normalization
    "normalize_components",
    "extract_groups",
    "coerce_file_spec",
    "component_selectors",
    "GROUP_DELIMITER",
    "GROUP_PATH_KEY",
    "GROUP_BASE_KEY",
    # Validation
    "validate_manifest",
    "validate_component_dict",
]


# Backwards compatibility aliases
def normalize_manifest_components(components):
    """Alias for normalize_components (backwards compatibility)."""
    return normalize_components(components)


def _coerce_file_spec(spec):
    """Alias for coerce_file_spec (backwards compatibility)."""
    return coerce_file_spec(spec)


def _component_selectors(comp):
    """Alias for component_selectors (backwards compatibility)."""
    return component_selectors(comp)
