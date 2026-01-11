"""
Contextualize - Content gathering and formatting for LLMs.

High-level API (mirrors CLI commands):
    cat(paths, **options) -> str
        Gather files and format them for LLM input.

    map(paths, **options) -> str
        Generate repository structure maps.

    payload(manifest, **options) -> str
        Render a manifest to payload text.

    hydrate(manifest, **options) -> None
        Materialize a manifest into a directory structure.

    shell(commands, **options) -> str
        Capture shell command outputs.

    paste(**options) -> str
        Capture clipboard content.

Low-level API (building blocks):
    from contextualize.core import (
        resolve,           # Resolve targets to references
        format_content,    # Format content with handlers
        count_tokens,      # Count tokens in text
        process_text,      # Process and format text
    )

    from contextualize.core.manifest import (
        parse_manifest,    # Parse manifest files
        Manifest,          # Manifest dataclass
        Component,         # Component dataclass
    )

    from contextualize.core.references import (
        Reference,         # Reference protocol
        FileReference,     # Local file reference
        URLReference,      # URL reference
        ShellReference,    # Command output reference
    )
"""

from typing import Any

# Version
__version__ = "0.1.0"


def cat(
    paths: list[str],
    *,
    format: str = "md",
    label: str = "relative",
    ignore: list[str] | None = None,
    tokens: bool = False,
    token_target: str = "cl100k_base",
    inject: bool = False,
    depth: int = 5,
) -> str:
    """Gather files and format them for LLM input.

    Args:
        paths: List of file paths, directories, URLs, or git targets
        format: Output format (md, xml, shell, raw)
        label: Label style (relative, name, ext)
        ignore: Patterns to ignore
        tokens: Include token counts
        token_target: Token counting target/encoding
        inject: Process {cx::...} markers
        depth: Maximum depth for content injection

    Returns:
        Formatted content string
    """
    from .core.references import create_file_references, concat_refs

    result = create_file_references(
        paths,
        ignore_patterns=ignore,
        format=format,
        label=label,
        include_token_count=tokens,
        token_target=token_target,
        inject=inject,
        depth=depth,
    )
    return result["concatenated"]


def map(
    paths: list[str],
    *,
    max_tokens: int = 10000,
    format: str = "raw",
    ignore: list[str] | None = None,
    tokens: bool = False,
    token_target: str = "cl100k_base",
) -> str:
    """Generate repository structure maps.

    Args:
        paths: List of paths to map
        max_tokens: Maximum tokens for the map
        format: Output format (raw, md, xml)
        ignore: Patterns to ignore
        tokens: Include token annotations
        token_target: Token counting target/encoding

    Returns:
        Repository map string
    """
    from .core.repomap import generate_repo_map_data

    result = generate_repo_map_data(
        paths,
        max_tokens,
        format,
        ignore=ignore,
        annotate_tokens=tokens,
        token_target=token_target,
    )

    if "error" in result:
        raise ValueError(result["error"])

    return result["repo_map"]


def payload(
    manifest: str,
    *,
    inject: bool = False,
    depth: int = 5,
    exclude: list[str] | None = None,
    map_mode: bool = False,
    map_keys: list[str] | None = None,
    token_target: str = "cl100k_base",
) -> str:
    """Render a manifest to payload text.

    Args:
        manifest: Path to manifest YAML file
        inject: Process {cx::...} markers
        depth: Maximum depth for content injection
        exclude: Component names to exclude
        map_mode: Generate repo maps for components
        map_keys: Component names to map
        token_target: Token counting target/encoding

    Returns:
        Rendered payload string
    """
    from .core.payload import render_manifest

    result = render_manifest(
        manifest,
        inject=inject,
        depth=depth,
        exclude_keys=exclude,
        map_mode=map_mode,
        map_keys=map_keys,
        token_target=token_target,
    )
    return result.payload


def hydrate(
    manifest: str,
    *,
    dir: str | None = None,
    access: str | None = None,
    clear: bool = False,
    force: bool = False,
) -> None:
    """Materialize a manifest into a directory structure.

    Args:
        manifest: Path to manifest YAML file
        dir: Output directory (overrides manifest config)
        access: Access mode (writable, read-only)
        clear: Clear existing directory first
        force: Force overwrite existing files
    """
    from .core.hydrate import hydrate_manifest

    hydrate_manifest(
        manifest,
        output_dir_override=dir,
        access_override=access,
        clear=clear,
        force=force,
    )


def shell(
    commands: list[str],
    *,
    format: str = "shell",
    capture_stderr: bool = True,
    shell_executable: str | None = None,
) -> str:
    """Capture shell command outputs.

    Args:
        commands: List of commands to execute
        format: Output format (shell, md, xml, raw)
        capture_stderr: Include stderr in output
        shell_executable: Shell to use for execution

    Returns:
        Formatted command outputs
    """
    from .core.references import create_command_references

    result = create_command_references(
        commands,
        format=format,
        capture_stderr=capture_stderr,
        shell_executable=shell_executable,
    )
    return result["concatenated"]


def paste(
    *,
    format: str = "md",
    label: str = "clipboard",
    tokens: bool = False,
    token_target: str = "cl100k_base",
) -> str:
    """Capture clipboard content.

    Args:
        format: Output format (md, xml, shell, raw)
        label: Label for the content
        tokens: Include token count
        token_target: Token counting target/encoding

    Returns:
        Formatted clipboard content
    """
    from .core.references import create_clipboard_reference

    ref = create_clipboard_reference(
        format=format,
        label=label,
        include_token_count=tokens,
        token_target=token_target,
    )
    return ref.output


# Low-level API re-exports
from .core import (
    count_tokens,
    process_text,
    resolve,
    Reference,
    FileReference,
    URLReference,
    GitRevFileReference,
    GitCacheReference,
    ClipboardReference,
    ShellReference,
    CommandReference,
)

from .core.formats import format_content, register_format, FORMATS

from .core.manifest import (
    parse_manifest,
    Manifest,
    Component,
    FileSpec,
    ManifestConfig,
    ContextConfig,
)


__all__ = [
    # Version
    "__version__",
    # High-level API
    "cat",
    "map",
    "payload",
    "hydrate",
    "shell",
    "paste",
    # Low-level API - Core
    "count_tokens",
    "process_text",
    "resolve",
    "format_content",
    "register_format",
    "FORMATS",
    # Low-level API - References
    "Reference",
    "FileReference",
    "URLReference",
    "GitRevFileReference",
    "GitCacheReference",
    "ClipboardReference",
    "ShellReference",
    "CommandReference",
    # Low-level API - Manifest
    "parse_manifest",
    "Manifest",
    "Component",
    "FileSpec",
    "ManifestConfig",
    "ContextConfig",
]
