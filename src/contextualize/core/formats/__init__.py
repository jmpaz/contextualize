"""
Output format handlers for content wrapping.

This package provides format handlers for wrapping content in
various output formats: raw, markdown, xml, and shell.

Public API:
    format_content(content, fmt, label, **options) -> str
        Format content using the specified format handler.

    register_format(name, handler) -> None
        Register a custom format handler.

    FORMATS: dict[str, FormatHandler]
        Registry of available format handlers.

Built-in Formats:
    raw   - No wrapping, content as-is
    md    - Markdown code fence with label
    xml   - XML tags with label attribute
    shell - Shell-style header comment
"""

from typing import Callable, Protocol

from .raw import wrap_raw
from .md import wrap_md
from .xml import wrap_xml
from .shell import wrap_shell


class FormatHandler(Protocol):
    """Protocol for format handlers."""

    def __call__(
        self,
        content: str,
        label: str | None = None,
        **kwargs,
    ) -> str:
        """Format content with the given label and options."""
        ...


# Format registry
FORMATS: dict[str, FormatHandler] = {
    "raw": wrap_raw,
    "md": wrap_md,
    "markdown": wrap_md,  # Alias
    "xml": wrap_xml,
    "shell": wrap_shell,
}


def register_format(name: str, handler: FormatHandler) -> None:
    """Register a custom format handler.

    Args:
        name: Name of the format
        handler: Function that takes (content, label, **kwargs) and returns formatted string
    """
    FORMATS[name] = handler


def format_content(
    content: str,
    fmt: str = "raw",
    label: str | None = None,
    **kwargs,
) -> str:
    """Format content using the specified format handler.

    Args:
        content: The content to format
        fmt: Format name (raw, md, xml, shell)
        label: Optional label for the content
        **kwargs: Additional options passed to the format handler

    Returns:
        Formatted content string

    Raises:
        KeyError: If the format is not registered
    """
    if fmt not in FORMATS:
        raise KeyError(f"Unknown format: {fmt}. Available formats: {list(FORMATS.keys())}")
    return FORMATS[fmt](content, label, **kwargs)


__all__ = [
    # Public API
    "format_content",
    "register_format",
    "FORMATS",
    "FormatHandler",
    # Individual format handlers
    "wrap_raw",
    "wrap_md",
    "wrap_xml",
    "wrap_shell",
]
