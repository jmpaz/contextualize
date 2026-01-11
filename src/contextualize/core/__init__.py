"""
Core content-processing utilities.

This package provides the low-level building blocks for the contextualize CLI.

Public API:
    resolve(target, **options) -> list[Reference]
        Resolve a target string into references.

    format_content(content, fmt, label, **options) -> str
        Format content using a format handler.

    count_tokens(text, target) -> dict
        Count tokens in text.

    process_text(text, **options) -> str
        Process and format text with various options.

Reference Types:
    Reference - Protocol that all reference types implement
    FileReference - Local filesystem paths
    URLReference - HTTP/HTTPS URLs
    GitRevFileReference - Content from git revisions
    GitCacheReference - Content from cached remote repos
    ClipboardReference - Clipboard captures
    ShellReference - Command output (also CommandReference)
"""

from .render import process_text
from .references import (
    # Protocol
    Reference,
    # Factory
    resolve,
    # Reference types
    FileReference,
    URLReference,
    GitRevFileReference,
    GitCacheReference,
    ClipboardReference,
    ShellReference,
    CommandReference,
    # Helper functions
    concat_refs,
    create_file_references,
    create_command_references,
    split_path_and_symbols,
)
from .formats import format_content, register_format, FORMATS
from .utils import count_tokens

__all__ = [
    # Rendering
    "process_text",
    "format_content",
    "register_format",
    "FORMATS",
    # References
    "Reference",
    "resolve",
    "FileReference",
    "URLReference",
    "GitRevFileReference",
    "GitCacheReference",
    "ClipboardReference",
    "ShellReference",
    "CommandReference",
    "concat_refs",
    "create_file_references",
    "create_command_references",
    "split_path_and_symbols",
    # Token counting
    "count_tokens",
]
