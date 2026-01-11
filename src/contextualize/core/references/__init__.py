"""
Reference types and resolution for content sources.

This package provides the Reference protocol and implementations for
various content sources: local files, URLs, git repositories, clipboard,
and shell commands.

Public API:
    resolve(target, **options) -> list[Reference]
        Resolve a target string into references.

    Reference
        Protocol that all reference types implement.

Reference Types:
    FileReference - Local filesystem paths
    URLReference - HTTP/HTTPS URLs
    GitRevFileReference - Content from git revisions
    GitCacheReference - Content from cached remote repos
    ClipboardReference - Clipboard captures
    ShellReference - Command output (also exported as CommandReference)
"""

from .protocol import Reference
from .file import FileReference, _is_utf8_file
from .url import URLReference
from .git import GitRevFileReference, GitCacheReference
from .clipboard import ClipboardReference, create_clipboard_reference, get_clipboard_content
from .shell import ShellReference, CommandReference, create_command_references, remove_ansi
from .symbols import split_path_and_symbols, extract_symbols_from_text, get_language_for_path
from .resolve import resolve
from .factory import create_file_references

# Re-export legacy functions for backwards compatibility
from .file import _MARKITDOWN_PREFERRED_EXTENSIONS, _DISALLOWED_EXTENSIONS

__all__ = [
    # Protocol
    "Reference",
    # Factory
    "resolve",
    # Reference types
    "FileReference",
    "URLReference",
    "GitRevFileReference",
    "GitCacheReference",
    "ClipboardReference",
    "ShellReference",
    "CommandReference",  # Alias for ShellReference
    # Factory functions
    "create_file_references",
    "create_clipboard_reference",
    "create_command_references",
    # Utilities
    "split_path_and_symbols",
    "extract_symbols_from_text",
    "get_language_for_path",
    "get_clipboard_content",
    "remove_ansi",
    # Internal utilities (for backwards compatibility)
    "_is_utf8_file",
    "_MARKITDOWN_PREFERRED_EXTENSIONS",
    "_DISALLOWED_EXTENSIONS",
]


def concat_refs(refs: list) -> str:
    """Concatenate references into a single string."""
    return "\n\n".join(ref.output for ref in refs)
