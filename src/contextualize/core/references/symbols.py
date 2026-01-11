"""Symbol extraction utilities for references."""

from pathlib import Path


def split_path_and_symbols(raw_path: str) -> tuple[str, list[str]]:
    """
    Split a path of the form "path:sym1,sym2" into the base path and symbol list.

    Examples:
        "src/file.py:MyClass" -> ("src/file.py", ["MyClass"])
        "src/file.py:func1,func2" -> ("src/file.py", ["func1", "func2"])
        "src/file.py" -> ("src/file.py", [])
    """
    if ":" not in raw_path:
        return raw_path, []

    # Handle Windows paths like C:\path
    if len(raw_path) > 1 and raw_path[1] == ":" and raw_path[0].isalpha():
        # This looks like a Windows absolute path
        if len(raw_path) > 2 and ":" in raw_path[2:]:
            # There's another colon after the drive letter
            idx = raw_path.index(":", 2)
            base = raw_path[:idx]
            suffix = raw_path[idx + 1 :]
            symbols = [part.strip() for part in suffix.split(",") if part.strip()]
            return base, symbols
        return raw_path, []

    base, _, suffix = raw_path.partition(":")
    symbols = [part.strip() for part in suffix.split(",") if part.strip()]
    return base or raw_path, symbols


def extract_symbols_from_text(
    text: str, symbols: list[str], file_path: str | None = None
) -> dict[str, tuple[int, int]]:
    """
    Extract symbol ranges from text content.

    Args:
        text: The source code text
        symbols: List of symbol names to find
        file_path: Optional file path for language detection

    Returns:
        Dictionary mapping symbol names to (start_line, end_line) tuples
    """
    try:
        from ..repomap import find_symbol_ranges

        return find_symbol_ranges(file_path or "", symbols, text=text)
    except Exception:
        return {}


def get_language_for_path(path: str) -> str | None:
    """
    Determine the programming language based on file extension.

    Returns the tree-sitter language name or None if unknown.
    """
    extension_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "c_sharp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".lua": "lua",
        ".r": "r",
        ".R": "r",
        ".jl": "julia",
    }
    suffix = Path(path).suffix.lower()
    return extension_map.get(suffix)
