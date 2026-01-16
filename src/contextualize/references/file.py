"""FileReference - Local filesystem file references."""

import os
from pathlib import Path

from ..render.text import process_text
from ..utils import count_tokens
from .helpers import (
    DISALLOWED_EXTENSIONS,
    MARKITDOWN_PREFERRED_EXTENSIONS,
    is_utf8_file,
    resolve_symbol_ranges,
)


class FileReference:
    """A reference to a local filesystem file."""

    def __init__(
        self,
        path,
        range=None,
        ranges=None,
        format="md",
        label="relative",
        clean_contents=False,
        *,
        label_suffix: str | None = None,
        include_token_count=False,
        token_target="cl100k_base",
        inject=False,
        depth=5,
        trace_collector=None,
        symbols=None,
    ):
        self.range = range
        self.ranges = ranges
        if self.range and not self.ranges:
            self.ranges = [self.range]
        self.symbols = [s for s in (symbols or []) if s]

        self.path = path
        self.format = format
        self._label_style = label
        self.label_suffix = label_suffix
        self.clean_contents = clean_contents
        self.include_token_count = include_token_count
        self.token_target = token_target
        self.inject = inject
        self.depth = depth
        self.trace_collector = trace_collector
        self.file_content = self.original_file_content = ""
        self.output = self._get_contents()

    @property
    def label(self) -> str:
        """Return the label for this reference."""
        return self.get_label()

    def read(self) -> str:
        """Read and return the raw file content."""
        return self.original_file_content

    def exists(self) -> bool:
        """Check if the file exists."""
        return os.path.isfile(self.path)

    def token_count(self, encoding: str = "cl100k_base") -> int:
        """Count tokens in the file content."""
        return count_tokens(self.original_file_content, target=encoding)["count"]

    def _get_contents(self):
        """Read and process the file contents."""
        suffix = Path(self.path).suffix.lower()
        if suffix in DISALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {self.path}")
        prefer_markitdown = suffix in MARKITDOWN_PREFERRED_EXTENSIONS
        if not prefer_markitdown:
            try:
                with open(self.path, "r", encoding="utf-8") as file:
                    self.file_content = self.original_file_content = file.read()
            except UnicodeDecodeError:
                if self.format == "raw":
                    return ""
                prefer_markitdown = True
            except Exception as e:
                print(f"Error reading file {self.path}: {str(e)}")
                return ""

        if prefer_markitdown:
            from ..render.markitdown import (
                MarkItDownConversionError,
                convert_path_to_markdown,
            )

            try:
                result = convert_path_to_markdown(self.path)
                self.file_content = self.original_file_content = result.markdown
            except MarkItDownConversionError as e:
                print(f"Error converting file {self.path}: {e}")
                return ""
        if self.inject:
            from ..render.inject import inject_content_in_text

            self.file_content = inject_content_in_text(
                self.file_content, self.depth, self.trace_collector, self.path
            )

        ranges = self.ranges
        if self.symbols and ranges is None:
            ranges, symbols, _ = resolve_symbol_ranges(
                self.path,
                self.symbols,
                text=self.file_content,
                warn_label=self.path,
            )
            self.symbols = symbols or []

        return process_text(
            self.file_content,
            self.clean_contents,
            ranges=ranges,
            format=self.format,
            label=self.get_label(),
            label_suffix=self.label_suffix,
            token_target=self.token_target,
            include_token_count=self.include_token_count,
            symbols=self.symbols,
        )

    def get_label(self):
        """Compute the label based on label style."""
        if self._label_style == "relative":
            from ..git.cache import CACHE_ROOT

            cache_root = os.path.join(CACHE_ROOT, "")
            if self.path.startswith(cache_root):
                rel = os.path.relpath(self.path, CACHE_ROOT)
                parts = rel.split(os.sep)
                if parts and parts[0] in ("github", "ext"):
                    return os.path.join(*parts[1:])
                return rel
            return self.path
        elif self._label_style == "name":
            return os.path.basename(self.path)
        elif self._label_style == "ext":
            return os.path.splitext(self.path)[1]
        else:
            return self._label_style

    # Legacy alias
    def get_contents(self):
        """Legacy method - returns cached output."""
        return self.output
