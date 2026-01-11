"""File reference implementation."""

import codecs
import os
import sys
from pathlib import Path
from typing import Any

from ..render import process_text
from ..utils import count_tokens


_MARKITDOWN_PREFERRED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".pptx",
        ".xls",
        ".xlsx",
        ".csv",
        ".epub",
        ".msg",
        ".jpg",
        ".jpeg",
        ".png",
        ".wav",
        ".mp3",
        ".m4a",
        ".mp4",
    }
)

_DISALLOWED_EXTENSIONS: frozenset[str] = frozenset({".zip"})


def _is_utf8_file(path: str, sample_size: int = 4096) -> bool:
    """Check if a file is valid UTF-8."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_size)
    except OSError:
        return False

    if not chunk:
        return True

    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        decoder.decode(chunk, final=False)
    except UnicodeDecodeError:
        return False

    return True


class FileReference:
    """Reference to a local file."""

    def __init__(
        self,
        path: str,
        range: tuple[int, int] | None = None,
        ranges: list[tuple[int, int]] | None = None,
        format: str = "md",
        label: str = "relative",
        clean_contents: bool = False,
        *,
        label_suffix: str | None = None,
        include_token_count: bool = False,
        token_target: str = "cl100k_base",
        inject: bool = False,
        depth: int = 5,
        trace_collector: list | None = None,
        symbols: list[str] | None = None,
    ):
        self.range = range
        self.ranges = ranges
        if self.range and not self.ranges:
            self.ranges = [self.range]
        self.symbols = [s for s in (symbols or []) if s]

        self._path = path
        self.format = format
        self._label_style = label
        self.label_suffix = label_suffix
        self.clean_contents = clean_contents
        self.include_token_count = include_token_count
        self.token_target = token_target
        self.inject = inject
        self.depth = depth
        self.trace_collector = trace_collector
        self._file_content = self._original_file_content = ""
        self._output = self._get_contents()

    @property
    def path(self) -> str:
        return self._path

    @property
    def file_content(self) -> str:
        return self._file_content

    @property
    def original_file_content(self) -> str:
        return self._original_file_content

    @property
    def output(self) -> str:
        return self._output

    @property
    def label(self) -> str:
        return self._get_label()

    def read(self) -> str:
        """Read and return the raw file content."""
        return self._file_content

    def exists(self) -> bool:
        """Check if the file exists."""
        return os.path.isfile(self._path)

    def token_count(self, encoding: str = "cl100k_base") -> int:
        """Count tokens in the file content."""
        return count_tokens(self._file_content, target=encoding)["count"]

    def _get_contents(self) -> str:
        suffix = Path(self._path).suffix.lower()
        if suffix in _DISALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {self._path}")
        prefer_markitdown = suffix in _MARKITDOWN_PREFERRED_EXTENSIONS
        if not prefer_markitdown:
            try:
                with open(self._path, "r", encoding="utf-8") as file:
                    self._file_content = self._original_file_content = file.read()
            except UnicodeDecodeError:
                prefer_markitdown = True
            except Exception as e:
                print(f"Error reading file {self._path}: {str(e)}")
                return ""

        if prefer_markitdown:
            from ..markitdown_adapter import convert_path_to_markdown

            result = convert_path_to_markdown(self._path)
            self._file_content = self._original_file_content = result.markdown
        if self.inject:
            from ..links import inject_content_in_text

            self._file_content = inject_content_in_text(
                self._file_content, self.depth, self.trace_collector, self._path
            )

        ranges = self.ranges
        if self.symbols and ranges is None:
            try:
                from ..repomap import find_symbol_ranges

                match_map = find_symbol_ranges(
                    self._path, self.symbols, text=self._file_content
                )
            except Exception:
                match_map = {}

            missing = [s for s in self.symbols if s not in match_map]
            if missing:
                print(
                    f"Warning: symbol(s) not found in {self._path}: {', '.join(missing)}",
                    file=sys.stderr,
                )
            if match_map:
                matched = [s for s in self.symbols if s in match_map]
                ranges = [match_map[s] for s in matched]
                self.symbols = matched

        return process_text(
            self._file_content,
            self.clean_contents,
            ranges=ranges,
            format=self.format,
            label=self._get_label(),
            label_suffix=self.label_suffix,
            token_target=self.token_target,
            include_token_count=self.include_token_count,
            symbols=self.symbols,
        )

    def _get_label(self) -> str:
        if self._label_style == "relative":
            from ...git.cache import CACHE_ROOT

            cache_root = os.path.join(CACHE_ROOT, "")
            if self._path.startswith(cache_root):
                rel = os.path.relpath(self._path, CACHE_ROOT)
                parts = rel.split(os.sep)
                if parts and parts[0] in ("github", "ext"):
                    return os.path.join(*parts[1:])
                return rel
            return self._path
        elif self._label_style == "name":
            return os.path.basename(self._path)
        elif self._label_style == "ext":
            return os.path.splitext(self._path)[1]
        else:
            return self._label_style
