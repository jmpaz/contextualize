"""Git reference implementations."""

import os
import sys
from dataclasses import dataclass, field

from ..render import process_text
from ..utils import count_tokens


@dataclass
class GitRevFileReference:
    """Reference to a file at a specific git revision."""

    repo_root: str
    rev: str
    rel_path: str
    format: str = "md"
    _label_style: str = "relative"
    include_token_count: bool = False
    token_target: str = "cl100k_base"
    _file_content: str | None = field(default=None, init=False)
    _original_file_content: str | None = field(default=None, init=False)
    ranges: list[tuple[int, int]] | None = None
    symbols: list[str] | None = None
    _output: str | None = field(default=None, init=False)

    @property
    def path(self) -> str:
        return os.path.join(self.repo_root, self.rel_path)

    @property
    def file_content(self) -> str:
        if self._file_content is None:
            self._load_content()
        return self._file_content or ""

    @property
    def original_file_content(self) -> str:
        if self._original_file_content is None:
            self._load_content()
        return self._original_file_content or ""

    @property
    def label(self) -> str:
        return self._get_label()

    def _get_label(self) -> str:
        if self._label_style == "relative":
            return self.rel_path
        if self._label_style == "name":
            return os.path.basename(self.rel_path)
        if self._label_style == "ext":
            return os.path.splitext(self.rel_path)[1]
        return self._label_style

    def read(self) -> str:
        """Read and return the file content at the revision."""
        return self.file_content

    def exists(self) -> bool:
        """Check if the file exists at the revision."""
        from ...git.rev import read_file_at_rev

        return read_file_at_rev(self.repo_root, self.rev, self.rel_path) is not None

    def token_count(self, encoding: str = "cl100k_base") -> int:
        """Count tokens in the file content."""
        return count_tokens(self.file_content, target=encoding)["count"]

    def _load_content(self) -> None:
        """Load the file content from the git revision."""
        from ...git.rev import read_file_at_rev

        text = read_file_at_rev(self.repo_root, self.rev, self.rel_path)
        if text is None:
            self._file_content = ""
            self._original_file_content = ""
        else:
            self._file_content = text
            self._original_file_content = text

    @property
    def output(self) -> str:
        if self._output is not None:
            return self._output

        from ...git.rev import read_file_at_rev

        text = read_file_at_rev(self.repo_root, self.rev, self.rel_path)
        if text is None:
            self._output = ""
            return self._output
        self._original_file_content = text
        self._file_content = text

        ranges = self.ranges
        symbols = [s for s in (self.symbols or []) if s]
        if symbols and ranges is None:
            try:
                from ..repomap import find_symbol_ranges

                match_map = find_symbol_ranges(
                    self.rel_path, symbols, text=self._file_content
                )
            except Exception:
                match_map = {}

            missing = [s for s in symbols if s not in match_map]
            if missing:
                print(
                    f"Warning: symbol(s) not found in {self.rel_path}@{self.rev}: {', '.join(missing)}",
                    file=sys.stderr,
                )
            matched = [s for s in symbols if s in match_map]
            if match_map and matched:
                ranges = [match_map[s] for s in matched]
                symbols = matched

        self._output = process_text(
            text,
            format=self.format,
            label=self._get_label(),
            rev=self.rev,
            token_target=self.token_target,
            include_token_count=self.include_token_count,
            ranges=ranges,
            symbols=symbols,
        )
        return self._output


@dataclass
class GitCacheReference:
    """Reference to a file in a cached git repository.

    This is used for remote git repositories that have been cloned
    to the local cache.
    """

    cache_dir: str
    rel_path: str
    format: str = "md"
    _label_style: str = "relative"
    include_token_count: bool = False
    token_target: str = "cl100k_base"
    label_suffix: str | None = None
    inject: bool = False
    depth: int = 5
    trace_collector: list | None = None
    symbols: list[str] | None = None
    _file_content: str = field(default="", init=False)
    _output: str = field(default="", init=False)

    def __post_init__(self) -> None:
        # Import FileReference here to avoid circular imports
        from .file import FileReference

        full_path = os.path.join(self.cache_dir, self.rel_path)
        self._file_ref = FileReference(
            full_path,
            format=self.format,
            label=self._label_style,
            label_suffix=self.label_suffix,
            include_token_count=self.include_token_count,
            token_target=self.token_target,
            inject=self.inject,
            depth=self.depth,
            trace_collector=self.trace_collector,
            symbols=self.symbols,
        )
        self._file_content = self._file_ref.file_content
        self._output = self._file_ref.output

    @property
    def path(self) -> str:
        return os.path.join(self.cache_dir, self.rel_path)

    @property
    def file_content(self) -> str:
        return self._file_content

    @property
    def output(self) -> str:
        return self._output

    @property
    def label(self) -> str:
        return self._file_ref.label

    def read(self) -> str:
        """Read and return the file content."""
        return self._file_content

    def exists(self) -> bool:
        """Check if the file exists in the cache."""
        return os.path.isfile(self.path)

    def token_count(self, encoding: str = "cl100k_base") -> int:
        """Count tokens in the file content."""
        return count_tokens(self._file_content, target=encoding)["count"]
