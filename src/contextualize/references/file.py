"""FileReference - Local filesystem file references."""

import os
from datetime import timedelta
from pathlib import Path

from ..render.text import process_text
from ..utils import count_tokens
from .helpers import (
    DISALLOWED_EXTENSIONS,
    MARKITDOWN_PREFERRED_EXTENSIONS,
    resolve_symbol_ranges,
)
from .audio_transcription import is_media_suffix, is_video_suffix, transcribe_media_file
from .video_context import render_video_file
from ..render.markitdown import IMAGE_SUFFIXES as _IMAGE_SUFFIXES


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
        use_cache: bool = True,
        cache_ttl: timedelta | None = None,
        refresh_cache: bool = False,
        plugin_overrides: dict | None = None,
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
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.refresh_cache = refresh_cache
        self.plugin_overrides = plugin_overrides
        self.cache_miss = False
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
        if suffix in _IMAGE_SUFFIXES or is_video_suffix(suffix):
            from ..runtime import get_skip_media

            if get_skip_media():
                return ""
        if is_media_suffix(suffix):
            try:
                from ..references.audio_transcription import CacheMissError

                media_kwargs = {
                    "use_cache": self.use_cache,
                    "refresh_cache": self.refresh_cache,
                }
                if self.plugin_overrides is not None:
                    media_kwargs["plugin_overrides"] = self.plugin_overrides
                if is_video_suffix(suffix):
                    text = render_video_file(self.path, **media_kwargs)
                else:
                    text = transcribe_media_file(self.path, **media_kwargs)
            except CacheMissError:
                self.cache_miss = True
                return ""
            except Exception as e:
                raise ValueError(
                    f"Media transcription failed for {self.path}: {e}"
                ) from e
            self.file_content = self.original_file_content = text
            if self.inject:
                from ..render.inject import inject_content_in_text

                self.file_content = inject_content_in_text(
                    self.file_content,
                    self.depth,
                    self.trace_collector,
                    self.path,
                    use_cache=self.use_cache,
                    cache_ttl=self.cache_ttl,
                    refresh_cache=self.refresh_cache,
                    plugin_overrides=self.plugin_overrides,
                )
            return process_text(
                self.file_content,
                self.clean_contents,
                ranges=self.ranges,
                format=self.format,
                label=self.get_label(),
                label_suffix=self.label_suffix,
                token_target=self.token_target,
                include_token_count=self.include_token_count,
                symbols=self.symbols,
            )
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
                self.file_content,
                self.depth,
                self.trace_collector,
                self.path,
                use_cache=self.use_cache,
                cache_ttl=self.cache_ttl,
                refresh_cache=self.refresh_cache,
                plugin_overrides=self.plugin_overrides,
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
            from ..git.target import CACHE_ROOT

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


class FileExistenceReference:
    def __init__(
        self,
        path,
        *,
        format="md",
        label="relative",
        label_suffix: str | None = None,
        include_token_count=False,
        token_target="cl100k_base",
    ):
        self.path = path
        self.format = format
        self._label_style = label
        self.label_suffix = label_suffix
        self.include_token_count = include_token_count
        self.token_target = token_target
        self.file_content = self.original_file_content = self._content()
        self.output = process_text(
            self.file_content,
            False,
            format=self.format,
            label=self.get_label(),
            label_suffix=self.label_suffix,
            token_target=self.token_target,
            include_token_count=self.include_token_count,
        )

    @property
    def label(self) -> str:
        return self.get_label()

    def read(self) -> str:
        return self.original_file_content

    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def token_count(self, encoding: str = "cl100k_base") -> int:
        return count_tokens(self.original_file_content, target=encoding)["count"]

    def _content(self) -> str:
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return "[binary file exists]"
        return f"[binary file exists: {size} bytes]"

    def get_label(self):
        if self._label_style == "relative":
            from ..git.target import CACHE_ROOT

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
