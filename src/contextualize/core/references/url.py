"""URL reference implementation."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

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

_TEXTUAL_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "application/javascript",
        "application/xml",
        "application/x-yaml",
        "application/yaml",
    }
)

_DISALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {"application/zip", "application/x-zip-compressed"}
)

_CD_FILENAME_RE = re.compile(
    r"filename\*?=(?:UTF-8''|\"|')?(?P<name>[^\"';]+)", flags=re.IGNORECASE
)


def _strip_content_type(value: str) -> str:
    return value.split(";", 1)[0].strip().lower()


def _content_disposition_filename(value: str) -> str | None:
    if not value:
        return None
    match = _CD_FILENAME_RE.search(value)
    if not match:
        return None
    name = unquote(match.group("name").strip())
    return os.path.basename(name) if name else None


def _infer_url_suffix(url: str, headers: dict[str, str]) -> str | None:
    path = unquote(urlparse(url).path or "")
    suffix = Path(path).suffix.lower()
    if suffix:
        return suffix
    filename = _content_disposition_filename(headers.get("Content-Disposition", ""))
    if filename:
        cd_suffix = Path(filename).suffix.lower()
        if cd_suffix:
            return cd_suffix
    return None


def _looks_like_text_content_type(content_type: str) -> bool:
    if not content_type:
        return False
    if content_type.startswith("text/"):
        return True
    if content_type in _TEXTUAL_CONTENT_TYPES:
        return True
    if "json" in content_type:
        return True
    if content_type.endswith("+json"):
        return True
    if content_type.endswith("+xml"):
        return True
    return False


@dataclass
class URLReference:
    """Reference to content at a URL."""

    url: str
    format: str = "md"
    _label_style: str = "relative"
    token_target: str = "cl100k_base"
    include_token_count: bool = False
    label_suffix: str | None = None
    inject: bool = False
    depth: int = 5
    trace_collector: list = None
    _file_content: str = field(default="", init=False)
    _original_file_content: str = field(default="", init=False)
    _output: str = field(default="", init=False)

    def __post_init__(self) -> None:
        # Handle dataclass field name aliasing
        if hasattr(self, "label") and not hasattr(self, "_label_style"):
            self._label_style = self.label
        self._output = self._get_contents()

    @property
    def path(self) -> str:
        return self.url

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
        """Read and return the raw content."""
        return self._file_content

    def exists(self) -> bool:
        """Check if the URL is accessible."""
        import requests

        try:
            r = requests.head(self.url, timeout=10, headers={"User-Agent": "contextualize"})
            return r.status_code < 400
        except Exception:
            return False

    def token_count(self, encoding: str = "cl100k_base") -> int:
        """Count tokens in the content."""
        return count_tokens(self._file_content, target=encoding)["count"]

    def _get_label(self) -> str:
        path = urlparse(self.url).path
        if self._label_style == "relative":
            return self.url
        if self._label_style == "name":
            return os.path.basename(path)
        if self._label_style == "ext":
            return os.path.splitext(path)[1]
        return self._label_style

    def _get_contents(self) -> str:
        import json

        import requests

        r = requests.get(self.url, timeout=30, headers={"User-Agent": "contextualize"})
        r.raise_for_status()
        content_type = _strip_content_type(r.headers.get("Content-Type", ""))
        suffix = _infer_url_suffix(self.url, dict(r.headers))
        if (suffix and suffix in _DISALLOWED_EXTENSIONS) or (
            content_type in _DISALLOWED_CONTENT_TYPES
        ):
            raise ValueError(f"Unsupported file type: {self.url}")

        data = r.content
        prefer_markitdown = bool(suffix and suffix in _MARKITDOWN_PREFERRED_EXTENSIONS)
        is_text = _looks_like_text_content_type(content_type)

        if prefer_markitdown:
            from ..markitdown_adapter import convert_response_to_markdown

            text = convert_response_to_markdown(r).markdown
            self._original_file_content = text
        elif is_text:
            text = r.text
            self._original_file_content = text
            if "json" in content_type:
                try:
                    text = json.dumps(r.json(), indent=2)
                except Exception:
                    pass
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                from ..markitdown_adapter import convert_response_to_markdown

                text = convert_response_to_markdown(r).markdown
            self._original_file_content = text
        if self.inject:
            from ..links import inject_content_in_text

            text = inject_content_in_text(
                text, self.depth, self.trace_collector, self.url
            )
        self._file_content = text
        return process_text(
            text,
            format=self.format,
            label=self._get_label(),
            label_suffix=self.label_suffix,
            token_target=self.token_target,
            include_token_count=self.include_token_count,
        )
