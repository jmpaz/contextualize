import codecs
import os
import re
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import unquote, urlparse


@runtime_checkable
class Reference(Protocol):
    label: str

    def read(self) -> str: ...
    def exists(self) -> bool: ...
    def token_count(self, encoding: str = "cl100k_base") -> int: ...


def split_path_and_symbols(raw_path: str) -> tuple[str, list[str]]:
    if ":" not in raw_path:
        return raw_path, []
    base, _, suffix = raw_path.partition(":")
    symbols = [part.strip() for part in suffix.split(",") if part.strip()]
    return base or raw_path, symbols


def is_utf8_file(path: str, sample_size: int = 4096) -> bool:
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


MARKITDOWN_PREFERRED_EXTENSIONS: frozenset[str] = frozenset(
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

DISALLOWED_EXTENSIONS: frozenset[str] = frozenset({".zip"})

TEXTUAL_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "application/javascript",
        "application/xml",
        "application/x-yaml",
        "application/yaml",
    }
)

DISALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {"application/zip", "application/x-zip-compressed"}
)


_CD_FILENAME_RE = re.compile(
    r"filename\*?=(?:UTF-8''|\"|')?(?P<name>[^\"';]+)", flags=re.IGNORECASE
)


def strip_content_type(value: str) -> str:
    return value.split(";", 1)[0].strip().lower()


def content_disposition_filename(value: str) -> str | None:
    if not value:
        return None
    match = _CD_FILENAME_RE.search(value)
    if not match:
        return None
    name = unquote(match.group("name").strip())
    return os.path.basename(name) if name else None


def infer_url_suffix(url: str, headers: dict[str, str]) -> str | None:
    path = unquote(urlparse(url).path or "")
    suffix = Path(path).suffix.lower()
    if suffix:
        return suffix
    filename = content_disposition_filename(headers.get("Content-Disposition", ""))
    if filename:
        cd_suffix = Path(filename).suffix.lower()
        if cd_suffix:
            return cd_suffix
    return None


def looks_like_text_content_type(content_type: str) -> bool:
    if not content_type:
        return False
    if content_type.startswith("text/"):
        return True
    if content_type in TEXTUAL_CONTENT_TYPES:
        return True
    if "json" in content_type:
        return True
    if content_type.endswith("+json"):
        return True
    if content_type.endswith("+xml"):
        return True
    return False


def remove_ansi(text: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)
