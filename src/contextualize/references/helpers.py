from __future__ import annotations

import codecs
import os
import re
from pathlib import Path
from typing import Protocol, TYPE_CHECKING, runtime_checkable
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    from ..git.cache import GitTarget


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


_SPEC_OPTION_RE = re.compile(r'(filename|params|root|wrap)=(?:"([^"]*)"|([^"]*))')
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_URL_PREFIXES = ("http://", "https://")


def is_http_url(value: str) -> bool:
    return value.startswith(_URL_PREFIXES)


def parse_target_spec(spec: str) -> dict[str, str | None]:
    parts = spec.split("::")
    opts: dict[str, str | None] = {}
    target_parts: list[str] = []
    for part in parts:
        match = _SPEC_OPTION_RE.fullmatch(part)
        if match:
            opts[match.group(1)] = match.group(2) or match.group(3)
        else:
            target_parts.append(part)
    opts["target"] = "::".join(target_parts)
    return opts


def looks_like_windows_drive(spec: str) -> bool:
    return bool(_WINDOWS_DRIVE_RE.match(spec))


def split_spec_symbols(spec: str) -> tuple[str, list[str]]:
    if is_http_url(spec) or looks_like_windows_drive(spec):
        return spec, []
    from ..git.cache import parse_git_target

    if parse_git_target(spec):
        return spec, []
    return split_path_and_symbols(spec)


def parse_git_url_target(url: str) -> GitTarget | None:
    from ..git.cache import parse_git_target

    tgt = parse_git_target(url)
    if not tgt:
        return None
    if tgt.path is None and not tgt.repo_url.endswith(".git") and tgt.repo_url == url:
        return None
    return tgt


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


def _find_symbol_ranges(
    file_path: str, symbols: list[str], *, text: str | None = None
) -> dict[str, tuple[int, int]]:
    try:
        from ..render.map import find_symbol_ranges
    except Exception:
        return {}
    return find_symbol_ranges(file_path, symbols, text=text)


def warn_missing_symbols(path_label: str, symbols: list[str]) -> None:
    import sys

    print(
        f"Warning: symbol(s) not found in {path_label}: {', '.join(symbols)}",
        file=sys.stderr,
    )


def resolve_symbol_ranges(
    file_path: str,
    symbols: list[str] | None,
    *,
    text: str | None = None,
    ranges: list[tuple[int, int]] | None = None,
    warn_label: str | None = None,
    append_to_ranges: bool = False,
    keep_missing: bool = True,
    skip_on_missing: bool = False,
    warn_on_partial: bool = True,
) -> tuple[list[tuple[int, int]] | None, list[str] | None, bool]:
    if not symbols:
        return ranges, symbols, False

    match_map = _find_symbol_ranges(file_path, symbols, text=text)
    matched = [s for s in symbols if s in match_map]
    missing = [s for s in symbols if s not in match_map]

    if missing and (warn_on_partial or not matched):
        if warn_label:
            warn_missing_symbols(warn_label, missing)

    if not matched:
        if skip_on_missing and ranges is None:
            return ranges, symbols, True
        if keep_missing:
            return ranges, symbols, False
        return ranges, None, False

    sym_ranges = [match_map[s] for s in matched]
    if append_to_ranges and ranges:
        ranges = ranges + sym_ranges
    else:
        ranges = sym_ranges
    return ranges, matched, False


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
