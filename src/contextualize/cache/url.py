from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

URL_CACHE_ROOT = Path(
    os.environ.get(
        "CONTEXTUALIZE_URL_CACHE",
        os.path.expanduser("~/.local/share/contextualize/cache/url/v1"),
    )
)
DEFAULT_TTL = timedelta(days=7)
CACHE_VERSION = 1


@dataclass
class CacheMetadata:
    url: str
    normalized_url: str
    fetched_at: str
    content_type: str | None
    size_bytes: int
    cache_version: int = CACHE_VERSION


def parse_duration(s: str) -> timedelta:
    if not s or s == "0":
        return timedelta(0)

    s = s.strip().lower()
    match = re.match(r"^(\d+)\s*(w|d|h|m|s)?$", s)
    if not match:
        raise ValueError(f"Invalid duration format: {s!r}")

    value = int(match.group(1))
    unit = match.group(2) or "s"

    multipliers = {
        "w": timedelta(weeks=1),
        "d": timedelta(days=1),
        "h": timedelta(hours=1),
        "m": timedelta(minutes=1),
        "s": timedelta(seconds=1),
    }

    return value * multipliers[unit]


def normalize_url(url: str) -> str:
    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    path = parsed.path
    if path and path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query_params = parse_qsl(parsed.query, keep_blank_values=True)
    sorted_params = sorted(query_params, key=lambda x: x[0])
    query = urlencode(sorted_params)

    normalized = urlunparse((scheme, netloc, path, "", query, ""))
    return normalized


def get_cache_key(url: str) -> str:
    normalized = normalize_url(url)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _get_cache_paths(url: str) -> tuple[Path, Path]:
    key = get_cache_key(url)
    content_path = URL_CACHE_ROOT / f"{key}.content"
    meta_path = URL_CACHE_ROOT / f"{key}.meta.json"
    return content_path, meta_path


def _load_metadata(meta_path: Path) -> CacheMetadata | None:
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return CacheMetadata(**data)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def is_expired(metadata: CacheMetadata, ttl: timedelta) -> bool:
    if ttl == timedelta(0):
        return True
    try:
        fetched_at = datetime.fromisoformat(metadata.fetched_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (now - fetched_at) > ttl
    except (ValueError, AttributeError):
        return True


def get_cached(url: str, ttl: timedelta | None = None) -> str | None:
    if ttl is None:
        ttl = DEFAULT_TTL

    content_path, meta_path = _get_cache_paths(url)

    if not content_path.exists():
        return None

    metadata = _load_metadata(meta_path)
    if metadata is None:
        return None

    if is_expired(metadata, ttl):
        return None

    try:
        return content_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def store_cached(
    url: str,
    content: str,
    content_type: str | None = None,
) -> None:
    URL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    content_path, meta_path = _get_cache_paths(url)

    content_path.write_text(content, encoding="utf-8")

    metadata = CacheMetadata(
        url=url,
        normalized_url=normalize_url(url),
        fetched_at=datetime.now(timezone.utc).isoformat(),
        content_type=content_type,
        size_bytes=len(content.encode("utf-8")),
        cache_version=CACHE_VERSION,
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)
