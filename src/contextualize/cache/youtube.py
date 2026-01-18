from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

YOUTUBE_CACHE_ROOT = Path(
    os.environ.get(
        "CONTEXTUALIZE_YOUTUBE_CACHE",
        os.path.expanduser("~/.local/share/contextualize/cache/youtube/v1"),
    )
)
DEFAULT_TTL = timedelta(days=30)
CACHE_VERSION = 1


@dataclass
class YouTubeCacheMetadata:
    video_id: str
    cached_at: str
    source: str
    size_bytes: int
    cache_version: int = CACHE_VERSION


def _get_cache_key(video_id: str) -> str:
    return hashlib.sha256(video_id.encode("utf-8")).hexdigest()


def _get_cache_paths(video_id: str) -> tuple[Path, Path]:
    key = _get_cache_key(video_id)
    content_path = YOUTUBE_CACHE_ROOT / f"{key}.content"
    meta_path = YOUTUBE_CACHE_ROOT / f"{key}.meta.json"
    return content_path, meta_path


def _load_metadata(meta_path: Path) -> YouTubeCacheMetadata | None:
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return YouTubeCacheMetadata(**data)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def _is_expired(metadata: YouTubeCacheMetadata, ttl: timedelta) -> bool:
    if ttl == timedelta(0):
        return True
    try:
        cached_at = datetime.fromisoformat(metadata.cached_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (now - cached_at) > ttl
    except (ValueError, AttributeError):
        return True


def get_cached_transcript(
    video_id: str,
    ttl: timedelta | None = None,
    whisper_available: bool = False,
) -> str | None:
    if ttl is None:
        ttl = DEFAULT_TTL

    content_path, meta_path = _get_cache_paths(video_id)

    if not content_path.exists():
        return None

    metadata = _load_metadata(meta_path)
    if metadata is None:
        return None

    if _is_expired(metadata, ttl):
        return None

    if whisper_available and metadata.source == "captions":
        return None

    try:
        return content_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def store_transcript(
    video_id: str,
    content: str,
    source: str = "unknown",
) -> None:
    YOUTUBE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    content_path, meta_path = _get_cache_paths(video_id)

    content_path.write_text(content, encoding="utf-8")

    metadata = YouTubeCacheMetadata(
        video_id=video_id,
        cached_at=datetime.now(timezone.utc).isoformat(),
        source=source,
        size_bytes=len(content.encode("utf-8")),
        cache_version=CACHE_VERSION,
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)
