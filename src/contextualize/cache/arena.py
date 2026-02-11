from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARENA_CACHE_ROOT = Path(
    os.environ.get(
        "CONTEXTUALIZE_ARENA_CACHE",
        os.path.expanduser("~/.local/share/contextualize/cache/arena/v1"),
    )
)
BLOCK_CACHE_ROOT = ARENA_CACHE_ROOT / "blocks"
COMMENTS_CACHE_ROOT = ARENA_CACHE_ROOT / "comments"
MEDIA_CACHE_ROOT = ARENA_CACHE_ROOT / "media"
DEFAULT_TTL = timedelta(days=7)
CACHE_VERSION = 1


@dataclass
class ArenaCacheMetadata:
    slug: str
    cached_at: str
    block_count: int
    size_bytes: int
    cache_version: int = CACHE_VERSION


def _get_cache_key(slug: str) -> str:
    return hashlib.sha256(slug.encode("utf-8")).hexdigest()


def _get_cache_paths(slug: str) -> tuple[Path, Path]:
    key = _get_cache_key(slug)
    content_path = ARENA_CACHE_ROOT / f"{key}.content"
    meta_path = ARENA_CACHE_ROOT / f"{key}.meta.json"
    return content_path, meta_path


def _load_metadata(meta_path: Path) -> ArenaCacheMetadata | None:
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return ArenaCacheMetadata(**data)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def _is_expired(metadata: ArenaCacheMetadata, ttl: timedelta) -> bool:
    if ttl == timedelta(0):
        return True
    try:
        cached_at = datetime.fromisoformat(metadata.cached_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (now - cached_at) > ttl
    except (ValueError, AttributeError):
        return True


def get_cached_channel(
    slug: str,
    ttl: timedelta | None = None,
) -> str | None:
    if ttl is None:
        ttl = DEFAULT_TTL

    content_path, meta_path = _get_cache_paths(slug)

    if not content_path.exists():
        return None

    metadata = _load_metadata(meta_path)
    if metadata is None:
        return None

    if _is_expired(metadata, ttl):
        return None

    try:
        return content_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def store_channel(
    slug: str,
    content: str,
    block_count: int = 0,
) -> None:
    ARENA_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    content_path, meta_path = _get_cache_paths(slug)

    content_path.write_text(content, encoding="utf-8")

    metadata = ArenaCacheMetadata(
        slug=slug,
        cached_at=datetime.now(timezone.utc).isoformat(),
        block_count=block_count,
        size_bytes=len(content.encode("utf-8")),
        cache_version=CACHE_VERSION,
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)


def _block_cache_key(
    block_id: int, updated_at: str, render_variant: str | None = None
) -> str:
    raw = f"{block_id}:{updated_at}"
    if render_variant:
        raw = f"{raw}:{render_variant}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_block_render(
    block_id: int, updated_at: str, render_variant: str | None = None
) -> str | None:
    key = _block_cache_key(block_id, updated_at, render_variant)
    path = BLOCK_CACHE_ROOT / f"{key}.txt"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def store_block_render(
    block_id: int,
    updated_at: str,
    rendered: str,
    render_variant: str | None = None,
) -> None:
    BLOCK_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    key = _block_cache_key(block_id, updated_at, render_variant)
    path = BLOCK_CACHE_ROOT / f"{key}.txt"
    path.write_text(rendered, encoding="utf-8")


def get_cached_block_comments(
    block_id: int,
    ttl: timedelta | None = None,
) -> str | None:
    if ttl is None:
        ttl = DEFAULT_TTL
    if ttl == timedelta(0):
        return None

    path = COMMENTS_CACHE_ROOT / f"{block_id}.txt"
    if not path.exists():
        return None

    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        if (now - modified) > ttl:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def store_block_comments(block_id: int, rendered: str) -> None:
    COMMENTS_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = COMMENTS_CACHE_ROOT / f"{block_id}.txt"
    path.write_text(rendered, encoding="utf-8")


def _media_cache_key(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def get_cached_media_bytes(identity: str) -> bytes | None:
    key = _media_cache_key(identity)
    path = MEDIA_CACHE_ROOT / f"{key}.bin"
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def store_media_bytes(identity: str, content: bytes) -> None:
    if not content:
        return
    MEDIA_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    key = _media_cache_key(identity)
    path = MEDIA_CACHE_ROOT / f"{key}.bin"
    path.write_bytes(content)
