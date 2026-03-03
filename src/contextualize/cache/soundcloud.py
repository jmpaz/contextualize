from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .media import (
    get_cached_media_bytes as _get_cached_media_bytes,
    store_media_bytes as _store_media_bytes,
)

SOUNDCLOUD_CACHE_ROOT = Path(
    os.environ.get(
        "CONTEXTUALIZE_SOUNDCLOUD_CACHE",
        os.path.expanduser("~/.local/share/contextualize/cache/soundcloud/v1"),
    )
)
API_CACHE_ROOT = SOUNDCLOUD_CACHE_ROOT / "api"
RENDER_CACHE_ROOT = SOUNDCLOUD_CACHE_ROOT / "render"
MEDIA_CACHE_ROOT = SOUNDCLOUD_CACHE_ROOT / "media"
TOKEN_CACHE_ROOT = SOUNDCLOUD_CACHE_ROOT / "token"
DEFAULT_API_TTL = timedelta(hours=6)
CACHE_VERSION = 1


@dataclass
class SoundCloudCacheMetadata:
    identity: str
    cached_at: str
    size_bytes: int
    cache_version: int = CACHE_VERSION


@dataclass
class SoundCloudTokenMetadata:
    identity: str
    cached_at: str
    expires_at: str
    cache_version: int = CACHE_VERSION


def _cache_key(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _cache_paths(root: Path, identity: str, ext: str) -> tuple[Path, Path]:
    key = _cache_key(identity)
    content = root / f"{key}.{ext}"
    meta = root / f"{key}.meta.json"
    return content, meta


def _load_meta(path: Path) -> SoundCloudCacheMetadata | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return SoundCloudCacheMetadata(**payload)
    except TypeError:
        return None


def _is_expired(meta: SoundCloudCacheMetadata, ttl: timedelta | None) -> bool:
    if ttl is None:
        return False
    if ttl == timedelta(0):
        return True
    try:
        cached_at = datetime.fromisoformat(meta.cached_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    now = datetime.now(timezone.utc)
    return (now - cached_at) > ttl


def get_cached_api_json(identity: str, ttl: timedelta | None = None) -> Any | None:
    effective_ttl = DEFAULT_API_TTL if ttl is None else ttl
    content_path, meta_path = _cache_paths(API_CACHE_ROOT, identity, "json")
    if not content_path.exists():
        return None
    meta = _load_meta(meta_path)
    if meta is None or _is_expired(meta, effective_ttl):
        return None
    try:
        return json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def store_api_json(identity: str, payload: Any) -> None:
    try:
        API_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        content_path, meta_path = _cache_paths(API_CACHE_ROOT, identity, "json")
        text = json.dumps(payload, ensure_ascii=False)
        content_path.write_text(text, encoding="utf-8")
        meta = SoundCloudCacheMetadata(
            identity=identity,
            cached_at=datetime.now(timezone.utc).isoformat(),
            size_bytes=len(text.encode("utf-8")),
        )
        meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    except OSError:
        return


def get_cached_rendered(identity: str, ttl: timedelta | None = None) -> str | None:
    content_path, meta_path = _cache_paths(RENDER_CACHE_ROOT, identity, "txt")
    if not content_path.exists():
        return None
    meta = _load_meta(meta_path)
    if meta is None or _is_expired(meta, ttl):
        return None
    try:
        return content_path.read_text(encoding="utf-8")
    except OSError:
        return None


def store_rendered(identity: str, content: str) -> None:
    try:
        RENDER_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        content_path, meta_path = _cache_paths(RENDER_CACHE_ROOT, identity, "txt")
        content_path.write_text(content, encoding="utf-8")
        meta = SoundCloudCacheMetadata(
            identity=identity,
            cached_at=datetime.now(timezone.utc).isoformat(),
            size_bytes=len(content.encode("utf-8")),
        )
        meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    except OSError:
        return


def get_cached_media_bytes(identity: str) -> bytes | None:
    return _get_cached_media_bytes(MEDIA_CACHE_ROOT, identity)


def store_media_bytes(identity: str, content: bytes) -> None:
    _store_media_bytes(MEDIA_CACHE_ROOT, identity, content)


def _token_paths(identity: str) -> tuple[Path, Path]:
    return _cache_paths(TOKEN_CACHE_ROOT, identity, "json")


def _load_token_meta(path: Path) -> SoundCloudTokenMetadata | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return SoundCloudTokenMetadata(**payload)
    except TypeError:
        return None


def get_cached_client_token(identity: str) -> str | None:
    content_path, meta_path = _token_paths(identity)
    if not content_path.exists():
        return None
    meta = _load_token_meta(meta_path)
    if meta is None:
        return None
    try:
        expires_at = datetime.fromisoformat(meta.expires_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if expires_at <= datetime.now(timezone.utc):
        return None
    try:
        payload = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    token = payload.get("access_token")
    if isinstance(token, str) and token.strip():
        return token
    return None


def store_client_token(
    identity: str,
    *,
    access_token: str,
    expires_in_seconds: int,
) -> None:
    if not access_token:
        return
    try:
        TOKEN_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        content_path, meta_path = _token_paths(identity)
        content_path.write_text(
            json.dumps({"access_token": access_token}, indent=2),
            encoding="utf-8",
        )
        now = datetime.now(timezone.utc)
        max_age = max(60, int(expires_in_seconds) - 60)
        expires_at = now + timedelta(seconds=max_age)
        meta = SoundCloudTokenMetadata(
            identity=identity,
            cached_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
        )
        meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    except OSError:
        return
