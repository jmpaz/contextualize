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

ARENA_CACHE_ROOT = Path(
    os.environ.get(
        "CONTEXTUALIZE_ARENA_CACHE",
        os.path.expanduser("~/.local/share/contextualize/cache/arena/v1"),
    )
)
BLOCK_CACHE_ROOT = ARENA_CACHE_ROOT / "blocks"
COMMENTS_CACHE_ROOT = ARENA_CACHE_ROOT / "comments"
MEDIA_CACHE_ROOT = ARENA_CACHE_ROOT / "media"
TOKEN_CACHE_ROOT = ARENA_CACHE_ROOT / "token"
DEFAULT_TTL = timedelta(days=7)
CACHE_VERSION = 1


@dataclass
class ArenaCacheMetadata:
    slug: str
    cached_at: str
    block_count: int
    size_bytes: int
    cache_version: int = CACHE_VERSION


@dataclass
class ArenaTokenMetadata:
    identity: str
    cached_at: str
    expires_at: str
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


def get_cached_media_bytes(identity: str) -> bytes | None:
    return _get_cached_media_bytes(MEDIA_CACHE_ROOT, identity)


def store_media_bytes(identity: str, content: bytes) -> None:
    _store_media_bytes(MEDIA_CACHE_ROOT, identity, content)


def _token_paths(identity: str) -> tuple[Path, Path]:
    key = _get_cache_key(identity)
    return TOKEN_CACHE_ROOT / f"{key}.json", TOKEN_CACHE_ROOT / f"{key}.meta.json"


def _load_token_meta(meta_path: Path) -> ArenaTokenMetadata | None:
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return ArenaTokenMetadata(**payload)
    except TypeError:
        return None


def _secure_file_permissions(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        return


def _token_expired(expires_at_iso: str, *, min_valid_seconds: int = 0) -> bool:
    try:
        expires_at = datetime.fromisoformat(expires_at_iso.replace("Z", "+00:00"))
    except ValueError:
        return True
    return expires_at <= (
        datetime.now(timezone.utc) + timedelta(seconds=min_valid_seconds)
    )


def _read_user_token_payload() -> tuple[dict[str, Any], ArenaTokenMetadata] | None:
    content_path, meta_path = _token_paths("arena-user:active")
    if not content_path.exists():
        return None
    meta = _load_token_meta(meta_path)
    if meta is None:
        return None
    try:
        payload = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload, meta


def get_cached_user_token_record() -> dict[str, Any] | None:
    loaded = _read_user_token_payload()
    if loaded is None:
        return None
    payload, meta = loaded
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        return None
    refresh_token = payload.get("refresh_token")
    token_type = payload.get("token_type")
    scope = payload.get("scope")
    return {
        "access_token": access_token.strip(),
        "refresh_token": refresh_token.strip()
        if isinstance(refresh_token, str) and refresh_token.strip()
        else None,
        "token_type": token_type.strip()
        if isinstance(token_type, str) and token_type.strip()
        else "Bearer",
        "scope": scope.strip() if isinstance(scope, str) else "",
        "expires_at": meta.expires_at,
        "cached_at": meta.cached_at,
        "identity": meta.identity,
    }


def get_cached_user_access_token(min_valid_seconds: int = 60) -> str | None:
    record = get_cached_user_token_record()
    if record is None:
        return None
    expires_at = record.get("expires_at")
    if not isinstance(expires_at, str) or _token_expired(
        expires_at, min_valid_seconds=min_valid_seconds
    ):
        return None
    token = record.get("access_token")
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def clear_cached_user_token() -> None:
    content_path, meta_path = _token_paths("arena-user:active")
    for path in (content_path, meta_path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def store_user_token(
    *,
    access_token: str,
    refresh_token: str | None,
    expires_in_seconds: int,
    token_type: str = "Bearer",
    scope: str = "",
) -> None:
    if not access_token:
        return
    try:
        TOKEN_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        identity = "arena-user:active"
        content_path, meta_path = _token_paths(identity)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=max(60, int(expires_in_seconds)))
        content_path.write_text(
            json.dumps(
                {
                    "access_token": access_token,
                    "refresh_token": refresh_token or "",
                    "token_type": token_type,
                    "scope": scope,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _secure_file_permissions(content_path)
        meta = ArenaTokenMetadata(
            identity=identity,
            cached_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
        )
        meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
        _secure_file_permissions(meta_path)
    except OSError:
        return
