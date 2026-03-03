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

ATPROTO_CACHE_ROOT = Path(
    os.environ.get(
        "CONTEXTUALIZE_ATPROTO_CACHE",
        os.path.expanduser("~/.local/share/contextualize/cache/atproto/v1"),
    )
)
API_CACHE_ROOT = ATPROTO_CACHE_ROOT / "api"
RENDER_CACHE_ROOT = ATPROTO_CACHE_ROOT / "render"
MEDIA_CACHE_ROOT = ATPROTO_CACHE_ROOT / "media"
IDENTITY_CACHE_ROOT = ATPROTO_CACHE_ROOT / "identity"
TOKEN_CACHE_ROOT = ATPROTO_CACHE_ROOT / "token"
DEFAULT_API_TTL = timedelta(hours=6)
DEFAULT_IDENTITY_TTL = timedelta(days=7)
CACHE_VERSION = 1


@dataclass
class AtprotoCacheMetadata:
    identity: str
    cached_at: str
    size_bytes: int
    cache_version: int = CACHE_VERSION


@dataclass
class AtprotoTokenMetadata:
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


def _load_meta(path: Path) -> AtprotoCacheMetadata | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return AtprotoCacheMetadata(**payload)
    except TypeError:
        return None


def _is_expired(meta: AtprotoCacheMetadata, ttl: timedelta | None) -> bool:
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
        meta = AtprotoCacheMetadata(
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
        meta = AtprotoCacheMetadata(
            identity=identity,
            cached_at=datetime.now(timezone.utc).isoformat(),
            size_bytes=len(content.encode("utf-8")),
        )
        meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    except OSError:
        return


def get_cached_identity(identity: str, ttl: timedelta | None = None) -> str | None:
    effective_ttl = DEFAULT_IDENTITY_TTL if ttl is None else ttl
    content_path, meta_path = _cache_paths(IDENTITY_CACHE_ROOT, identity, "txt")
    if not content_path.exists():
        return None
    meta = _load_meta(meta_path)
    if meta is None or _is_expired(meta, effective_ttl):
        return None
    try:
        value = content_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def store_identity(identity: str, value: str) -> None:
    if not value:
        return
    try:
        IDENTITY_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        content_path, meta_path = _cache_paths(IDENTITY_CACHE_ROOT, identity, "txt")
        content_path.write_text(value, encoding="utf-8")
        meta = AtprotoCacheMetadata(
            identity=identity,
            cached_at=datetime.now(timezone.utc).isoformat(),
            size_bytes=len(value.encode("utf-8")),
        )
        meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    except OSError:
        return


def get_cached_handle_did(handle: str, ttl: timedelta | None = None) -> str | None:
    key = f"handle:{handle.lower().strip()}"
    return get_cached_identity(key, ttl=ttl)


def store_handle_did(handle: str, did: str) -> None:
    key = f"handle:{handle.lower().strip()}"
    store_identity(key, did)


def get_cached_media_bytes(identity: str) -> bytes | None:
    return _get_cached_media_bytes(MEDIA_CACHE_ROOT, identity)


def store_media_bytes(identity: str, content: bytes) -> None:
    _store_media_bytes(MEDIA_CACHE_ROOT, identity, content)


def _token_paths(identity: str) -> tuple[Path, Path]:
    return _cache_paths(TOKEN_CACHE_ROOT, identity, "json")


def _load_token_meta(path: Path) -> AtprotoTokenMetadata | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return AtprotoTokenMetadata(**payload)
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


def _read_oauth_session_payload() -> tuple[dict[str, Any], AtprotoTokenMetadata] | None:
    content_path, meta_path = _token_paths("atproto-oauth:active")
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


def get_cached_oauth_session_record() -> dict[str, Any] | None:
    loaded = _read_oauth_session_payload()
    if loaded is None:
        return None
    payload, meta = loaded
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        return None
    refresh_token = payload.get("refresh_token")
    token_type = payload.get("token_type")
    scope = payload.get("scope")
    client_id = payload.get("client_id")
    auth_server = payload.get("auth_server")
    resource_server = payload.get("resource_server")
    dpop_private_key_pem = payload.get("dpop_private_key_pem")
    dpop_public_jwk = payload.get("dpop_public_jwk")
    auth_server_nonce = payload.get("auth_server_nonce")
    resource_server_nonce = payload.get("resource_server_nonce")
    subject_did = payload.get("subject_did")
    return {
        "access_token": access_token.strip(),
        "refresh_token": refresh_token.strip()
        if isinstance(refresh_token, str) and refresh_token.strip()
        else None,
        "token_type": token_type.strip()
        if isinstance(token_type, str) and token_type.strip()
        else "DPoP",
        "scope": scope.strip() if isinstance(scope, str) else "",
        "client_id": client_id.strip() if isinstance(client_id, str) else "",
        "auth_server": auth_server.strip() if isinstance(auth_server, str) else "",
        "resource_server": resource_server.strip()
        if isinstance(resource_server, str)
        else "",
        "dpop_private_key_pem": dpop_private_key_pem.strip()
        if isinstance(dpop_private_key_pem, str)
        else "",
        "dpop_public_jwk": dpop_public_jwk if isinstance(dpop_public_jwk, dict) else {},
        "auth_server_nonce": auth_server_nonce.strip()
        if isinstance(auth_server_nonce, str) and auth_server_nonce.strip()
        else None,
        "resource_server_nonce": resource_server_nonce.strip()
        if isinstance(resource_server_nonce, str) and resource_server_nonce.strip()
        else None,
        "subject_did": subject_did.strip()
        if isinstance(subject_did, str) and subject_did.strip()
        else None,
        "expires_at": meta.expires_at,
        "cached_at": meta.cached_at,
        "identity": meta.identity,
    }


def get_cached_oauth_access_token(min_valid_seconds: int = 60) -> str | None:
    record = get_cached_oauth_session_record()
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


def clear_cached_oauth_session() -> None:
    content_path, meta_path = _token_paths("atproto-oauth:active")
    for path in (content_path, meta_path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def store_oauth_session(
    *,
    access_token: str,
    refresh_token: str | None,
    expires_in_seconds: int,
    token_type: str,
    scope: str,
    client_id: str,
    auth_server: str,
    resource_server: str,
    dpop_private_key_pem: str,
    dpop_public_jwk: dict[str, str],
    auth_server_nonce: str | None = None,
    resource_server_nonce: str | None = None,
    subject_did: str | None = None,
) -> None:
    if not access_token:
        return
    try:
        TOKEN_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        identity = "atproto-oauth:active"
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
                    "client_id": client_id,
                    "auth_server": auth_server,
                    "resource_server": resource_server,
                    "dpop_private_key_pem": dpop_private_key_pem,
                    "dpop_public_jwk": dpop_public_jwk,
                    "auth_server_nonce": auth_server_nonce or "",
                    "resource_server_nonce": resource_server_nonce or "",
                    "subject_did": subject_did or "",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _secure_file_permissions(content_path)
        meta = AtprotoTokenMetadata(
            identity=identity,
            cached_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
        )
        meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
        _secure_file_permissions(meta_path)
    except OSError:
        return
