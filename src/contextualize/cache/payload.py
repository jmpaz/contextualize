from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .url import parse_duration

PAYLOAD_CACHE_ROOT = Path(
    os.environ.get(
        "CONTEXTUALIZE_PAYLOAD_CACHE",
        os.path.expanduser("~/.local/share/contextualize/cache/payload/v1"),
    )
)
DEFAULT_TTL = timedelta(minutes=30)
CACHE_VERSION = 1


def _payload_cache_ttl() -> timedelta:
    raw = (os.environ.get("CONTEXTUALIZE_PAYLOAD_CACHE_TTL") or "").strip()
    if not raw:
        return DEFAULT_TTL
    try:
        return parse_duration(raw)
    except ValueError:
        return DEFAULT_TTL


def payload_cache_enabled() -> bool:
    raw = (os.environ.get("CONTEXTUALIZE_DISABLE_PAYLOAD_CACHE") or "").strip().lower()
    return raw not in {"1", "true", "yes"}


def _cache_key(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(key: str) -> Path:
    return PAYLOAD_CACHE_ROOT / key[:2] / f"{key}.json"


def _is_expired(cached_at: str, ttl: timedelta) -> bool:
    if ttl == timedelta(0):
        return True
    try:
        ts = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    return (datetime.now(timezone.utc) - ts) > ttl


def load_payload_cache(key_data: dict[str, Any]) -> dict[str, Any] | None:
    ttl = _payload_cache_ttl()
    key = _cache_key(key_data)
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if payload.get("cache_version") != CACHE_VERSION:
        return None
    if _is_expired(str(payload.get("cached_at") or ""), ttl):
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    return data


def store_payload_cache(key_data: dict[str, Any], data: dict[str, Any]) -> None:
    key = _cache_key(key_data)
    path = _cache_path(key)
    body = {
        "cache_version": CACHE_VERSION,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return
