from __future__ import annotations

import hashlib
from pathlib import Path


def _media_cache_key(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def get_cached_media_bytes(root: Path, identity: str) -> bytes | None:
    key = _media_cache_key(identity)
    path = root / f"{key}.bin"
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def store_media_bytes(root: Path, identity: str, content: bytes) -> None:
    if not content:
        return
    root.mkdir(parents=True, exist_ok=True)
    key = _media_cache_key(identity)
    path = root / f"{key}.bin"
    path.write_bytes(content)
