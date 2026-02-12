from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path


def _write_temp_file(content: bytes, *, suffix: str) -> Path | None:
    if not content:
        return None
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, content)
    finally:
        os.close(fd)
    return Path(path)


def download_cached_media_to_temp(
    url: str,
    *,
    suffix: str,
    headers: Mapping[str, str] | None,
    cache_identity: str,
    get_cached_media_bytes: Callable[[str], bytes | None],
    store_media_bytes: Callable[[str, bytes], None],
    refresh_cache: bool,
    timeout_seconds: float = 30.0,
    on_cache_hit: Callable[[str], None] | None = None,
    on_cache_miss: Callable[[str], None] | None = None,
) -> Path | None:
    cached = None if refresh_cache else get_cached_media_bytes(cache_identity)
    if cached:
        if on_cache_hit is not None:
            on_cache_hit(cache_identity)
        return _write_temp_file(cached, suffix=suffix)

    if on_cache_miss is not None:
        on_cache_miss(cache_identity)

    import requests

    try:
        response = requests.get(
            url,
            headers=dict(headers or {}),
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        content = response.content
    except Exception:
        return None

    if not content:
        return None
    store_media_bytes(cache_identity, content)
    return _write_temp_file(content, suffix=suffix)
