from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

LOCAL_MEDIA_CACHE_ROOT = Path(
    os.environ.get(
        "CONTEXTUALIZE_LOCAL_MEDIA_CACHE",
        os.path.expanduser("~/.local/share/contextualize/cache/local-media/v1"),
    )
)
TRANSCRIPT_CACHE_ROOT = LOCAL_MEDIA_CACHE_ROOT / "transcript"
GATE_CACHE_ROOT = LOCAL_MEDIA_CACHE_ROOT / "gate"
CACHE_VERSION = 1


@dataclass
class LocalMediaCacheMetadata:
    identity: str
    cached_at: str
    operation: str
    source_sha256: str
    source_suffix: str
    size_bytes: int
    cache_version: int = CACHE_VERSION


def _cache_key(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _cache_paths(root: Path, identity: str, ext: str) -> tuple[Path, Path]:
    key = _cache_key(identity)
    content_path = root / f"{key}.{ext}"
    meta_path = root / f"{key}.meta.json"
    return content_path, meta_path


def _load_meta(path: Path) -> LocalMediaCacheMetadata | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        meta = LocalMediaCacheMetadata(**payload)
    except TypeError:
        return None
    if meta.cache_version != CACHE_VERSION:
        return None
    return meta


def get_cached_transcript(identity: str) -> str | None:
    content_path, meta_path = _cache_paths(TRANSCRIPT_CACHE_ROOT, identity, "txt")
    if not content_path.exists():
        return None
    if _load_meta(meta_path) is None:
        return None
    try:
        content = content_path.read_text(encoding="utf-8")
    except OSError:
        return None
    return content or None


def store_transcript(
    identity: str,
    content: str,
    *,
    operation: str,
    source_sha256: str,
    source_suffix: str,
) -> None:
    if not content:
        return
    try:
        TRANSCRIPT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        content_path, meta_path = _cache_paths(TRANSCRIPT_CACHE_ROOT, identity, "txt")
        content_path.write_text(content, encoding="utf-8")
        meta = LocalMediaCacheMetadata(
            identity=identity,
            cached_at=datetime.now(timezone.utc).isoformat(),
            operation=operation,
            source_sha256=source_sha256,
            source_suffix=source_suffix,
            size_bytes=len(content.encode("utf-8")),
        )
        meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    except OSError:
        return


def get_cached_gate_decision(identity: str) -> dict[str, object] | None:
    content_path, meta_path = _cache_paths(GATE_CACHE_ROOT, identity, "json")
    if not content_path.exists():
        return None
    if _load_meta(meta_path) is None:
        return None
    try:
        payload = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def store_gate_decision(
    identity: str,
    payload: dict[str, object],
    *,
    operation: str,
    source_sha256: str,
    source_suffix: str,
) -> None:
    if not payload:
        return
    try:
        GATE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        content_path, meta_path = _cache_paths(GATE_CACHE_ROOT, identity, "json")
        content_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        meta = LocalMediaCacheMetadata(
            identity=identity,
            cached_at=datetime.now(timezone.utc).isoformat(),
            operation=operation,
            source_sha256=source_sha256,
            source_suffix=source_suffix,
            size_bytes=len(content_path.read_bytes()),
        )
        meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    except OSError:
        return
