from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ARTIFACT_CACHE_ROOT = Path(
    os.environ.get(
        "CONTEXTUALIZE_ARTIFACT_CACHE",
        os.path.expanduser("~/.local/share/contextualize/cache/artifacts/v1"),
    )
)
ARTIFACT_CACHE_VERSION = 1


@dataclass(frozen=True)
class SourceArtifact:
    sha256: str
    byte_size: int
    path: Path
    metadata_path: Path
    content_type: str | None
    provenance: Mapping[str, Any] | None


@dataclass(frozen=True)
class SourceArtifactMetadata:
    sha256: str
    byte_size: int
    imported_at: str
    content_type: str | None = None
    provenance: Mapping[str, Any] | None = None
    cache_version: int = ARTIFACT_CACHE_VERSION


def _validate_digest(digest: str) -> str:
    normalized = digest.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("Artifact digest must be a hexadecimal SHA-256 value")
    return normalized


def artifact_paths(
    digest: str, *, root: Path | None = None
) -> tuple[Path, Path]:
    normalized = _validate_digest(digest)
    cache_root = root or ARTIFACT_CACHE_ROOT
    directory = cache_root / "sha256" / normalized[:2] / normalized[2:4]
    return directory / normalized, directory / f"{normalized}.meta.json"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_source_artifact(
    digest: str, *, root: Path | None = None
) -> SourceArtifact | None:
    normalized = _validate_digest(digest)
    content_path, metadata_path = artifact_paths(normalized, root=root)
    try:
        if not content_path.is_file() or content_path.is_symlink():
            return None
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        metadata = SourceArtifactMetadata(**payload)
        if metadata.cache_version != ARTIFACT_CACHE_VERSION:
            return None
        if metadata.sha256 != normalized or metadata.byte_size != content_path.stat().st_size:
            return None
        if _sha256_file(content_path) != normalized:
            return None
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return SourceArtifact(
        sha256=normalized,
        byte_size=metadata.byte_size,
        path=content_path,
        metadata_path=metadata_path,
        content_type=metadata.content_type,
        provenance=metadata.provenance,
    )


def import_source_artifact(
    content: bytes,
    *,
    content_type: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> SourceArtifact:
    if not content:
        raise ValueError("Source artifact content must not be empty")
    digest = hashlib.sha256(content).hexdigest()
    existing = get_source_artifact(digest, root=root)
    if existing is not None:
        return existing
    content_path, metadata_path = artifact_paths(digest, root=root)
    content_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = SourceArtifactMetadata(
        sha256=digest,
        byte_size=len(content),
        imported_at=datetime.now(timezone.utc).isoformat(),
        content_type=content_type,
        provenance=provenance,
    )
    content_fd, content_temp = tempfile.mkstemp(
        prefix=f".{digest}.", suffix=".partial", dir=content_path.parent
    )
    metadata_fd, metadata_temp = tempfile.mkstemp(
        prefix=f".{digest}.", suffix=".meta.partial", dir=content_path.parent
    )
    try:
        with os.fdopen(content_fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        with os.fdopen(metadata_fd, "w", encoding="utf-8") as handle:
            json.dump(asdict(metadata), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(content_temp, 0o600)
        os.chmod(metadata_temp, 0o600)
        os.replace(content_temp, content_path)
        os.replace(metadata_temp, metadata_path)
    finally:
        for temporary in (content_temp, metadata_temp):
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
    artifact = get_source_artifact(digest, root=root)
    if artifact is None:
        raise OSError("Imported source artifact failed integrity validation")
    return artifact
