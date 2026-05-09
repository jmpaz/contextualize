from __future__ import annotations

from contextvars import ContextVar
import json
import os
from pathlib import Path
import tempfile
from typing import Any

RUN_METADATA_ENV = "CONTEXTUALIZE_RUN_METADATA_PATH"
RUN_METADATA_VERSION = 1

_RUN_TRANSCRIPTIONS: ContextVar[tuple[dict[str, Any], ...]] = ContextVar(
    "contextualize_run_transcriptions", default=()
)


def reset_run_metadata() -> None:
    _RUN_TRANSCRIPTIONS.set(())


def record_transcription(
    *,
    provider: str | None,
    model: str | None,
    diarize: bool,
    speakers: int | None,
    language: str | None,
    filename: str,
    source: str,
) -> None:
    if not _metadata_path():
        return
    event = {
        "provider": provider,
        "model": model,
        "diarize": bool(diarize),
        "speakers": speakers,
        "language": language,
        "filename": filename,
        "source": source,
    }
    _RUN_TRANSCRIPTIONS.set((*_RUN_TRANSCRIPTIONS.get(), event))


def flush_run_metadata() -> None:
    path = _metadata_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": RUN_METADATA_VERSION,
        "transcriptions": list(_RUN_TRANSCRIPTIONS.get()),
    }
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, sort_keys=True)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _metadata_path() -> Path | None:
    raw_path = (os.environ.get(RUN_METADATA_ENV) or "").strip()
    return Path(raw_path) if raw_path else None
