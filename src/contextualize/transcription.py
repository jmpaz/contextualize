from __future__ import annotations

from pathlib import Path
from typing import Any

from contextualize.references.audio_transcription import transcribe_audio_bytes
from contextualize.references.audio_transcription import transcribe_audio_file
from contextualize.references.audio_transcription import transcribe_media_bytes
from contextualize.references.audio_transcription import transcribe_media_bytes_result
from contextualize.references.audio_transcription import transcribe_media_file
from contextualize.references.audio_transcription import transcribe_media_file_result
from contextualize.references.audio_transcription import (
    record_transcription_routing_summary,
)
from contextualize.references.audio_transcription import transcription_routing_summary
from contextualize.references.audio_transcription import _routing_cache_identity

__all__ = [
    "record_transcription_routing_summary",
    "transcribe_audio_bytes",
    "transcribe_audio_file",
    "transcribe_media_bytes",
    "transcribe_media_bytes_result",
    "transcribe_media_file",
    "transcribe_media_file_result",
    "transcription_routing_identity",
    "transcription_routing_summary",
]


def transcription_routing_identity(
    *,
    filename: str | Path,
    content_type: str,
    plugin_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _routing_cache_identity(
        filename=str(filename),
        content_type=content_type,
        plugin_overrides=plugin_overrides,
    )
