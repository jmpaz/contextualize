from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any

_AUDIO_SUFFIX_TO_MIME: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
}

AUDIO_SUFFIXES: frozenset[str] = frozenset(_AUDIO_SUFFIX_TO_MIME)


def is_audio_suffix(suffix: str) -> bool:
    return suffix.lower() in AUDIO_SUFFIXES


def transcribe_audio_file(path: str | Path, *, timeout: float = 600) -> str:
    audio_path = Path(path)
    data = audio_path.read_bytes()
    content_type = _guess_audio_content_type(audio_path.name)
    return transcribe_audio_bytes(
        data,
        filename=audio_path.name,
        content_type=content_type,
        timeout=timeout,
    )


def transcribe_audio_bytes(
    data: bytes,
    *,
    filename: str,
    content_type: str | None = None,
    timeout: float = 600,
) -> str:
    import httpx

    api_base = os.environ.get("WHISPER_API_BASE", "https://api.openai.com/v1")
    api_key = os.environ.get("WHISPER_API_KEY")
    model = os.environ.get("WHISPER_MODEL", "whisper-1")
    endpoint = f"{api_base.rstrip('/')}/audio/transcriptions"

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resolved_content_type = content_type or _guess_audio_content_type(filename)
    response = httpx.post(
        endpoint,
        headers=headers,
        files={"file": (filename, data, resolved_content_type)},
        data={"model": model, "response_format": "verbose_json"},
        timeout=timeout,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Whisper API error: {response.status_code} {response.text}")

    payload = response.json()
    return _extract_transcription_text(payload)


def _guess_audio_content_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in _AUDIO_SUFFIX_TO_MIME:
        return _AUDIO_SUFFIX_TO_MIME[suffix]
    guessed, _ = mimetypes.guess_type(filename)
    if guessed and guessed.startswith("audio/"):
        return guessed
    return "audio/mpeg"


def _extract_transcription_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    segments = payload.get("segments")
    if isinstance(segments, list):
        extracted: list[str] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            text = segment.get("text")
            if isinstance(text, str):
                trimmed = text.strip()
                if trimmed:
                    extracted.append(trimmed)
        if extracted:
            return "\n\n".join(extracted)

    text = payload.get("text")
    if isinstance(text, str):
        return text.strip()
    return ""
