from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from contextualize.cache.local_media import (
    get_cached_transcript as get_cached_local_media_transcript,
)
from contextualize.cache.local_media import (
    store_transcript as store_local_media_transcript,
)

_AUDIO_SUFFIX_TO_MIME: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
}
_VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mov",
        ".m4v",
        ".webm",
        ".mkv",
        ".avi",
        ".wmv",
        ".flv",
        ".mpeg",
        ".mpg",
    }
)
_VIDEO_CONTENT_TYPE_TO_SUFFIX: dict[str, str] = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
    "video/x-ms-wmv": ".wmv",
    "video/mpeg": ".mpeg",
}

AUDIO_SUFFIXES: frozenset[str] = frozenset(_AUDIO_SUFFIX_TO_MIME)
VIDEO_SUFFIXES: frozenset[str] = _VIDEO_SUFFIXES


def is_audio_suffix(suffix: str) -> bool:
    return suffix.lower() in AUDIO_SUFFIXES


def is_video_suffix(suffix: str) -> bool:
    return suffix.lower() in VIDEO_SUFFIXES


def is_media_suffix(suffix: str) -> bool:
    return is_audio_suffix(suffix) or is_video_suffix(suffix)


def transcribe_media_file(
    path: str | Path,
    *,
    timeout: float = 600,
    use_cache: bool = True,
    refresh_cache: bool | None = None,
) -> str:
    media_path = Path(path)
    suffix = media_path.suffix.lower()
    if is_audio_suffix(suffix):
        return transcribe_audio_file(
            media_path,
            timeout=timeout,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )
    if is_video_suffix(suffix):
        source_sha256 = _sha256_file(media_path)
        cache_identity = _transcription_cache_identity(
            operation="video-transcription",
            source_sha256=source_sha256,
            source_suffix=suffix,
        )
        should_refresh = _should_refresh_transcription_cache(
            "video", refresh_cache=refresh_cache
        )
        if use_cache and not should_refresh:
            cached = get_cached_local_media_transcript(cache_identity)
            if cached is not None:
                return cached

        transcript = _transcribe_video_file(media_path, timeout=timeout)
        if use_cache and transcript.strip():
            store_local_media_transcript(
                cache_identity,
                transcript,
                operation="video-transcription",
                source_sha256=source_sha256,
                source_suffix=suffix,
            )
        return transcript
    raise RuntimeError(
        f"Unsupported media suffix for transcription: {suffix or '<none>'}"
    )


def transcribe_media_bytes(
    data: bytes,
    *,
    filename: str,
    content_type: str | None = None,
    timeout: float = 600,
) -> str:
    kind = _infer_media_kind(filename=filename, content_type=content_type)
    if kind == "audio":
        return transcribe_audio_bytes(
            data,
            filename=filename,
            content_type=content_type,
            timeout=timeout,
        )
    if kind == "video":
        suffix = Path(filename).suffix.lower()
        if not suffix:
            suffix = _video_suffix_from_content_type(content_type) or ".mp4"
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / f"media{suffix}"
            video_path.write_bytes(data)
            return _transcribe_video_file(video_path, timeout=timeout)
    raise RuntimeError("Unsupported media payload for transcription")


def transcribe_audio_file(
    path: str | Path,
    *,
    timeout: float = 600,
    use_cache: bool = True,
    refresh_cache: bool | None = None,
) -> str:
    audio_path = Path(path)
    data = audio_path.read_bytes()
    suffix = audio_path.suffix.lower()
    source_sha256 = _sha256_bytes(data)
    cache_identity = _transcription_cache_identity(
        operation="audio-transcription",
        source_sha256=source_sha256,
        source_suffix=suffix,
    )
    should_refresh = _should_refresh_transcription_cache(
        "audio", refresh_cache=refresh_cache
    )
    if use_cache and not should_refresh:
        cached = get_cached_local_media_transcript(cache_identity)
        if cached is not None:
            return cached

    content_type = _guess_audio_content_type(audio_path.name)
    try:
        transcript = transcribe_audio_bytes(
            data,
            filename=audio_path.name,
            content_type=content_type,
            timeout=timeout,
        )
    except RuntimeError as exc:
        if not _should_retry_chunked_transcription(exc):
            raise
        chunked_transcript = _transcribe_audio_file_in_chunks(
            audio_path, timeout=timeout
        )
        if chunked_transcript:
            transcript = chunked_transcript
        else:
            raise

    if use_cache and transcript.strip():
        store_local_media_transcript(
            cache_identity,
            transcript,
            operation="audio-transcription",
            source_sha256=source_sha256,
            source_suffix=suffix,
        )
    return transcript


def transcribe_audio_bytes(
    data: bytes,
    *,
    filename: str,
    content_type: str | None = None,
    timeout: float = 600,
) -> str:
    if not _is_whisper_configured():
        raise RuntimeError(
            "Whisper transcription requires WHISPER_API_BASE or WHISPER_API_KEY"
        )
    import httpx

    api_base = _whisper_api_base()
    api_key = os.environ.get("WHISPER_API_KEY")
    model = _whisper_model()
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
    text = _extract_transcription_text(payload)
    if not text:
        raise RuntimeError("Whisper API returned no transcription text")
    return text


def _guess_audio_content_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in _AUDIO_SUFFIX_TO_MIME:
        return _AUDIO_SUFFIX_TO_MIME[suffix]
    guessed, _ = mimetypes.guess_type(filename)
    if guessed and guessed.startswith("audio/"):
        return guessed
    return "audio/mpeg"


def _is_whisper_configured() -> bool:
    return bool(
        (os.environ.get("WHISPER_API_BASE") or "").strip()
        or (os.environ.get("WHISPER_API_KEY") or "").strip()
    )


def _whisper_api_base() -> str:
    value = (os.environ.get("WHISPER_API_BASE") or "").strip()
    return value or "https://api.openai.com/v1"


def _whisper_model() -> str:
    value = (os.environ.get("WHISPER_MODEL") or "").strip()
    return value or "whisper-1"


def _chunk_seconds_cache_component() -> str:
    value = (os.environ.get("WHISPER_CHUNK_SECONDS") or "").strip()
    return value or "25"


def _infer_media_kind(*, filename: str, content_type: str | None) -> str | None:
    suffix = Path(filename).suffix.lower()
    if is_audio_suffix(suffix):
        return "audio"
    if is_video_suffix(suffix):
        return "video"
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized.startswith("audio/"):
        return "audio"
    if normalized.startswith("video/"):
        return "video"
    return None


def _video_suffix_from_content_type(content_type: str | None) -> str | None:
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    return _VIDEO_CONTENT_TYPE_TO_SUFFIX.get(normalized)


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


def _should_retry_chunked_transcription(exc: RuntimeError) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    message = str(exc)
    return message.startswith("Whisper API error: 5")


def _transcribe_video_file(video_path: Path, *, timeout: float) -> str:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Video transcription requires ffmpeg in PATH")
    with tempfile.TemporaryDirectory() as tmpdir:
        extracted_path = Path(tmpdir) / "audio.wav"
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(extracted_path),
            ],
            capture_output=True,
            text=True,
            timeout=max(timeout, 120),
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout or "unknown error"
            raise RuntimeError(f"ffmpeg video audio extraction failed: {detail}")
        if not extracted_path.exists() or extracted_path.stat().st_size == 0:
            raise RuntimeError("Video transcription failed: no audio stream found")
        return transcribe_audio_file(
            extracted_path,
            timeout=timeout,
            use_cache=False,
            refresh_cache=False,
        )


def _transcribe_audio_file_in_chunks(audio_path: Path, *, timeout: float) -> str:
    chunk_seconds = _get_chunk_seconds()
    with tempfile.TemporaryDirectory() as tmpdir:
        pattern = str(Path(tmpdir) / "chunk_%04d.wav")
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(audio_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "segment",
                "-segment_time",
                str(chunk_seconds),
                "-c:a",
                "pcm_s16le",
                pattern,
            ],
            capture_output=True,
            text=True,
            timeout=max(timeout, 120),
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout or "unknown error"
            raise RuntimeError(f"ffmpeg audio chunking failed: {detail}")

        chunk_paths = sorted(Path(tmpdir).glob("chunk_*.wav"))
        if not chunk_paths:
            raise RuntimeError("ffmpeg audio chunking produced no chunks")

        parts: list[str] = []
        for chunk_path in chunk_paths:
            chunk_data = chunk_path.read_bytes()
            text = transcribe_audio_bytes(
                chunk_data,
                filename=chunk_path.name,
                content_type="audio/wav",
                timeout=timeout,
            ).strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)


def _get_chunk_seconds() -> int:
    raw = os.environ.get("WHISPER_CHUNK_SECONDS", "25")
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(
            f"Invalid WHISPER_CHUNK_SECONDS value: {raw!r}. Expected a positive integer."
        ) from None
    if value <= 0:
        raise RuntimeError(
            f"Invalid WHISPER_CHUNK_SECONDS value: {raw!r}. Expected a positive integer."
        )
    return value


def _should_refresh_transcription_cache(
    kind: str, *, refresh_cache: bool | None
) -> bool:
    if refresh_cache is not None:
        return bool(refresh_cache)

    from contextualize.runtime import get_refresh_audio, get_refresh_videos

    if kind == "audio":
        return get_refresh_audio()
    if kind == "video":
        return get_refresh_videos()
    return False


def _transcription_cache_identity(
    *,
    operation: str,
    source_sha256: str,
    source_suffix: str,
) -> str:
    payload = {
        "operation": operation,
        "source_sha256": source_sha256,
        "source_suffix": source_suffix,
        "whisper_api_base": _whisper_api_base(),
        "whisper_model": _whisper_model(),
        "whisper_chunk_seconds": _chunk_seconds_cache_component(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
