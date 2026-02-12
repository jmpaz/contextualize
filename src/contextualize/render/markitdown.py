from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Protocol
from urllib.parse import urlparse


class MarkItDownConversionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MarkItDownResult:
    markdown: str
    title: str | None


class ResponseLike(Protocol):
    content: bytes
    headers: Mapping[str, str]
    url: str


@lru_cache(maxsize=1)
def _load_dotenv_once() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv
    except Exception:
        return

    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=False)


def _data_home() -> Path:
    base = (os.getenv("XDG_DATA_HOME") or "").strip()
    if base:
        return Path(base)
    return Path.home() / ".local" / "share"


def _llm_cache_dir() -> Path:
    return _data_home() / "contextualize" / "cache" / "llm"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@lru_cache(maxsize=1)
def _markitdown_version() -> str | None:
    try:
        from importlib.metadata import version
    except Exception:
        return None

    try:
        return version("markitdown")
    except Exception:
        return None


_DESCRIPTION_HEADING_RE = re.compile(
    r"(?m)^# Description(?: \(auto-generated\))?:?\s*$"
)
_HAS_LLM_DESCRIPTION_RE = re.compile(r"(?m)^# Description")
_AUTO_DESCRIPTION_HEADING = "# Description (auto-generated):"
_AUTO_VIDEO_HEADING = "# Video (auto-generated):"
_IMAGE_SUFFIXES = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".heic",
        ".heif",
        ".avif",
        ".tif",
        ".tiff",
    }
)
_IMAGE_CONVERT_SUFFIXES = frozenset(
    {".gif", ".webp", ".heic", ".heif", ".avif", ".tif", ".tiff"}
)
_VIDEO_CONVERT_SUFFIXES = frozenset({".mov", ".avi", ".mkv", ".webm", ".m4v", ".mp4"})
_VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mov", ".mpeg", ".mpg", ".webm", ".avi", ".mkv", ".m4v", ".gif"}
)
_AUDIO_SUFFIXES = frozenset(
    {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".aiff", ".pcm", ".pcm16"}
)
_CONTENT_TYPE_SUFFIXES: dict[str, str] = {
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/webp": ".webp",
    "image/avif": ".avif",
    "image/tiff": ".tiff",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/mp4": ".mp4",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/aiff": ".aiff",
}
_VIDEO_MIME_BY_SUFFIX: dict[str, str] = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".m4v": "video/mp4",
}
_AUDIO_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".aiff": "aiff",
    ".aac": "aac",
    ".ogg": "ogg",
    ".flac": "flac",
    ".m4a": "m4a",
    ".pcm": "pcm16",
    ".pcm16": "pcm16",
}


def _postprocess_image_markdown(markdown: str) -> str:
    return _DESCRIPTION_HEADING_RE.sub(_AUTO_DESCRIPTION_HEADING, markdown, count=1)


def _format_auto_generated_description(text: str) -> str:
    return f"{_AUTO_DESCRIPTION_HEADING}\n{text}\n"


def _is_non_recoverable_video_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "error code: 401" in message
        or "error code: 402" in message
        or "error code: 403" in message
        or "'code': 401" in message
        or "'code': 402" in message
        or "'code': 403" in message
        or "insufficient" in message
        or "requires at least $" in message
        or "unauthorized" in message
        or "forbidden" in message
    )


def _format_auto_generated_video_fallback(path: Path) -> str:
    lines = [_AUTO_VIDEO_HEADING]
    duration = _video_duration_seconds(path)
    if duration is not None:
        lines.append(f"DurationSeconds: {duration:.3f}")
    has_audio = _video_has_audio_stream(path)
    lines.append(f"HasAudio: {'yes' if has_audio else 'no'}")
    if has_audio:
        is_silent = _video_audio_is_silent(path)
        lines.append(f"AudioSilent: {'yes' if is_silent else 'no'}")
    lines.append(
        "Detailed video analysis was unavailable; this fallback preserves video modality."
    )
    return "\n".join(lines) + "\n"


def _verbose_log(message: str) -> None:
    from ..runtime import get_verbose_logging

    if get_verbose_logging():
        print(message)


def _cache_entries_dir() -> Path:
    return _llm_cache_dir() / "v1"


def _cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _cache_entry_path(key: str) -> Path:
    return _cache_entries_dir() / key[:2] / f"{key}.json"


def _ffmpeg_path() -> str | None:
    configured = (os.getenv("FFMPEG_PATH") or "").strip()
    if configured:
        return configured
    return shutil.which("ffmpeg")


def _ffprobe_path() -> str | None:
    configured = (os.getenv("FFPROBE_PATH") or "").strip()
    if configured:
        return configured
    return shutil.which("ffprobe")


def _mktemp_output(suffix: str) -> Path:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return Path(path)


def _transcode_image_to_jpg(source: Path) -> Path:
    output = _mktemp_output(".jpg")
    try:
        from PIL import Image
    except Exception as exc:
        output.unlink(missing_ok=True)
        raise MarkItDownConversionError(
            f"Auto-conversion to JPG requires Pillow for {source}: {exc}"
        ) from exc

    try:
        with Image.open(source) as image:
            image.convert("RGB").save(output, format="JPEG", quality=95)
    except Exception as exc:
        output.unlink(missing_ok=True)
        raise MarkItDownConversionError(
            f"Failed to auto-convert {source} to JPG: {exc}"
        ) from exc
    return output


def _transcode_video_to_mp4(source: Path) -> Path:
    ffmpeg = _ffmpeg_path()
    if ffmpeg is None:
        raise MarkItDownConversionError(
            f"Auto-conversion to MP4 requires ffmpeg for {source}"
        )
    output = _mktemp_output(".mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        output.unlink(missing_ok=True)
        raise MarkItDownConversionError(
            f"Failed to invoke ffmpeg for {source}: {exc}"
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "unknown ffmpeg error"
        output.unlink(missing_ok=True)
        raise MarkItDownConversionError(
            f"Failed to auto-convert {source} to MP4: {stderr}"
        )
    return output


def _extract_video_frame_to_jpg(source: Path) -> Path:
    ffmpeg = _ffmpeg_path()
    if ffmpeg is None:
        raise MarkItDownConversionError(
            f"Extracting a JPG frame requires ffmpeg for {source}"
        )
    output = _mktemp_output(".jpg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        output.unlink(missing_ok=True)
        raise MarkItDownConversionError(
            f"Failed to invoke ffmpeg for {source}: {exc}"
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "unknown ffmpeg error"
        output.unlink(missing_ok=True)
        raise MarkItDownConversionError(
            f"Failed to extract JPG frame from {source}: {stderr}"
        )
    return output


def _normalization_candidates(source: Path) -> list[Path]:
    suffix = source.suffix.lower()
    if suffix in _IMAGE_CONVERT_SUFFIXES:
        if suffix == ".gif":
            candidates: list[Path] = []
            try:
                mp4 = _transcode_video_to_mp4(source)
                candidates.append(mp4)
                try:
                    frame = _extract_video_frame_to_jpg(mp4)
                except MarkItDownConversionError:
                    frame = _extract_video_frame_to_jpg(source)
                candidates.append(frame)
            except MarkItDownConversionError:
                candidates.append(_extract_video_frame_to_jpg(source))
            return candidates
        return [_transcode_image_to_jpg(source)]
    if suffix in _VIDEO_CONVERT_SUFFIXES:
        if suffix == ".mp4":
            return [_extract_video_frame_to_jpg(source)]
        mp4 = _transcode_video_to_mp4(source)
        return [mp4, _extract_video_frame_to_jpg(mp4)]
    return []


def _suffix_from_content_type(content_type: str) -> str | None:
    return _CONTENT_TYPE_SUFFIXES.get(content_type.lower())


def _extract_llm_text(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return None
    message = getattr(choices[0], "message", None)
    if message is None:
        return None
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return "\n\n".join(parts)
    return None


def _llm_max_attempts() -> int:
    raw = (os.getenv("OPENAI_MAX_ATTEMPTS") or "").strip()
    if not raw:
        return 4
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


def _llm_retry_delay_seconds(attempt: int) -> float:
    import random

    base = min(20.0, 1.0 * (2 ** max(0, attempt - 1)))
    return base + random.uniform(0.0, 0.25)


def _is_transient_llm_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {
        408,
        409,
        425,
        429,
        500,
        502,
        503,
        504,
    }:
        return True
    name = type(exc).__name__
    transient_names = {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }
    if name in transient_names:
        return True
    msg = str(exc).lower()
    return (
        "temporarily unavailable" in msg
        or "cloudflare" in msg
        or "timeout" in msg
        or "connection" in msg
    )


def _llm_chat_completion(
    client: Any, *, model: str, messages: list[dict[str, Any]]
) -> Any:
    import time

    max_attempts = _llm_max_attempts()
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except Exception as exc:
            last_exc = exc
            if not _is_transient_llm_error(exc) or attempt >= max_attempts:
                break
            wait = _llm_retry_delay_seconds(attempt)
            logging.getLogger(__name__).warning(
                "Transient LLM error (%s). Retrying in %.1fs (%d/%d).",
                type(exc).__name__,
                wait,
                attempt,
                max_attempts,
            )
            time.sleep(wait)
    raise MarkItDownConversionError(f"LLM request failed: {last_exc}")


def _llm_video_markdown(
    data: bytes | None, *, suffix: str, prompt: str, video_url: str | None = None
) -> str:
    llm_client, llm_model = _build_llm_config()
    if llm_client is None or llm_model is None:
        raise MarkItDownConversionError("LLM client not configured for video analysis")
    if video_url:
        input_url = video_url
    else:
        if data is None:
            raise MarkItDownConversionError("Video data is required for video analysis")
        mime = _VIDEO_MIME_BY_SUFFIX.get(suffix, "video/mp4")
        encoded = base64.b64encode(data).decode("ascii")
        input_url = f"data:{mime};base64,{encoded}"
    response = _llm_chat_completion(
        llm_client,
        model=llm_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": input_url}},
                ],
            }
        ],
    )
    text = _extract_llm_text(response)
    if not text:
        raise MarkItDownConversionError("No text returned from video LLM response")
    return _format_auto_generated_description(text)


def _llm_audio_markdown(data: bytes, *, suffix: str, prompt: str) -> str:
    llm_client, llm_model = _build_llm_config()
    if llm_client is None or llm_model is None:
        raise MarkItDownConversionError("LLM client not configured for audio analysis")
    audio_format = _AUDIO_FORMAT_BY_SUFFIX.get(suffix, "wav")
    encoded = base64.b64encode(data).decode("ascii")
    response = _llm_chat_completion(
        llm_client,
        model=llm_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded, "format": audio_format},
                    },
                ],
            }
        ],
    )
    text = _extract_llm_text(response)
    if not text:
        raise MarkItDownConversionError("No text returned from audio LLM response")
    return _format_auto_generated_description(text)


def _is_video_media(*, suffix: str, content_type: str = "") -> bool:
    return suffix in _VIDEO_SUFFIXES or content_type.lower().startswith("video/")


def _is_audio_media(*, suffix: str, content_type: str = "") -> bool:
    return suffix in _AUDIO_SUFFIXES or content_type.lower().startswith("audio/")


def _maybe_convert_gif_to_mp4(path: Path) -> Path:
    if path.suffix.lower() != ".gif":
        return path
    return _transcode_video_to_mp4(path)


def _read_cache_entry(key: str) -> dict[str, Any] | None:
    path = _cache_entry_path(key)
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_cache_entry(
    key: str, *, payload: dict[str, Any] | None, markdown: str, title: str | None
) -> None:
    if not isinstance(markdown, str):
        return
    entry_path = _cache_entry_path(key)
    try:
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        cache_entry = {"v": 1, "payload": payload, "markdown": markdown, "title": title}
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=entry_path.parent,
            prefix=f"{key}.",
            suffix=".tmp",
        ) as f:
            f.write(_stable_json(cache_entry))
            tmp_name = f.name
        Path(tmp_name).replace(entry_path)
    except OSError:
        try:
            if "tmp_name" in locals():
                Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_openrouter_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"openrouter.ai", "www.openrouter.ai"}


def _llm_provider_label(base_url: str) -> str:
    host = (urlparse(base_url).hostname or "").lower()
    if host in {"openrouter.ai", "www.openrouter.ai"}:
        return "openrouter"
    if host in {"api.openai.com", "www.api.openai.com"}:
        return "openai"
    if host:
        return host
    return "unknown"


def _openrouter_extra_body(extra_body: Any) -> dict[str, Any]:
    payload: dict[str, Any] = (
        dict(extra_body) if isinstance(extra_body, Mapping) else {}
    )
    provider = payload.get("provider")
    provider_payload = dict(provider) if isinstance(provider, Mapping) else {}
    provider_payload["data_collection"] = "deny"
    payload["provider"] = provider_payload
    return payload


class _OpenRouterCompletionsProxy:
    def __init__(
        self, completions: Any, *, provider: str, add_openrouter_defaults: bool
    ):
        self._completions = completions
        self._provider = provider
        self._add_openrouter_defaults = add_openrouter_defaults

    def create(self, *args: Any, **kwargs: Any) -> Any:
        if self._add_openrouter_defaults:
            kwargs["extra_body"] = _openrouter_extra_body(kwargs.get("extra_body"))
        model = kwargs.get("model")
        model_label = (
            model.strip() if isinstance(model, str) and model.strip() else "unknown"
        )
        _verbose_log(
            "  sending to model: "
            f"provider={self._provider} model={model_label} endpoint=chat.completions"
        )
        return self._completions.create(*args, **kwargs)


class _OpenRouterChatProxy:
    def __init__(self, chat: Any, *, provider: str, add_openrouter_defaults: bool):
        self.completions = _OpenRouterCompletionsProxy(
            chat.completions,
            provider=provider,
            add_openrouter_defaults=add_openrouter_defaults,
        )


class _OpenRouterClientProxy:
    def __init__(self, client: Any, *, provider: str, add_openrouter_defaults: bool):
        self.chat = _OpenRouterChatProxy(
            client.chat,
            provider=provider,
            add_openrouter_defaults=add_openrouter_defaults,
        )


def _configured_llm_base_url() -> str:
    _load_dotenv_once()
    return (
        os.getenv("OPENAI_BASE_URL") or ""
    ).strip() or "https://openrouter.ai/api/v1"


def _image_context() -> tuple[bool, str, str, str, str | None]:
    _load_dotenv_once()
    llm_enabled = bool(
        (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("OPENROUTER_API_KEY") or "").strip()
    )
    base_url = _configured_llm_base_url()
    model = (os.getenv("OPENAI_MODEL") or "").strip() or "google/gemini-3-flash-preview"
    prompt = _image_prompt()
    exiftool_path = (os.getenv("EXIFTOOL_PATH") or "").strip() or shutil.which(
        "exiftool"
    )
    return llm_enabled, base_url, model, prompt, exiftool_path


def _image_prompt() -> str:
    return _compose_alt_text_prompt(modality="image")


def _audio_prompt() -> str:
    return _compose_alt_text_prompt(modality="audio", include_transcript_hint=True)


def _merge_prompt(base_prompt: str, prompt_append: str | None) -> str:
    append = (prompt_append or "").strip()
    if not append:
        return base_prompt
    return f"{base_prompt}\n\n{append}"


def _video_has_audio_stream(path: Path) -> bool:
    ffprobe = _ffprobe_path()
    if ffprobe is None:
        return True
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return True
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _video_duration_seconds(path: Path) -> float | None:
    ffprobe = _ffprobe_path()
    if ffprobe is None:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    raw = (completed.stdout or "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _video_max_volume_db(path: Path, *, start_seconds: float = 0.0) -> float | None:
    ffmpeg = _ffmpeg_path()
    if ffmpeg is None:
        return None
    cmd = [
        ffmpeg,
        "-v",
        "info",
        "-ss",
        f"{max(0.0, start_seconds):.3f}",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-t",
        "8",
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    stderr = completed.stderr or ""
    match = re.search(r"max_volume:\s*([-\w.]+)\s*dB", stderr)
    if not match:
        return None
    max_volume = match.group(1).strip().lower()
    if max_volume == "-inf":
        return float("-inf")
    try:
        return float(max_volume)
    except ValueError:
        return None


def _video_audio_is_silent(path: Path) -> bool:
    first_max = _video_max_volume_db(path, start_seconds=0.0)
    if first_max is None:
        return False
    if first_max != float("-inf"):
        return False
    duration = _video_duration_seconds(path)
    if duration is None or duration <= 8.0:
        return True
    tail_start = max(0.0, duration - 8.0)
    last_max = _video_max_volume_db(path, start_seconds=tail_start)
    if last_max is None:
        return False
    return last_max == float("-inf")


def _video_prompt(path: Path) -> str:
    _load_dotenv_once()
    custom = (os.getenv("OPENAI_VIDEO_PROMPT") or "").strip()
    if custom:
        return custom
    shared = (os.getenv("OPENAI_PROMPT") or "").strip()
    if shared:
        return shared
    if not _video_has_audio_stream(path):
        return _compose_alt_text_prompt(modality="video")
    if _video_audio_is_silent(path):
        return _compose_alt_text_prompt(modality="video")
    return _compose_alt_text_prompt(modality="video", include_transcript_hint=True)


def _compose_alt_text_prompt(
    *, modality: str, include_transcript_hint: bool = False
) -> str:
    base = f"Write detailed alt text for this {modality}"
    if include_transcript_hint:
        return f"{base}, including a transcript, if any speech is present."
    return f"{base}."


def _refresh_images_enabled(explicit: bool) -> bool:
    if explicit:
        return True
    from ..runtime import get_refresh_images, get_refresh_media

    return get_refresh_images() or get_refresh_media()


def _image_cache_payload(
    media_md5: str,
    *,
    llm_enabled: bool,
    base_url: str,
    model: str,
    prompt: str,
    exiftool_path: str | None,
) -> dict[str, Any]:
    return {
        "v": 1,
        "type": "image",
        "media_md5": media_md5,
        "markitdown_version": _markitdown_version(),
        "llm_enabled": llm_enabled,
        "provider": base_url if llm_enabled else None,
        "model": model if llm_enabled else None,
        "prompt": prompt if llm_enabled else None,
        "exiftool_path": exiftool_path,
        "description_heading": "auto-generated",
    }


def _image_cache_lookup(
    payload: dict[str, Any],
) -> tuple[str, MarkItDownResult | None]:
    key = _cache_key(payload)
    cached = _read_cache_entry(key)
    if isinstance(cached, dict):
        cached_markdown = cached.get("markdown")
        cached_title = cached.get("title")
        if isinstance(cached_markdown, str):
            requires_llm_description = bool(payload.get("llm_enabled"))
            if requires_llm_description and not _HAS_LLM_DESCRIPTION_RE.search(
                cached_markdown
            ):
                return key, None
            title = cached_title if isinstance(cached_title, str) else None
            return (
                key,
                MarkItDownResult(
                    markdown=_postprocess_image_markdown(cached_markdown),
                    title=title,
                ),
            )
    return key, None


def _convert_markitdown(
    source: object,
    *,
    error_label: str,
) -> tuple[str, str | None]:
    try:
        result = _get_converter().convert(source)
    except Exception as exc:
        raise MarkItDownConversionError(
            f"MarkItDown failed to convert {error_label}: {exc}"
        ) from exc

    markdown = getattr(result, "markdown", None)
    if not isinstance(markdown, str):
        markdown = str(result)
    title = getattr(result, "title", None)
    if title is not None and not isinstance(title, str):
        title = str(title)
    return markdown, title


def _convert_markitdown_with_normalization(path: Path) -> tuple[str, str | None]:
    try:
        return _convert_markitdown(path, error_label=str(path))
    except MarkItDownConversionError as original_exc:
        try:
            normalized_paths = _normalization_candidates(path)
        except MarkItDownConversionError as normalization_exc:
            raise MarkItDownConversionError(
                f"{original_exc}; auto-normalization preparation failed: {normalization_exc}"
            ) from normalization_exc
        if not normalized_paths:
            raise
        errors: list[str] = [str(original_exc)]
        for normalized_path in normalized_paths:
            try:
                return _convert_markitdown(
                    normalized_path,
                    error_label=f"{path} (normalized to {normalized_path.suffix})",
                )
            except MarkItDownConversionError as normalized_exc:
                errors.append(str(normalized_exc))
            finally:
                normalized_path.unlink(missing_ok=True)
        raise MarkItDownConversionError(" | ".join(errors))


@lru_cache(maxsize=1)
def _build_llm_config() -> tuple[object | None, str | None]:
    _load_dotenv_once()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or (
        os.getenv("OPENROUTER_API_KEY") or ""
    ).strip()
    if not api_key:
        return None, None

    base_url = _configured_llm_base_url()
    model = (os.getenv("OPENAI_MODEL") or "").strip() or "google/gemini-3-flash-preview"

    from openai import OpenAI

    raw_client: object = OpenAI(api_key=api_key, base_url=base_url)
    client = _OpenRouterClientProxy(
        raw_client,
        provider=_llm_provider_label(base_url),
        add_openrouter_defaults=_is_openrouter_base_url(base_url),
    )
    return client, model


@lru_cache(maxsize=1)
def _image_text_tools_available() -> bool:
    _load_dotenv_once()
    has_openai_key = bool(
        (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("OPENROUTER_API_KEY") or "").strip()
    )
    has_exiftool = shutil.which("exiftool") is not None or bool(
        (os.getenv("EXIFTOOL_PATH") or "").strip()
    )
    return has_openai_key or has_exiftool


@lru_cache(maxsize=1)
def _get_converter():
    from markitdown import MarkItDown

    logging.getLogger("pdfminer").setLevel(logging.ERROR)

    llm_client, llm_model = _build_llm_config()
    if llm_client is None or llm_model is None:
        return MarkItDown()
    llm_prompt = _image_prompt()
    return MarkItDown(llm_client=llm_client, llm_model=llm_model, llm_prompt=llm_prompt)


def convert_path_to_markdown(
    path: str | Path,
    *,
    refresh_images: bool = False,
    prompt_append: str | None = None,
) -> MarkItDownResult:
    path_obj = Path(path)
    is_image = path_obj.suffix.lower() in _IMAGE_SUFFIXES
    is_video = _is_video_media(suffix=path_obj.suffix.lower())
    is_audio = _is_audio_media(suffix=path_obj.suffix.lower())
    media_md5: str | None = None
    cache_key_payload: dict[str, Any] | None = None
    cache_key = ""

    if is_image:
        media_md5 = _file_md5(path_obj)
        llm_enabled, base_url, model, prompt, exiftool_path = _image_context()
        prompt = _merge_prompt(prompt, prompt_append)

        cache_key_payload = _image_cache_payload(
            media_md5,
            llm_enabled=llm_enabled,
            base_url=base_url,
            model=model,
            prompt=prompt,
            exiftool_path=exiftool_path,
        )
        if not _refresh_images_enabled(refresh_images):
            cache_key, cached = _image_cache_lookup(cache_key_payload)
            if cached:
                return cached

        if not _image_text_tools_available():
            raise MarkItDownConversionError(
                "Image conversion requires either `exiftool` (or EXIFTOOL_PATH) or "
                "OPENAI_API_KEY/OPENROUTER_API_KEY."
            )

    if is_video:
        video_path = path_obj
        cleanup_video = False
        try:
            video_path = _maybe_convert_gif_to_mp4(path_obj)
            cleanup_video = video_path is not path_obj
            markdown = _llm_video_markdown(
                video_path.read_bytes(),
                suffix=video_path.suffix.lower(),
                prompt=_merge_prompt(_video_prompt(video_path), prompt_append),
            )
            return MarkItDownResult(markdown=markdown, title=None)
        except MarkItDownConversionError as exc:
            if _is_non_recoverable_video_error(exc):
                raise
            markdown = _format_auto_generated_video_fallback(video_path)
            return MarkItDownResult(markdown=markdown, title=None)
        finally:
            if cleanup_video:
                video_path.unlink(missing_ok=True)

    if is_audio:
        try:
            markdown = _llm_audio_markdown(
                path_obj.read_bytes(),
                suffix=path_obj.suffix.lower(),
                prompt=_merge_prompt(_audio_prompt(), prompt_append),
            )
            return MarkItDownResult(markdown=markdown, title=None)
        except MarkItDownConversionError:
            pass

    markdown, title = _convert_markitdown_with_normalization(path_obj)

    if is_image and llm_enabled and not _HAS_LLM_DESCRIPTION_RE.search(markdown):
        raise MarkItDownConversionError(
            f"LLM description missing from image conversion of {path_obj}"
        )

    out_markdown = _postprocess_image_markdown(markdown) if is_image else markdown
    out = MarkItDownResult(markdown=out_markdown, title=title)
    if is_image and media_md5 is not None:
        _write_cache_entry(
            cache_key, payload=cache_key_payload, markdown=markdown, title=out.title
        )
    return out


def convert_response_to_markdown(
    response: ResponseLike,
    *,
    refresh_images: bool = False,
    prompt_append: str | None = None,
) -> MarkItDownResult:
    content_type = (
        str(response.headers.get("Content-Type", "")).split(";", 1)[0].strip()
    )
    url_suffix = Path(urlparse(str(response.url)).path).suffix.lower()
    is_image = url_suffix in _IMAGE_SUFFIXES or content_type.startswith("image/")
    is_video = _is_video_media(suffix=url_suffix, content_type=content_type)
    is_audio = _is_audio_media(suffix=url_suffix, content_type=content_type)

    media_md5: str | None = None
    cache_key_payload: dict[str, Any] | None = None
    cache_key = ""

    if is_image:
        llm_enabled, base_url, model, prompt, exiftool_path = _image_context()
        prompt = _merge_prompt(prompt, prompt_append)
        media_md5 = hashlib.md5(response.content).hexdigest()

        cache_key_payload = _image_cache_payload(
            media_md5,
            llm_enabled=llm_enabled,
            base_url=base_url,
            model=model,
            prompt=prompt,
            exiftool_path=exiftool_path,
        )
        if not _refresh_images_enabled(refresh_images):
            cache_key, cached = _image_cache_lookup(cache_key_payload)
            if cached:
                return cached

        if not _image_text_tools_available():
            raise MarkItDownConversionError(
                "Image conversion requires either `exiftool` (or EXIFTOOL_PATH) or "
                "OPENAI_API_KEY/OPENROUTER_API_KEY."
            )

    if is_video:
        temp_suffix = url_suffix or _suffix_from_content_type(content_type) or ".mp4"
        temp_path = _mktemp_output(temp_suffix)
        video_path = temp_path
        cleanup_video = False
        try:
            temp_path.write_bytes(response.content)
            video_path = _maybe_convert_gif_to_mp4(temp_path)
            cleanup_video = video_path is not temp_path
            prompt = _merge_prompt(_video_prompt(video_path), prompt_append)
            remote_url = str(response.url).strip()
            if remote_url and remote_url.startswith(("http://", "https://")):
                try:
                    markdown = _llm_video_markdown(
                        None,
                        suffix=video_path.suffix.lower(),
                        prompt=prompt,
                        video_url=remote_url,
                    )
                    return MarkItDownResult(markdown=markdown, title=None)
                except MarkItDownConversionError as exc:
                    if _is_non_recoverable_video_error(exc):
                        raise
            markdown = _llm_video_markdown(
                video_path.read_bytes(),
                suffix=video_path.suffix.lower(),
                prompt=prompt,
            )
            return MarkItDownResult(markdown=markdown, title=None)
        except MarkItDownConversionError as exc:
            if _is_non_recoverable_video_error(exc):
                raise
            markdown = _format_auto_generated_video_fallback(video_path)
            return MarkItDownResult(markdown=markdown, title=None)
        finally:
            if cleanup_video:
                video_path.unlink(missing_ok=True)
            temp_path.unlink(missing_ok=True)

    if is_audio:
        try:
            suffix = url_suffix or _suffix_from_content_type(content_type) or ".wav"
            markdown = _llm_audio_markdown(
                response.content,
                suffix=suffix.lower(),
                prompt=_merge_prompt(_audio_prompt(), prompt_append),
            )
            return MarkItDownResult(markdown=markdown, title=None)
        except MarkItDownConversionError:
            pass

    try:
        markdown, title = _convert_markitdown(response, error_label=str(response.url))
    except MarkItDownConversionError:
        temp_suffix = url_suffix or _suffix_from_content_type(content_type) or ".bin"
        temp_path = _mktemp_output(temp_suffix)
        try:
            temp_path.write_bytes(response.content)
            markdown, title = _convert_markitdown_with_normalization(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    if is_image and llm_enabled and not _HAS_LLM_DESCRIPTION_RE.search(markdown):
        raise MarkItDownConversionError(
            f"LLM description missing from image conversion of {response.url}"
        )

    out_markdown = _postprocess_image_markdown(markdown) if is_image else markdown
    out = MarkItDownResult(markdown=out_markdown, title=title)
    if is_image and media_md5 is not None:
        _write_cache_entry(
            cache_key, payload=cache_key_payload, markdown=markdown, title=out.title
        )
    return out
