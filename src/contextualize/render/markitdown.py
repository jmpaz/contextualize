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
import sys
import tempfile
import threading
from typing import Any, Mapping, Protocol
from urllib.parse import urlparse

from ..progress import record_progress


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
IMAGE_SUFFIXES = frozenset(
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
_IMAGE_SUFFIXES = IMAGE_SUFFIXES
_IMAGE_CONVERT_SUFFIXES = frozenset(
    {".gif", ".webp", ".heic", ".heif", ".avif", ".tif", ".tiff"}
)
_VIDEO_CONVERT_SUFFIXES = frozenset({".mov", ".avi", ".mkv", ".webm", ".m4v", ".mp4"})
_VIDEO_SUFFIXES = frozenset(
    {
        ".mp4",
        ".mov",
        ".mpeg",
        ".mpg",
        ".webm",
        ".avi",
        ".mkv",
        ".m4v",
        ".gif",
        ".m3u8",
        ".m3u",
    }
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
    "application/vnd.apple.mpegurl": ".m3u8",
    "application/x-mpegurl": ".m3u8",
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
    ".m3u8": "application/vnd.apple.mpegurl",
    ".m3u": "application/vnd.apple.mpegurl",
}
_VIDEO_CONTENT_TYPES = frozenset(
    {"application/vnd.apple.mpegurl", "application/x-mpegurl"}
)
_HLS_SUFFIXES = frozenset({".m3u8", ".m3u"})
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
_PDF_SUFFIXES = frozenset({".pdf"})
_PDF_BATCH_TRANSCRIPTION_PROMPT = (
    "Transcribe these consecutive scanned PDF pages into clean Markdown prose. "
    "The images are supplied in page order and belong to the same document. "
    "Earlier page batches may already be present in this thread; use that "
    "context only to continue the transcription naturally. Return only text "
    "from the supplied images, not earlier pages. Preserve the "
    "author's wording, punctuation, emphasis, true headings, paragraphs, and "
    "footnotes, but repair layout artifacts from the scan. Use logical reading "
    "order, not visual placement. Reflow wrapped lines into paragraphs; do not "
    "preserve source line breaks. Join words split by line-end hyphenation. "
    "Ignore running headers, footers, page numbers, decorative pull quotes, "
    "and repeated text that is not part of the main body. For columns or inset "
    "text, follow the main body reading order and do not let callouts interrupt "
    "paragraphs. Do not summarize or describe the page. Return only the "
    "transcribed text. If no text is readable, return an empty string."
)
_PDFTOPPM_ENV = "PDFTOPPM_PATH"
_PDF_OCR_DPI_ENV = "CONTEXTUALIZE_MD_PDF_OCR_DPI"
_PDF_OCR_TIMEOUT_ENV = "CONTEXTUALIZE_MD_PDF_OCR_TIMEOUT"
_PDF_OCR_BATCH_SIZE_ENV = "CONTEXTUALIZE_MD_PDF_OCR_BATCH_SIZE"
_DEFAULT_PDF_OCR_DPI = 180
_DEFAULT_PDF_OCR_TIMEOUT_SECONDS = 180.0
_DEFAULT_PDF_OCR_BATCH_SIZE = 5
_IMAGE_PROVIDER_MODES = frozenset({"auto", "app-server", "openrouter"})
_DEFAULT_IMAGE_PROVIDER_MODE = "auto"
_DEFAULT_CODEX_APP_SERVER_COMMAND = "codex app-server --listen stdio://"
_DEFAULT_OPENROUTER_MODEL = "google/gemini-3.1-flash-lite"
_DEFAULT_CODEX_APP_SERVER_MODEL = "gpt-5.4"
_DEFAULT_CODEX_APP_SERVER_EFFORT = "medium"
_IMAGE_CACHE_STRICT_ENV = "CONTEXTUALIZE_MD_IMAGE_CACHE_STRICT"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True, slots=True)
class _ImageProviderSelection:
    requested_mode: str
    effective_provider: str
    app_server_live: bool
    app_server_error: str | None


_IMAGE_PROVIDER_SELECTION_CACHE: dict[
    tuple[str, str],
    _ImageProviderSelection,
] = {}
_IMAGE_PROVIDER_SELECTION_LOCK = threading.Lock()


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


def _is_app_server_model_unsupported_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "model" in message and "not supported" in message


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
        print(message, file=sys.stderr)


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


def _pdftoppm_path() -> str | None:
    configured = (os.getenv(_PDFTOPPM_ENV) or "").strip()
    if configured:
        return configured
    return shutil.which("pdftoppm")


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


def _transcode_remote_video_url_to_mp4(source_url: str) -> Path:
    ffmpeg = _ffmpeg_path()
    if ffmpeg is None:
        raise MarkItDownConversionError(
            f"Auto-conversion to MP4 requires ffmpeg for {source_url}"
        )
    output = _mktemp_output(".mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        source_url,
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        "-t",
        "30",
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
            f"Failed to invoke ffmpeg for {source_url}: {exc}"
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "unknown ffmpeg error"
        output.unlink(missing_ok=True)
        raise MarkItDownConversionError(
            f"Failed to auto-convert {source_url} to MP4: {stderr}"
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
    normalized_content_type = content_type.lower()
    return (
        suffix in _VIDEO_SUFFIXES
        or normalized_content_type.startswith("video/")
        or normalized_content_type in _VIDEO_CONTENT_TYPES
    )


def _is_audio_media(*, suffix: str, content_type: str = "") -> bool:
    return suffix in _AUDIO_SUFFIXES or content_type.lower().startswith("audio/")


def _maybe_convert_gif_to_mp4(path: Path) -> Path:
    if path.suffix.lower() != ".gif":
        return path
    return _transcode_video_to_mp4(path)


def _pdf_ocr_dpi() -> int:
    raw = (os.getenv(_PDF_OCR_DPI_ENV) or "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = _DEFAULT_PDF_OCR_DPI
        if 72 <= value <= 300:
            return value
    return _DEFAULT_PDF_OCR_DPI


def _pdf_ocr_timeout_seconds() -> float:
    raw = (os.getenv(_PDF_OCR_TIMEOUT_ENV) or "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = _DEFAULT_PDF_OCR_TIMEOUT_SECONDS
        if value > 0:
            return value
    return _DEFAULT_PDF_OCR_TIMEOUT_SECONDS


def _pdf_ocr_batch_size() -> int:
    raw = (os.getenv(_PDF_OCR_BATCH_SIZE_ENV) or "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = _DEFAULT_PDF_OCR_BATCH_SIZE
        if 1 <= value <= 20:
            return value
    return _DEFAULT_PDF_OCR_BATCH_SIZE


def _has_extractable_pdf_text(markdown: str) -> bool:
    return bool(markdown.strip())


def _rendered_pdf_page_number(path: Path) -> int | None:
    match = re.search(r"-(\d+)\.png$", path.name)
    if not match:
        return None
    return int(match.group(1))


def _render_pdf_pages_to_png(path: Path, output_dir: Path, *, dpi: int) -> list[Path]:
    pdftoppm = _pdftoppm_path()
    if pdftoppm is None:
        raise MarkItDownConversionError(
            f"Scanned PDF OCR fallback requires pdftoppm for {path}"
        )
    prefix = output_dir / "page"
    cmd = [pdftoppm, "-png", "-r", str(dpi), str(path), str(prefix)]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise MarkItDownConversionError(
            f"Failed to invoke pdftoppm for scanned PDF OCR of {path}: {exc}"
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "unknown pdftoppm error"
        raise MarkItDownConversionError(
            f"Failed to render scanned PDF pages for {path}: {stderr}"
        )
    page_paths = sorted(
        output_dir.glob("page-*.png"),
        key=lambda page: _rendered_pdf_page_number(page) or 0,
    )
    if not page_paths:
        raise MarkItDownConversionError(
            f"Scanned PDF OCR rendered no page images for {path}"
        )
    return page_paths


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


def _configured_image_provider_mode() -> str:
    _load_dotenv_once()
    configured = (os.getenv("CONTEXTUALIZE_MD_IMAGE_PROVIDER") or "").strip().lower()
    if configured in _IMAGE_PROVIDER_MODES:
        return configured
    if configured:
        logging.getLogger(__name__).warning(
            "Invalid CONTEXTUALIZE_MD_IMAGE_PROVIDER=%r; using %s.",
            configured,
            _DEFAULT_IMAGE_PROVIDER_MODE,
        )
    return _DEFAULT_IMAGE_PROVIDER_MODE


def _configured_codex_app_server_command() -> str:
    _load_dotenv_once()
    return (
        os.getenv("CODEX_APP_SERVER_COMMAND") or ""
    ).strip() or _DEFAULT_CODEX_APP_SERVER_COMMAND


def _image_cache_strict_enabled() -> bool:
    _load_dotenv_once()
    raw = (os.getenv(_IMAGE_CACHE_STRICT_ENV) or "").strip().lower()
    return raw in _TRUE_VALUES


def _resolve_app_server_request_model(model: str) -> str:
    configured_model = (os.getenv("OPENAI_MODEL") or "").strip()
    provided_model = model.strip()
    if configured_model:
        return configured_model
    if provided_model and provided_model != _DEFAULT_OPENROUTER_MODEL:
        return provided_model
    return _DEFAULT_CODEX_APP_SERVER_MODEL


def _resolve_image_provider(
    requested_mode: str, *, app_server_command: str
) -> _ImageProviderSelection:
    normalized_mode = (
        requested_mode
        if requested_mode in _IMAGE_PROVIDER_MODES
        else _DEFAULT_IMAGE_PROVIDER_MODE
    )
    cache_key = (normalized_mode, app_server_command)
    with _IMAGE_PROVIDER_SELECTION_LOCK:
        cached_selection = _IMAGE_PROVIDER_SELECTION_CACHE.get(cache_key)
    if cached_selection is not None:
        return cached_selection

    if normalized_mode == "openrouter":
        _verbose_log(
            "  image provider selection: mode=openrouter -> provider=openrouter"
        )
        selection = _ImageProviderSelection(
            requested_mode=normalized_mode,
            effective_provider="openrouter",
            app_server_live=False,
            app_server_error=None,
        )
        with _IMAGE_PROVIDER_SELECTION_LOCK:
            _IMAGE_PROVIDER_SELECTION_CACHE[cache_key] = selection
        return selection

    from .codex import is_shared_codex_app_server_live

    _verbose_log(
        "  probing codex app-server for image descriptions: "
        f"mode={normalized_mode} command={app_server_command!r}"
    )
    is_live, error = is_shared_codex_app_server_live(app_server_command)
    if is_live:
        _verbose_log(
            f"  image provider selection: mode={normalized_mode} -> provider=app-server"
        )
        selection = _ImageProviderSelection(
            requested_mode=normalized_mode,
            effective_provider="app-server",
            app_server_live=True,
            app_server_error=None,
        )
        with _IMAGE_PROVIDER_SELECTION_LOCK:
            _IMAGE_PROVIDER_SELECTION_CACHE[cache_key] = selection
        return selection
    _verbose_log(
        "  image provider selection: "
        f"mode={normalized_mode} -> provider=openrouter (app-server unavailable: {error})"
    )
    selection = _ImageProviderSelection(
        requested_mode=normalized_mode,
        effective_provider="openrouter",
        app_server_live=False,
        app_server_error=error,
    )
    with _IMAGE_PROVIDER_SELECTION_LOCK:
        _IMAGE_PROVIDER_SELECTION_CACHE[cache_key] = selection
    return selection


def _app_server_image_text_from_path(
    image_path: Path,
    *,
    prompt: str,
    model: str,
    app_server_command: str,
    timeout_seconds: float | None = None,
) -> str:
    from .codex import (
        CodexAppServerError,
        describe_image_with_shared_codex_app_server,
    )

    requested_model = _resolve_app_server_request_model(model)
    effort = _DEFAULT_CODEX_APP_SERVER_EFFORT
    _verbose_log(
        "  sending to model: "
        "provider=codex-app-server "
        f"model={requested_model} effort={effort} endpoint=turn/start"
    )
    try:
        description_result = describe_image_with_shared_codex_app_server(
            image_path,
            prompt=prompt,
            command=app_server_command,
            model=requested_model,
            effort=effort,
            timeout_seconds=timeout_seconds or 30.0,
        )
    except CodexAppServerError as exc:
        if (
            requested_model != _DEFAULT_CODEX_APP_SERVER_MODEL
            and _is_app_server_model_unsupported_error(exc)
        ):
            _verbose_log(
                "  codex app-server rejected requested model; retrying with default: "
                f"model={_DEFAULT_CODEX_APP_SERVER_MODEL} effort={effort}"
            )
            description_result = describe_image_with_shared_codex_app_server(
                image_path,
                prompt=prompt,
                command=app_server_command,
                model=_DEFAULT_CODEX_APP_SERVER_MODEL,
                effort=effort,
                timeout_seconds=timeout_seconds or 30.0,
            )
        else:
            raise
    record_progress(
        "codex-app-server",
        "image-description",
        "processed",
        target=image_path.name,
    )
    if description_result.rerouted_to_model:
        from_model = description_result.rerouted_from_model or requested_model
        reason = description_result.reroute_reason or "unspecified"
        _verbose_log(
            "  model rerouted: "
            f"provider=codex-app-server from={from_model} "
            f"to={description_result.rerouted_to_model} reason={reason}"
        )
    else:
        effective_model = description_result.requested_model or requested_model
        _verbose_log(
            "  model used: "
            f"provider=codex-app-server model={effective_model} effort={effort}"
        )
    return description_result.text.strip()


def _app_server_image_markdown_from_path(
    image_path: Path, *, prompt: str, model: str, app_server_command: str
) -> str:
    text = _app_server_image_text_from_path(
        image_path,
        prompt=prompt,
        model=model,
        app_server_command=app_server_command,
    )
    return _format_auto_generated_description(text)


def _app_server_image_markdown_from_bytes(
    image_bytes: bytes,
    *,
    suffix: str,
    prompt: str,
    model: str,
    app_server_command: str,
) -> str:
    temp_path = _mktemp_output(suffix)
    try:
        temp_path.write_bytes(image_bytes)
        return _app_server_image_markdown_from_path(
            temp_path,
            prompt=prompt,
            model=model,
            app_server_command=app_server_command,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def _image_context() -> tuple[bool, str, str, str, str | None, str, str]:
    _load_dotenv_once()
    from ..runtime import get_describe_media

    describe = get_describe_media()
    llm_enabled = describe and bool(
        (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("OPENROUTER_API_KEY") or "").strip()
    )
    base_url = _configured_llm_base_url()
    model = (os.getenv("OPENAI_MODEL") or "").strip() or _DEFAULT_OPENROUTER_MODEL
    prompt = _image_prompt()
    exiftool_path = (os.getenv("EXIFTOOL_PATH") or "").strip() or shutil.which(
        "exiftool"
    )
    provider_mode = _configured_image_provider_mode() if describe else "openrouter"
    return (
        llm_enabled,
        base_url,
        model,
        prompt,
        exiftool_path,
        provider_mode,
        _configured_codex_app_server_command(),
    )


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
    provider_mode: str,
    effective_provider: str | None,
    app_server_command: str,
    strict_cache: bool,
    app_server_model: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "v": 3,
        "type": "image",
        "media_md5": media_md5,
        "markitdown_version": _markitdown_version(),
        "llm_enabled": llm_enabled,
        "strict_cache": strict_cache,
        "prompt": prompt if llm_enabled else None,
        "exiftool_path": exiftool_path,
        "description_heading": "auto-generated",
    }
    if not strict_cache:
        return payload
    normalized_mode = (
        provider_mode
        if provider_mode in _IMAGE_PROVIDER_MODES
        else _DEFAULT_IMAGE_PROVIDER_MODE
    )
    if llm_enabled and effective_provider == "app-server":
        provider = "codex-app-server"
        model_for_cache = app_server_model or _resolve_app_server_request_model(model)
    elif llm_enabled:
        provider = base_url
        model_for_cache = model
    else:
        provider = None
        model_for_cache = None
    payload.update(
        {
            "provider_mode": normalized_mode,
            "provider": provider,
            "effective_provider": effective_provider,
            "app_server_command": app_server_command,
            "model": model_for_cache,
        }
    )
    return payload


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
                record_progress("markitdown", "image-cache", "cache_miss")
                return key, None
            title = cached_title if isinstance(cached_title, str) else None
            record_progress("markitdown", "image-cache", "cache_hit")
            return (
                key,
                MarkItDownResult(
                    markdown=_postprocess_image_markdown(cached_markdown),
                    title=title,
                ),
            )
    record_progress("markitdown", "image-cache", "cache_miss")
    return key, None


def _markdown_cache_lookup(
    payload: dict[str, Any],
) -> tuple[str, MarkItDownResult | None]:
    key = _cache_key(payload)
    cached = _read_cache_entry(key)
    if isinstance(cached, dict):
        cached_markdown = cached.get("markdown")
        cached_title = cached.get("title")
        if isinstance(cached_markdown, str):
            return (
                key,
                MarkItDownResult(
                    markdown=cached_markdown,
                    title=cached_title if isinstance(cached_title, str) else None,
                ),
            )
    return key, None


def _pdf_ocr_cache_payload(
    file_md5: str,
    *,
    dpi: int,
    batch_size: int,
    prompt: str,
    base_url: str,
    model: str,
    provider_selection: _ImageProviderSelection,
    app_server_command: str,
) -> dict[str, Any]:
    model_for_cache = model
    if provider_selection.effective_provider == "app-server":
        model_for_cache = _resolve_app_server_request_model(model)
    return {
        "type": "pdf-ocr",
        "file_md5": file_md5,
        "dpi": dpi,
        "batch_size": batch_size,
        "prompt": prompt,
        "provider_mode": provider_selection.requested_mode,
        "effective_provider": provider_selection.effective_provider,
        "provider": base_url,
        "model": model_for_cache,
        "app_server_command": app_server_command,
        "markitdown_version": _markitdown_version(),
        "renderer": "pdftoppm-png-v1",
    }


def _extract_llm_description_text(markdown: str) -> str:
    match = re.search(
        r"(?ms)^# Description(?: \(auto-generated\))?:?\s*(?P<text>.*)$",
        markdown,
    )
    if match:
        return match.group("text").strip()
    return markdown.strip()


def _openrouter_image_text_from_path(image_path: Path, *, prompt: str) -> str:
    llm_client, llm_model = _build_llm_config()
    if llm_client is None or llm_model is None:
        raise MarkItDownConversionError(
            "OpenRouter/OpenAI image OCR requires OPENAI_API_KEY or OPENROUTER_API_KEY"
        )
    from markitdown import MarkItDown

    try:
        result = MarkItDown(
            llm_client=llm_client,
            llm_model=llm_model,
            llm_prompt=prompt,
        ).convert(image_path)
    except Exception as exc:
        raise MarkItDownConversionError(
            f"Image OCR failed for rendered PDF page {image_path.name}: {exc}"
        ) from exc
    markdown = getattr(result, "markdown", None)
    if not isinstance(markdown, str):
        markdown = str(result)
    text = _extract_llm_description_text(markdown)
    if not text:
        raise MarkItDownConversionError(
            f"Image OCR returned no text for rendered PDF page {image_path.name}"
        )
    return text


def _chunks(values: list[Path], size: int) -> list[list[Path]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _page_numbers(page_paths: list[Path]) -> list[int]:
    return [
        page_number
        for page_path in page_paths
        if (page_number := _rendered_pdf_page_number(page_path)) is not None
    ]


def _page_range_label(page_paths: list[Path]) -> str:
    page_numbers = _page_numbers(page_paths)
    if not page_numbers:
        return "unknown pages"
    if len(page_numbers) == 1:
        return f"page {page_numbers[0]}"
    return f"pages {page_numbers[0]}-{page_numbers[-1]}"


def _pdf_batch_prompt(page_paths: list[Path], *, total_pages: int) -> str:
    page_label = _page_range_label(page_paths)
    return (
        f"{_PDF_BATCH_TRANSCRIPTION_PROMPT}\n\n"
        f"These images are {page_label} of {total_pages}."
    )


def _join_pdf_ocr_sections(sections: list[str]) -> str:
    joined: list[str] = []
    for section in (section.strip() for section in sections):
        if not section:
            continue
        if not joined:
            joined.append(section)
            continue
        previous = joined[-1].rstrip()
        current = section.lstrip()
        if _continues_previous_paragraph(previous, current):
            joiner = "" if previous.endswith("-") else " "
            if previous.endswith("-"):
                previous = previous[:-1].rstrip()
            joined[-1] = f"{previous}{joiner}{current}"
        else:
            joined.append(current)
    return "\n\n".join(joined).strip()


def _continues_previous_paragraph(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if "\n\n" in previous[-120:] or current.startswith(("#", "-", "*", ">")):
        return False
    if previous.endswith((".", "!", "?", ":", ";", '"', "'", "”", "’", ")", "]")):
        return False
    return bool(re.match(r"^[a-z,;:)\]'\"]", current))


def _app_server_pdf_batch_texts_from_pages(
    page_paths: list[Path],
    *,
    model: str,
    app_server_command: str,
    per_page_timeout_seconds: float,
    batch_size: int,
) -> list[str]:
    from .codex import (
        CodexAppServerError,
        transcribe_image_batches_with_shared_codex_app_server,
    )

    batches = _chunks(page_paths, batch_size)
    prompts = [
        _pdf_batch_prompt(batch, total_pages=len(page_paths)) for batch in batches
    ]
    requested_model = _resolve_app_server_request_model(model)
    effort = _DEFAULT_CODEX_APP_SERVER_EFFORT
    turn_timeout = per_page_timeout_seconds * max((len(batch) for batch in batches), default=1)
    _verbose_log(
        "  sending to model: "
        "provider=codex-app-server "
        f"model={requested_model} effort={effort} endpoint=turn/start "
        f"batches={len(batches)} batch_size={batch_size}"
    )
    try:
        results = transcribe_image_batches_with_shared_codex_app_server(
            batches,
            prompts=prompts,
            command=app_server_command,
            model=requested_model,
            effort=effort,
            timeout_seconds=turn_timeout,
        )
    except CodexAppServerError as exc:
        if (
            requested_model != _DEFAULT_CODEX_APP_SERVER_MODEL
            and _is_app_server_model_unsupported_error(exc)
        ):
            _verbose_log(
                "  codex app-server rejected requested model; retrying with default: "
                f"model={_DEFAULT_CODEX_APP_SERVER_MODEL} effort={effort}"
            )
            try:
                results = transcribe_image_batches_with_shared_codex_app_server(
                    batches,
                    prompts=prompts,
                    command=app_server_command,
                    model=_DEFAULT_CODEX_APP_SERVER_MODEL,
                    effort=effort,
                    timeout_seconds=turn_timeout,
                )
            except CodexAppServerError as retry_exc:
                raise MarkItDownConversionError(
                    f"Scanned PDF OCR failed for rendered page batch "
                    f"{_page_range_label(batches[0])}: {retry_exc}"
                ) from retry_exc
        else:
            raise MarkItDownConversionError(
                f"Scanned PDF OCR failed for rendered page batch "
                f"{_page_range_label(batches[0])}: {exc}"
            ) from exc

    record_progress("codex-app-server", "pdf-ocr", "processed", count=len(batches))
    for result in results:
        if result.rerouted_to_model:
            from_model = result.rerouted_from_model or requested_model
            reason = result.reroute_reason or "unspecified"
            _verbose_log(
                "  model rerouted: "
                f"provider=codex-app-server from={from_model} "
                f"to={result.rerouted_to_model} reason={reason}"
            )
    return [result.text.strip() for result in results if result.text.strip()]


def _pdf_page_image_text(
    image_path: Path,
    *,
    prompt: str,
    model: str,
    provider_selection: _ImageProviderSelection,
    app_server_command: str,
    timeout_seconds: float,
) -> str:
    try:
        if provider_selection.effective_provider == "app-server":
            return _app_server_image_text_from_path(
                image_path,
                prompt=prompt,
                model=model,
                app_server_command=app_server_command,
                timeout_seconds=timeout_seconds,
            )
        return _openrouter_image_text_from_path(image_path, prompt=prompt)
    except MarkItDownConversionError:
        raise
    except Exception as exc:
        raise MarkItDownConversionError(
            f"Scanned PDF OCR failed for rendered page {image_path.name}: {exc}"
        ) from exc


def _convert_scanned_pdf_to_markdown(
    path: Path,
    *,
    refresh_images: bool = False,
    source_label: str | None = None,
) -> MarkItDownResult:
    label = source_label or str(path)
    (
        llm_enabled,
        base_url,
        model,
        _image_prompt_value,
        _exiftool_path,
        provider_mode,
        app_server_command,
    ) = _image_context()
    provider_selection = _resolve_image_provider(
        provider_mode, app_server_command=app_server_command
    )
    if (
        provider_selection.requested_mode == "app-server"
        and not provider_selection.app_server_live
    ):
        detail = provider_selection.app_server_error or "app-server unavailable"
        raise MarkItDownConversionError(
            "PDF has no embedded text; scanned PDF OCR was configured for "
            f"codex app-server, but app-server is unavailable for {label}: {detail}"
        )
    if (
        provider_selection.requested_mode != "openrouter"
        and not provider_selection.app_server_live
        and provider_selection.app_server_error
    ):
        _verbose_log(
            "  app-server probe failed; falling back to OpenRouter image OCR: "
            f"{provider_selection.app_server_error}"
        )
        logging.getLogger(__name__).warning(
            "Codex app-server unavailable; falling back to OpenRouter image OCR: %s",
            provider_selection.app_server_error,
        )
    if provider_selection.effective_provider != "app-server" and not llm_enabled:
        raise MarkItDownConversionError(
            "PDF has no embedded text; scanned PDF OCR requires a live codex "
            f"app-server or OPENAI_API_KEY/OPENROUTER_API_KEY: {label}"
        )

    dpi = _pdf_ocr_dpi()
    timeout_seconds = _pdf_ocr_timeout_seconds()
    batch_size = _pdf_ocr_batch_size()
    prompt = _PDF_BATCH_TRANSCRIPTION_PROMPT
    cache_payload = _pdf_ocr_cache_payload(
        _file_md5(path),
        dpi=dpi,
        batch_size=batch_size,
        prompt=prompt,
        base_url=base_url,
        model=model,
        provider_selection=provider_selection,
        app_server_command=app_server_command,
    )
    cache_key, cached = _markdown_cache_lookup(cache_payload)
    if cached and not _refresh_images_enabled(refresh_images):
        return cached

    _verbose_log(
        "  rendering scanned PDF pages for OCR: "
        f"renderer=pdftoppm dpi={dpi} timeout={timeout_seconds:g}s "
        f"batch_size={batch_size} path={label}"
    )
    with tempfile.TemporaryDirectory(prefix="contextualize-pdf-ocr-") as tmp_dir:
        page_paths = _render_pdf_pages_to_png(path, Path(tmp_dir), dpi=dpi)
        if provider_selection.effective_provider == "app-server":
            sections = _app_server_pdf_batch_texts_from_pages(
                page_paths,
                model=model,
                app_server_command=app_server_command,
                per_page_timeout_seconds=timeout_seconds,
                batch_size=batch_size,
            )
        else:
            sections = [
                text
                for page_path in page_paths
                if (
                    text := _pdf_page_image_text(
                        page_path,
                        prompt=_pdf_batch_prompt(
                            [page_path], total_pages=len(page_paths)
                        ),
                        model=model,
                        provider_selection=provider_selection,
                        app_server_command=app_server_command,
                        timeout_seconds=timeout_seconds,
                    ).strip()
                )
            ]

    markdown = _join_pdf_ocr_sections(sections)
    if not markdown:
        raise MarkItDownConversionError(
            f"PDF has no embedded text and OCR returned no readable text: {label}"
        )
    result = MarkItDownResult(markdown=markdown, title=None)
    _write_cache_entry(
        cache_key,
        payload=cache_payload,
        markdown=result.markdown,
        title=result.title,
    )
    return result


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
    model = (os.getenv("OPENAI_MODEL") or "").strip() or _DEFAULT_OPENROUTER_MODEL

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
    source_url: str | None = None,
) -> MarkItDownResult:
    path_obj = Path(path)
    is_pdf = path_obj.suffix.lower() in _PDF_SUFFIXES
    is_image = path_obj.suffix.lower() in _IMAGE_SUFFIXES
    is_video = _is_video_media(suffix=path_obj.suffix.lower())
    is_audio = _is_audio_media(suffix=path_obj.suffix.lower())
    media_md5: str | None = None
    cache_key_payload: dict[str, Any] | None = None
    cache_key = ""
    llm_description_required = False

    if is_image:
        media_md5 = _file_md5(path_obj)
        (
            llm_enabled,
            base_url,
            model,
            prompt,
            exiftool_path,
            provider_mode,
            app_server_command,
        ) = _image_context()
        prompt = _merge_prompt(prompt, prompt_append)
        strict_cache = _image_cache_strict_enabled()
        provider_selection: _ImageProviderSelection | None = None
        cache_requires_description = llm_enabled
        app_server_model_for_cache: str | None = None

        if strict_cache:
            provider_selection = _resolve_image_provider(
                provider_mode, app_server_command=app_server_command
            )
            llm_description_required = bool(
                provider_selection.effective_provider == "app-server" or llm_enabled
            )
            cache_requires_description = llm_description_required
            if provider_selection.effective_provider == "app-server":
                app_server_model_for_cache = _resolve_app_server_request_model(model)
            if (
                provider_selection.requested_mode != "openrouter"
                and not provider_selection.app_server_live
                and provider_selection.app_server_error
            ):
                _verbose_log(
                    "  app-server probe failed; falling back to OpenRouter image flow: "
                    f"{provider_selection.app_server_error}"
                )
                logging.getLogger(__name__).warning(
                    "Codex app-server unavailable; falling back to OpenRouter path: %s",
                    provider_selection.app_server_error,
                )

        cache_key_payload = _image_cache_payload(
            media_md5,
            llm_enabled=cache_requires_description,
            base_url=base_url,
            model=model,
            prompt=prompt,
            exiftool_path=exiftool_path,
            provider_mode=(
                provider_selection.requested_mode
                if provider_selection
                else provider_mode
            ),
            effective_provider=(
                provider_selection.effective_provider if provider_selection else None
            ),
            app_server_command=app_server_command,
            strict_cache=strict_cache,
            app_server_model=app_server_model_for_cache,
        )
        cache_key = _cache_key(cache_key_payload)
        if not _refresh_images_enabled(refresh_images):
            cache_key, cached = _image_cache_lookup(cache_key_payload)
            if cached:
                return cached

        if provider_selection is None:
            provider_selection = _resolve_image_provider(
                provider_mode, app_server_command=app_server_command
            )
            llm_description_required = bool(
                provider_selection.effective_provider == "app-server" or llm_enabled
            )
            if (
                provider_selection.requested_mode != "openrouter"
                and not provider_selection.app_server_live
                and provider_selection.app_server_error
            ):
                _verbose_log(
                    "  app-server probe failed; falling back to OpenRouter image flow: "
                    f"{provider_selection.app_server_error}"
                )
                logging.getLogger(__name__).warning(
                    "Codex app-server unavailable; falling back to OpenRouter path: %s",
                    provider_selection.app_server_error,
                )

        if provider_selection.effective_provider == "app-server":
            try:
                markdown = _app_server_image_markdown_from_path(
                    path_obj,
                    prompt=prompt,
                    model=model,
                    app_server_command=app_server_command,
                )
                result = MarkItDownResult(
                    markdown=_postprocess_image_markdown(markdown),
                    title=None,
                )
                _write_cache_entry(
                    cache_key,
                    payload=cache_key_payload,
                    markdown=markdown,
                    title=result.title,
                )
                return result
            except Exception as exc:
                _verbose_log(
                    "  app-server image request failed; retrying with OpenRouter image flow: "
                    f"{exc}"
                )
                logging.getLogger(__name__).warning(
                    "Codex app-server image description failed; falling back to OpenRouter path: %s",
                    exc,
                )
                llm_description_required = llm_enabled
                cache_key_payload = _image_cache_payload(
                    media_md5,
                    llm_enabled=llm_description_required,
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    exiftool_path=exiftool_path,
                    provider_mode=provider_selection.requested_mode,
                    effective_provider="openrouter" if strict_cache else None,
                    app_server_command=app_server_command,
                    strict_cache=strict_cache,
                )
                cache_key = _cache_key(cache_key_payload)
                if not _refresh_images_enabled(refresh_images):
                    cache_key, cached = _image_cache_lookup(cache_key_payload)
                    if cached:
                        return cached

        if not _image_text_tools_available():
            raise MarkItDownConversionError(
                "Image conversion requires either `exiftool` (or EXIFTOOL_PATH), "
                "OPENAI_API_KEY/OPENROUTER_API_KEY, or a live codex app-server."
            )

    if is_video:
        video_path = path_obj
        cleanup_video = False
        hls_transcoded_path: Path | None = None
        try:
            video_path = _maybe_convert_gif_to_mp4(path_obj)
            cleanup_video = video_path is not path_obj
            prompt = _merge_prompt(_video_prompt(video_path), prompt_append)
            remote_url = (source_url or "").strip()
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
            if remote_url and video_path.suffix.lower() in _HLS_SUFFIXES:
                hls_transcoded_path = _transcode_remote_video_url_to_mp4(remote_url)
                markdown = _llm_video_markdown(
                    hls_transcoded_path.read_bytes(),
                    suffix=".mp4",
                    prompt=_merge_prompt(
                        _video_prompt(hls_transcoded_path), prompt_append
                    ),
                )
                return MarkItDownResult(markdown=markdown, title=None)
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
            if hls_transcoded_path is not None:
                hls_transcoded_path.unlink(missing_ok=True)

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
    if is_pdf and not _has_extractable_pdf_text(markdown):
        return _convert_scanned_pdf_to_markdown(
            path_obj,
            refresh_images=refresh_images,
        )

    if (
        is_image
        and llm_description_required
        and not _HAS_LLM_DESCRIPTION_RE.search(markdown)
    ):
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
    is_pdf = url_suffix in _PDF_SUFFIXES or content_type in {
        "application/pdf",
        "application/x-pdf",
    }
    is_image = url_suffix in _IMAGE_SUFFIXES or content_type.startswith("image/")
    is_video = _is_video_media(suffix=url_suffix, content_type=content_type)
    is_audio = _is_audio_media(suffix=url_suffix, content_type=content_type)

    media_md5: str | None = None
    cache_key_payload: dict[str, Any] | None = None
    cache_key = ""
    llm_description_required = False

    if is_image:
        (
            llm_enabled,
            base_url,
            model,
            prompt,
            exiftool_path,
            provider_mode,
            app_server_command,
        ) = _image_context()
        prompt = _merge_prompt(prompt, prompt_append)
        media_md5 = hashlib.md5(response.content).hexdigest()
        strict_cache = _image_cache_strict_enabled()
        provider_selection: _ImageProviderSelection | None = None
        cache_requires_description = llm_enabled
        app_server_model_for_cache: str | None = None

        if strict_cache:
            provider_selection = _resolve_image_provider(
                provider_mode, app_server_command=app_server_command
            )
            llm_description_required = bool(
                provider_selection.effective_provider == "app-server" or llm_enabled
            )
            cache_requires_description = llm_description_required
            if provider_selection.effective_provider == "app-server":
                app_server_model_for_cache = _resolve_app_server_request_model(model)
            if (
                provider_selection.requested_mode != "openrouter"
                and not provider_selection.app_server_live
                and provider_selection.app_server_error
            ):
                _verbose_log(
                    "  app-server probe failed; falling back to OpenRouter image flow: "
                    f"{provider_selection.app_server_error}"
                )
                logging.getLogger(__name__).warning(
                    "Codex app-server unavailable; falling back to OpenRouter path: %s",
                    provider_selection.app_server_error,
                )

        cache_key_payload = _image_cache_payload(
            media_md5,
            llm_enabled=cache_requires_description,
            base_url=base_url,
            model=model,
            prompt=prompt,
            exiftool_path=exiftool_path,
            provider_mode=(
                provider_selection.requested_mode
                if provider_selection
                else provider_mode
            ),
            effective_provider=(
                provider_selection.effective_provider if provider_selection else None
            ),
            app_server_command=app_server_command,
            strict_cache=strict_cache,
            app_server_model=app_server_model_for_cache,
        )
        cache_key = _cache_key(cache_key_payload)
        if not _refresh_images_enabled(refresh_images):
            cache_key, cached = _image_cache_lookup(cache_key_payload)
            if cached:
                return cached

        if provider_selection is None:
            provider_selection = _resolve_image_provider(
                provider_mode, app_server_command=app_server_command
            )
            llm_description_required = bool(
                provider_selection.effective_provider == "app-server" or llm_enabled
            )
            if (
                provider_selection.requested_mode != "openrouter"
                and not provider_selection.app_server_live
                and provider_selection.app_server_error
            ):
                _verbose_log(
                    "  app-server probe failed; falling back to OpenRouter image flow: "
                    f"{provider_selection.app_server_error}"
                )
                logging.getLogger(__name__).warning(
                    "Codex app-server unavailable; falling back to OpenRouter path: %s",
                    provider_selection.app_server_error,
                )

        if provider_selection.effective_provider == "app-server":
            image_suffix = (
                url_suffix or _suffix_from_content_type(content_type) or ".jpg"
            )
            try:
                markdown = _app_server_image_markdown_from_bytes(
                    response.content,
                    suffix=image_suffix,
                    prompt=prompt,
                    model=model,
                    app_server_command=app_server_command,
                )
                result = MarkItDownResult(
                    markdown=_postprocess_image_markdown(markdown),
                    title=None,
                )
                _write_cache_entry(
                    cache_key,
                    payload=cache_key_payload,
                    markdown=markdown,
                    title=result.title,
                )
                return result
            except Exception as exc:
                _verbose_log(
                    "  app-server image request failed; retrying with OpenRouter image flow: "
                    f"{exc}"
                )
                logging.getLogger(__name__).warning(
                    "Codex app-server image description failed; falling back to OpenRouter path: %s",
                    exc,
                )
                llm_description_required = llm_enabled
                cache_key_payload = _image_cache_payload(
                    media_md5,
                    llm_enabled=llm_description_required,
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    exiftool_path=exiftool_path,
                    provider_mode=provider_selection.requested_mode,
                    effective_provider="openrouter" if strict_cache else None,
                    app_server_command=app_server_command,
                    strict_cache=strict_cache,
                )
                cache_key = _cache_key(cache_key_payload)
                if not _refresh_images_enabled(refresh_images):
                    cache_key, cached = _image_cache_lookup(cache_key_payload)
                    if cached:
                        return cached

        if not _image_text_tools_available():
            raise MarkItDownConversionError(
                "Image conversion requires either `exiftool` (or EXIFTOOL_PATH), "
                "OPENAI_API_KEY/OPENROUTER_API_KEY, or a live codex app-server."
            )

    if is_video:
        temp_suffix = url_suffix or _suffix_from_content_type(content_type) or ".mp4"
        temp_path = _mktemp_output(temp_suffix)
        video_path = temp_path
        cleanup_video = False
        hls_transcoded_path: Path | None = None
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
            if remote_url and video_path.suffix.lower() in _HLS_SUFFIXES:
                hls_transcoded_path = _transcode_remote_video_url_to_mp4(remote_url)
                markdown = _llm_video_markdown(
                    hls_transcoded_path.read_bytes(),
                    suffix=".mp4",
                    prompt=_merge_prompt(
                        _video_prompt(hls_transcoded_path), prompt_append
                    ),
                )
                return MarkItDownResult(markdown=markdown, title=None)
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
            if hls_transcoded_path is not None:
                hls_transcoded_path.unlink(missing_ok=True)
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

    if is_pdf and not _has_extractable_pdf_text(markdown):
        temp_path = _mktemp_output(".pdf")
        try:
            temp_path.write_bytes(response.content)
            return _convert_scanned_pdf_to_markdown(
                temp_path,
                refresh_images=refresh_images,
                source_label=str(response.url),
            )
        finally:
            temp_path.unlink(missing_ok=True)

    if (
        is_image
        and llm_description_required
        and not _HAS_LLM_DESCRIPTION_RE.search(markdown)
    ):
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
