from __future__ import annotations

import hashlib
import html
import json
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from contextualize.concurrency import media_task_semaphore, run_indexed_tasks_fail_fast
from contextualize.cache.local_media import (
    get_cached_transcript as get_cached_local_media_text,
)
from contextualize.cache.local_media import (
    store_transcript as store_local_media_text,
)
from contextualize.plugins.api import TranscriptionResult
from contextualize.progress import record_progress
from contextualize.runtime import get_payload_media_jobs

from .audio_transcription import CacheMissError
from .audio_transcription import _video_suffix_from_content_type
from .audio_transcription import transcribe_media_file_result

_VALID_FRAME_MODES = frozenset({"duration", "speech"})


@dataclass(frozen=True)
class VideoFrameSettings:
    frames: bool = True
    frame_mode: str = "duration"
    frame_descriptions: bool = True
    frame_max: int = 10


@dataclass(frozen=True)
class VideoSegment:
    start: float
    end: float | None
    text: str


@dataclass(frozen=True)
class VideoWord:
    start: float
    end: float | None
    text: str


@dataclass(frozen=True)
class VideoFrame:
    index: int
    timestamp: float
    speech: str | None = None
    description: str | None = None


def render_video_file(
    path: str | Path,
    *,
    timeout: float | None = None,
    use_cache: bool = True,
    refresh_cache: bool | None = None,
    plugin_overrides: dict[str, Any] | None = None,
    source_url: str | None = None,
) -> str:
    video_path = Path(path)
    settings = resolve_video_frame_settings(plugin_overrides)
    transcript_result, transcript_error, cache_miss = _transcribe_video_for_context(
        video_path,
        timeout=timeout,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        plugin_overrides=plugin_overrides,
        settings=settings,
    )
    frame_section = render_video_frame_section(
        video_path,
        transcript_result=transcript_result,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        plugin_overrides=plugin_overrides,
        source_url=source_url,
    )
    if not frame_section and transcript_result is None and transcript_error:
        if cache_miss is not None and settings.frames:
            raise cache_miss
        if settings.frames:
            raise RuntimeError(transcript_error)
    return format_video_context(
        frame_section=frame_section,
        transcript_result=transcript_result,
        transcript_error=transcript_error,
        duration_seconds=probe_video_duration(video_path),
        source_url=source_url,
    )


def render_video_bytes(
    data: bytes,
    *,
    filename: str,
    content_type: str | None = None,
    timeout: float | None = None,
    refresh_cache: bool | None = None,
    plugin_overrides: dict[str, Any] | None = None,
    source_url: str | None = None,
) -> str:
    suffix = Path(filename).suffix.lower()
    if not suffix:
        suffix = _video_suffix_from_content_type(content_type) or ".mp4"
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / f"media{suffix}"
        video_path.write_bytes(data)
        return render_video_file(
            video_path,
            timeout=timeout,
            use_cache=False,
            refresh_cache=refresh_cache,
            plugin_overrides=plugin_overrides,
            source_url=source_url,
        )


def render_video_frame_section(
    path: str | Path,
    *,
    transcript_result: TranscriptionResult | None = None,
    use_cache: bool = True,
    refresh_cache: bool | None = None,
    plugin_overrides: dict[str, Any] | None = None,
    source_url: str | None = None,
) -> str:
    video_path = Path(path)
    settings = resolve_video_frame_settings(plugin_overrides)
    if not settings.frames:
        return ""
    duration = probe_video_duration(video_path)
    segments = transcript_segments(transcript_result)
    speech_units = transcript_speech_units(
        transcript_result,
        fallback_segments=segments,
    )
    timestamps = select_frame_timestamps(
        duration_seconds=duration,
        segments=segments,
        speech_units=speech_units,
        settings=settings,
    )
    if not timestamps:
        return ""

    source_sha256 = _sha256_file(video_path)
    source_suffix = video_path.suffix.lower() or ".video"
    cache_identity = _frame_section_cache_identity(
        source_sha256=source_sha256,
        source_suffix=source_suffix,
        settings=settings,
        duration_seconds=duration,
        segments=segments,
        speech_units=speech_units,
    )
    if use_cache and not _should_refresh_video(refresh_cache):
        cached = get_cached_local_media_text(cache_identity)
        if cached:
            record_progress("video", "frame-section", "cache_hit")
            return cached
    record_progress("video", "frame-section", "cache_miss", count=len(timestamps))

    from contextualize.runtime import get_cache_only

    if get_cache_only():
        return ""

    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_tasks = [
            (
                index,
                (
                    lambda idx=index, ts=timestamp: _build_video_frame(
                        video_path,
                        timestamp=ts,
                        index=idx,
                        output_dir=Path(tmpdir),
                        segments=segments,
                        speech_units=speech_units,
                        settings=settings,
                        source_url=source_url,
                    )
                )
            )
            for index, timestamp in enumerate(timestamps, start=1)
        ]
        for _, frame in run_indexed_tasks_fail_fast(
            frame_tasks,
            max_workers=get_payload_media_jobs(),
            semaphore=media_task_semaphore(),
        ):
            frames.append(frame)
    section = format_video_frame_section(frames)
    if use_cache and section.strip():
        store_local_media_text(
            cache_identity,
            section,
            operation="video-frame-section",
            source_sha256=source_sha256,
            source_suffix=source_suffix,
        )
        record_progress("video", "frame-section", "processed", count=len(frames))
    return section


def _build_video_frame(
    video_path: Path,
    *,
    timestamp: float,
    index: int,
    output_dir: Path,
    segments: list[VideoSegment],
    speech_units: list[VideoSegment],
    settings: VideoFrameSettings,
    source_url: str | None,
) -> VideoFrame:
    image_path = extract_video_frame(
        video_path,
        timestamp_seconds=timestamp,
        output_dir=output_dir,
        index=index,
    )
    speech = speech_anchor_for_timestamp(timestamp, segments)
    if speech_units:
        speech = speech_anchor_for_timestamp(timestamp, speech_units)
    description = None
    if settings.frame_descriptions and image_path is not None:
        description = describe_video_frame(
            image_path,
            timestamp_seconds=timestamp,
            speech=speech,
            source_url=source_url,
        )
    return VideoFrame(
        index=index,
        timestamp=timestamp,
        speech=speech,
        description=description,
    )


def resolve_video_frame_settings(
    plugin_overrides: dict[str, Any] | None = None,
) -> VideoFrameSettings:
    settings = VideoFrameSettings()
    env_settings = {
        "frames": _parse_optional_bool(os.environ.get("CONTEXTUALIZE_VIDEO_FRAMES")),
        "frame_mode": _parse_frame_mode(os.environ.get("CONTEXTUALIZE_VIDEO_FRAME_MODE")),
        "frame_descriptions": _parse_optional_bool(
            os.environ.get("CONTEXTUALIZE_VIDEO_FRAME_DESCRIPTIONS")
        ),
        "frame_max": _parse_positive_int(os.environ.get("CONTEXTUALIZE_VIDEO_FRAME_MAX")),
    }
    settings = _apply_settings(settings, env_settings)
    raw = {}
    if isinstance(plugin_overrides, dict) and isinstance(
        plugin_overrides.get("video"), dict
    ):
        raw = dict(plugin_overrides.get("video") or {})
    override_settings = {
        "frames": _coerce_optional_bool(raw.get("frames")),
        "frame_mode": _parse_frame_mode(raw.get("frame-mode", raw.get("frame_mode"))),
        "frame_descriptions": _coerce_optional_bool(
            raw.get("frame-descriptions", raw.get("frame_descriptions"))
        ),
        "frame_max": _parse_positive_int(raw.get("frame-max", raw.get("frame_max"))),
    }
    return _apply_settings(settings, override_settings)


def format_video_context(
    *,
    frame_section: str,
    transcript_result: TranscriptionResult | None,
    transcript_error: str | None,
    duration_seconds: float | None,
    source_url: str | None,
) -> str:
    lines: list[str] = []
    if duration_seconds is not None or source_url:
        lines.append("## Video")
        if duration_seconds is not None:
            lines.append(f"- duration_seconds: {duration_seconds:.3f}")
        if source_url:
            lines.append(f"- source_url: {source_url}")
        lines.append("")
    if frame_section:
        lines.append(frame_section)
        lines.append("")
    if transcript_result is not None and transcript_result.text.strip():
        lines.append("## Transcript")
        lines.append("")
        lines.append(transcript_result.text.strip())
    elif transcript_error and not frame_section:
        lines.append("## Transcript")
        lines.append("")
        lines.append(f"[transcript unavailable: {transcript_error}]")
    return "\n".join(lines).strip()


def select_frame_timestamps(
    *,
    duration_seconds: float | None,
    segments: list[VideoSegment],
    speech_units: list[VideoSegment] | None = None,
    settings: VideoFrameSettings,
) -> list[float]:
    if settings.frame_mode == "speech":
        units = speech_units or segments
        if units:
            return _evenly_limit(
                _bounded_timestamps(
                    [unit.start for unit in units if unit.start >= 0],
                    duration_seconds,
                ),
                settings.frame_max,
            )
    if duration_seconds is None or duration_seconds <= 0:
        if segments:
            return _evenly_limit(
                [segment.start for segment in segments if segment.start >= 0],
                settings.frame_max,
            )
        return []
    count = min(_duration_frame_count(duration_seconds), settings.frame_max)
    if count <= 0:
        return []
    step = duration_seconds / (count + 1)
    return [
        max(0.0, min(duration_seconds, step * index))
        for index in range(1, count + 1)
    ]


def transcript_segments(
    transcript_result: TranscriptionResult | None,
) -> list[VideoSegment]:
    if transcript_result is None:
        return []
    raw_segments = transcript_result.metadata.get("segments")
    if not isinstance(raw_segments, list):
        return []
    segments: list[VideoSegment] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        start = _timestamp_value(raw_segment, "start")
        if start is None:
            continue
        end = _timestamp_value(raw_segment, "end")
        text = raw_segment.get("text")
        text = text.strip() if isinstance(text, str) else ""
        segments.append(VideoSegment(start=start, end=end, text=text))
    return segments


def transcript_words(
    transcript_result: TranscriptionResult | None,
) -> list[VideoWord]:
    if transcript_result is None:
        return []
    raw_words = transcript_result.metadata.get("words")
    if not isinstance(raw_words, list):
        return []
    words: list[VideoWord] = []
    for raw_word in raw_words:
        if not isinstance(raw_word, dict):
            continue
        start = _timestamp_value(raw_word, "start")
        if start is None:
            continue
        text = raw_word.get("word", raw_word.get("text"))
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            continue
        words.append(
            VideoWord(
                start=start,
                end=_timestamp_value(raw_word, "end"),
                text=text,
            )
        )
    return words


def transcript_speech_units(
    transcript_result: TranscriptionResult | None,
    *,
    fallback_segments: list[VideoSegment] | None = None,
) -> list[VideoSegment]:
    words = transcript_words(transcript_result)
    word_units = _sentence_units_from_words(words)
    if word_units:
        return word_units
    segments = (
        fallback_segments
        if fallback_segments is not None
        else transcript_segments(transcript_result)
    )
    sentence_units = _sentence_units_from_segments(segments)
    return sentence_units or segments


def speech_anchor_for_timestamp(
    timestamp_seconds: float,
    segments: list[VideoSegment],
) -> str | None:
    if not segments:
        return None
    selected = None
    for segment in segments:
        if segment.end is not None and segment.start <= timestamp_seconds < segment.end:
            selected = segment
            break
        if segment.start <= timestamp_seconds:
            selected = segment
    if selected is None:
        selected = min(
            segments,
            key=lambda segment: abs(segment.start - timestamp_seconds),
        )
    return _speech_anchor(selected.text)


def format_video_frame_section(frames: list[VideoFrame]) -> str:
    if not frames:
        return ""
    lines = ["## Video Frames", ""]
    for frame in frames:
        attrs = [
            f'frame="{frame.index}"',
            f'timestamp="{_format_timestamp(frame.timestamp)}"',
        ]
        if frame.speech:
            attrs.append(f'speech="{html.escape(frame.speech, quote=True)}"')
        if frame.description:
            lines.append(f"<image {' '.join(attrs)}>")
            lines.append(frame.description.strip())
            lines.append("</image>")
        else:
            lines.append(f"<image {' '.join(attrs)} />")
        lines.append("")
    return "\n".join(lines).strip()


def probe_video_duration(path: str | Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return _as_float(result.stdout.strip())


def extract_video_frame(
    video_path: Path,
    *,
    timestamp_seconds: float,
    output_dir: Path,
    index: int,
) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    output_path = output_dir / f"frame-{index:03d}.jpg"
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp_seconds:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        return None
    return output_path


def describe_video_frame(
    image_path: Path,
    *,
    timestamp_seconds: float,
    speech: str | None,
    source_url: str | None,
) -> str | None:
    from contextualize.runtime import (
        get_refresh_images,
        get_refresh_media,
        get_refresh_videos,
    )
    from contextualize.render.markitdown import convert_path_to_markdown

    prompt_parts = [
        (
            "Describe this sampled video frame in one concise sentence. "
            f"It appears at {_format_timestamp(timestamp_seconds)}."
        )
    ]
    if speech:
        prompt_parts.append(f"Speech near this frame: {speech}")
    try:
        result = convert_path_to_markdown(
            str(image_path),
            refresh_images=get_refresh_images()
            or get_refresh_videos()
            or get_refresh_media(),
            prompt_append="\n".join(prompt_parts),
            source_url=source_url,
        )
    except Exception:
        return None
    return _normalize_image_description(result.markdown or "")


def video_transcription_overrides(
    plugin_overrides: dict[str, Any] | None,
    settings: VideoFrameSettings | None = None,
) -> dict[str, Any] | None:
    settings = settings or resolve_video_frame_settings(plugin_overrides)
    if not settings.frames:
        return plugin_overrides
    merged: dict[str, Any] = {}
    if isinstance(plugin_overrides, dict):
        for key, value in plugin_overrides.items():
            merged[key] = dict(value) if isinstance(value, dict) else value
    transcribe = dict(merged.get("transcribe") or {})
    granularities = transcribe.get("timestamp_granularities") or []
    if isinstance(granularities, str):
        granularities = [granularities]
    elif isinstance(granularities, (tuple, set)):
        granularities = list(granularities)
    elif not isinstance(granularities, list):
        granularities = []
    if "segment" not in granularities:
        granularities.append("segment")
    transcribe["timestamp_granularities"] = granularities
    merged["transcribe"] = transcribe
    return merged


def _transcribe_video_for_context(
    video_path: Path,
    *,
    timeout: float | None,
    use_cache: bool,
    refresh_cache: bool | None,
    plugin_overrides: dict[str, Any] | None,
    settings: VideoFrameSettings,
) -> tuple[TranscriptionResult | None, str | None, CacheMissError | None]:
    try:
        result = transcribe_media_file_result(
            video_path,
            timeout=timeout,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            plugin_overrides=video_transcription_overrides(
                plugin_overrides,
                settings,
            ),
        )
    except CacheMissError as exc:
        return None, str(exc), exc
    except Exception as exc:
        return None, str(exc), None
    return result, None, None


def _frame_section_cache_identity(
    *,
    source_sha256: str,
    source_suffix: str,
    settings: VideoFrameSettings,
    duration_seconds: float | None,
    segments: list[VideoSegment],
    speech_units: list[VideoSegment],
) -> str:
    payload = {
        "version": 2,
        "operation": "video-frame-section",
        "source_sha256": source_sha256,
        "source_suffix": source_suffix,
        "settings": asdict(settings),
        "duration_seconds": round(duration_seconds, 3)
        if duration_seconds is not None
        else None,
        "segments": [
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3) if segment.end is not None else None,
                "text": segment.text,
            }
            for segment in segments
        ],
        "speech_units": [
            {
                "start": round(unit.start, 3),
                "end": round(unit.end, 3) if unit.end is not None else None,
                "text": unit.text,
            }
            for unit in speech_units
        ],
    }
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "video-frame-section:" + hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_settings(
    settings: VideoFrameSettings,
    raw: dict[str, Any],
) -> VideoFrameSettings:
    values = {}
    for key, value in raw.items():
        if value is not None:
            values[key] = value
    if not values:
        return settings
    return replace(settings, **values)


def _parse_optional_bool(value: Any) -> bool | None:
    if not isinstance(value, str):
        return None
    return _coerce_optional_bool(value)


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_frame_mode(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in _VALID_FRAME_MODES else None


def _parse_positive_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _should_refresh_video(refresh_cache: bool | None) -> bool:
    if refresh_cache is not None:
        return bool(refresh_cache)
    from contextualize.runtime import get_refresh_videos

    return get_refresh_videos()


def _timestamp_value(value: dict[str, Any], base: str) -> float | None:
    candidates = (
        base,
        base.lower(),
        base.title(),
        base.upper(),
        f"{base}Time",
        f"{base}_time",
        f"{base}TimeSeconds",
        f"{base}_time_seconds",
    )
    for key in candidates:
        if key in value:
            parsed = _as_float(value.get(key))
            if parsed is not None:
                return parsed
    return None


def _sentence_units_from_words(words: list[VideoWord]) -> list[VideoSegment]:
    units: list[VideoSegment] = []
    current: list[VideoWord] = []
    for word in words:
        current.append(word)
        if _is_sentence_end(word.text):
            units.append(_word_unit(current))
            current = []
    if current:
        units.append(_word_unit(current))
    return units


def _sentence_units_from_segments(segments: list[VideoSegment]) -> list[VideoSegment]:
    units: list[VideoSegment] = []
    for segment in segments:
        sentences = _split_sentences(segment.text)
        if len(sentences) <= 1:
            continue
        if segment.end is None or segment.end <= segment.start:
            units.extend(
                VideoSegment(start=segment.start, end=segment.end, text=sentence)
                for sentence in sentences
            )
            continue
        step = (segment.end - segment.start) / len(sentences)
        units.extend(
            VideoSegment(
                start=segment.start + (index * step),
                end=segment.start + ((index + 1) * step),
                text=sentence,
            )
            for index, sentence in enumerate(sentences)
        )
    return units


def _word_unit(words: list[VideoWord]) -> VideoSegment:
    first = words[0]
    last = words[-1]
    return VideoSegment(
        start=first.start,
        end=last.end,
        text=_join_word_text(word.text for word in words),
    )


def _split_sentences(text: str) -> list[str]:
    parts = [
        match.group(0).strip()
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+[\"')\]]*)?", text)
        if match.group(0).strip()
    ]
    return parts or ([text.strip()] if text.strip() else [])


def _join_word_text(words: Any) -> str:
    text = " ".join(str(word).strip() for word in words if str(word).strip())
    text = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    return text.strip()


def _is_sentence_end(text: str) -> bool:
    return bool(re.search(r"[.!?]+[\"')\]]*$", text.strip()))


def _bounded_timestamps(values: list[float], duration_seconds: float | None) -> list[float]:
    if duration_seconds is None or duration_seconds <= 0:
        return values
    max_timestamp = max(0.0, duration_seconds - 0.001)
    return [max(0.0, min(value, max_timestamp)) for value in values]


def _duration_frame_count(duration_seconds: float) -> int:
    if duration_seconds < 2:
        return 1
    if duration_seconds < 15:
        return 2
    if duration_seconds < 60:
        return 3
    if duration_seconds < 600:
        return 6
    if duration_seconds < 3600:
        return 8
    return 10


def _evenly_limit(values: list[float], limit: int) -> list[float]:
    cleaned = list(dict.fromkeys(round(value, 3) for value in values if value >= 0))
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 1:
        return cleaned[:1]
    last = len(cleaned) - 1
    indexes = [round(index * last / (limit - 1)) for index in range(limit)]
    return [cleaned[index] for index in dict.fromkeys(indexes)]


def _format_timestamp(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        base = f"{minutes:02d}:{secs:02d}"
    return f"{base}.{millis:03d}"


def _speech_anchor(text: str) -> str | None:
    words = [word for word in text.split() if word]
    if not words:
        return None
    if len(words) <= 10:
        return " ".join(words)
    return f"{' '.join(words[:5])} ... {' '.join(words[-5:])}"


def _normalize_image_description(value: str) -> str | None:
    lines = [line.rstrip() for line in value.strip().splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip().lower().startswith("imagesize:"):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip().lower() in {
        "# description (auto-generated):",
        "## description (auto-generated):",
        "# description:",
        "## description:",
    }:
        lines.pop(0)
    text = "\n".join(line for line in lines).strip()
    return text or None


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
