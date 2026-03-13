from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from contextualize.cache.local_media import (
    get_cached_gate_decision as get_cached_local_media_gate_decision,
    get_cached_transcript as get_cached_local_media_transcript,
)
from contextualize.cache.local_media import (
    store_gate_decision as store_local_media_gate_decision,
    store_transcript as store_local_media_transcript,
)
from contextualize.plugins import (
    loaded_transcription_gates,
    loaded_transcription_providers,
)
from contextualize.plugins.api import (
    TranscriptionGateDecision,
    TranscriptionProvider,
    TranscriptionProviderAuthError,
    TranscriptionProviderError,
    TranscriptionProviderUnavailableError,
    TranscriptionProviderUnsupportedError,
    TranscriptionRequest,
    TranscriptionResult,
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
_DEFAULT_PROVIDER_PRIORITIES = {"openai": 200, "mistral": 100}

AUDIO_SUFFIXES: frozenset[str] = frozenset(_AUDIO_SUFFIX_TO_MIME)
VIDEO_SUFFIXES: frozenset[str] = _VIDEO_SUFFIXES


def is_audio_suffix(suffix: str) -> bool:
    return suffix.lower() in AUDIO_SUFFIXES


def is_video_suffix(suffix: str) -> bool:
    return suffix.lower() in VIDEO_SUFFIXES


def is_media_suffix(suffix: str) -> bool:
    return is_audio_suffix(suffix) or is_video_suffix(suffix)


def _log(message: str) -> None:
    try:
        from contextualize.runtime import get_verbose_logging

        if get_verbose_logging():
            print(f"[audio-transcription] {message}", file=sys.stderr, flush=True)
    except Exception:
        return


def transcribe_media_file(
    path: str | Path,
    *,
    timeout: float = 600,
    use_cache: bool = True,
    refresh_cache: bool | None = None,
    plugin_overrides: dict[str, Any] | None = None,
) -> str:
    media_path = Path(path)
    suffix = media_path.suffix.lower()
    if is_audio_suffix(suffix):
        return _transcribe_audio_path(
            media_path,
            timeout=timeout,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            plugin_overrides=plugin_overrides,
        ).text
    if is_video_suffix(suffix):
        source_sha256 = _sha256_file(media_path)
        cache_identity = _transcription_cache_identity(
            operation="video-transcription",
            source_sha256=source_sha256,
            source_suffix=suffix,
            provider_identity=_routing_cache_identity(
                filename=f"{media_path.stem or 'media'}.wav",
                content_type="audio/wav",
                plugin_overrides=plugin_overrides,
            ),
        )
        should_refresh = _should_refresh_transcription_cache(
            "video", refresh_cache=refresh_cache
        )
        if use_cache and not should_refresh:
            cached = get_cached_local_media_transcript(cache_identity)
            if cached is not None:
                return cached

        result = _transcribe_video_path(
            media_path,
            timeout=timeout,
            use_cache=False,
            refresh_cache=refresh_cache,
            plugin_overrides=plugin_overrides,
        )
        if use_cache and result.text.strip():
            store_local_media_transcript(
                cache_identity,
                result.text,
                operation="video-transcription",
                source_sha256=source_sha256,
                source_suffix=suffix,
            )
        return result.text
    raise RuntimeError(
        f"Unsupported media suffix for transcription: {suffix or '<none>'}"
    )


def transcribe_media_bytes(
    data: bytes,
    *,
    filename: str,
    content_type: str | None = None,
    timeout: float = 600,
    refresh_cache: bool | None = None,
    plugin_overrides: dict[str, Any] | None = None,
) -> str:
    kind = _infer_media_kind(filename=filename, content_type=content_type)
    if kind == "audio":
        return _transcribe_audio_bytes(
            data,
            filename=filename,
            content_type=content_type or _guess_audio_content_type(filename),
            timeout=timeout,
            plugin_overrides=plugin_overrides,
        ).text
    if kind == "video":
        suffix = Path(filename).suffix.lower()
        if not suffix:
            suffix = _video_suffix_from_content_type(content_type) or ".mp4"
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / f"media{suffix}"
            video_path.write_bytes(data)
            return _transcribe_video_path(
                video_path,
                timeout=timeout,
                use_cache=False,
                refresh_cache=refresh_cache,
                plugin_overrides=plugin_overrides,
                cache_source_path=Path(filename),
            ).text
    raise RuntimeError("Unsupported media payload for transcription")


def transcribe_audio_file(
    path: str | Path,
    *,
    timeout: float = 600,
    use_cache: bool = True,
    refresh_cache: bool | None = None,
    plugin_overrides: dict[str, Any] | None = None,
) -> str:
    return _transcribe_audio_path(
        Path(path),
        timeout=timeout,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        plugin_overrides=plugin_overrides,
    ).text


def transcribe_audio_bytes(
    data: bytes,
    *,
    filename: str,
    content_type: str | None = None,
    timeout: float = 600,
    plugin_overrides: dict[str, Any] | None = None,
) -> str:
    return _transcribe_audio_bytes(
        data,
        filename=filename,
        content_type=content_type or _guess_audio_content_type(filename),
        timeout=timeout,
        plugin_overrides=plugin_overrides,
    ).text


def _transcribe_audio_path(
    audio_path: Path,
    *,
    timeout: float,
    use_cache: bool,
    refresh_cache: bool | None,
    plugin_overrides: dict[str, Any] | None,
) -> TranscriptionResult:
    data = audio_path.read_bytes()
    return _transcribe_audio_bytes(
        data,
        filename=audio_path.name,
        content_type=_guess_audio_content_type(audio_path.name),
        timeout=timeout,
        plugin_overrides=plugin_overrides,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_source_sha256=_sha256_bytes(data),
        cache_source_suffix=audio_path.suffix.lower(),
        cache_operation="audio-transcription",
    )


def _transcribe_video_path(
    video_path: Path,
    *,
    timeout: float,
    use_cache: bool,
    refresh_cache: bool | None,
    plugin_overrides: dict[str, Any] | None,
    cache_source_path: Path | None = None,
) -> TranscriptionResult:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Video transcription requires ffmpeg in PATH")
    source_sha256 = _sha256_file(video_path)
    source_suffix = (cache_source_path or video_path).suffix.lower()
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
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"ffmpeg video audio extraction failed: {detail}")
        if not extracted_path.exists() or extracted_path.stat().st_size == 0:
            raise RuntimeError("Video transcription failed: no audio stream found")
        return _transcribe_audio_bytes(
            extracted_path.read_bytes(),
            filename=f"{(cache_source_path or video_path).stem or 'media'}.wav",
            content_type="audio/wav",
            timeout=timeout,
            plugin_overrides=plugin_overrides,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_source_sha256=source_sha256,
            cache_source_suffix=source_suffix or ".mp4",
            cache_operation="video-transcription",
        )


def _transcribe_audio_bytes(
    data: bytes,
    *,
    filename: str,
    content_type: str,
    timeout: float,
    plugin_overrides: dict[str, Any] | None,
    use_cache: bool = False,
    refresh_cache: bool | None = None,
    cache_source_sha256: str | None = None,
    cache_source_suffix: str | None = None,
    cache_operation: str = "audio-transcription",
) -> TranscriptionResult:
    resolved_config = _resolved_transcription_config(
        data=data,
        filename=filename,
        content_type=content_type,
        timeout=timeout,
        plugin_overrides=plugin_overrides,
    )
    request = _build_request(
        data,
        filename=filename,
        content_type=content_type,
        timeout=timeout,
        plugin_overrides=plugin_overrides,
        resolved_config=resolved_config,
    )
    explicit_provider = _selected_provider_name_from_config(resolved_config)
    providers = _ordered_providers(explicit_provider, resolved_config)
    if explicit_provider and not providers:
        raise RuntimeError(
            f"Requested transcription provider '{explicit_provider}' is not loaded."
        )
    if not providers:
        raise RuntimeError(
            "No transcription providers are loaded. Install a plugin that exposes transcription providers."
        )

    should_refresh = _should_refresh_transcription_cache(
        "video" if cache_operation.startswith("video") else "audio",
        refresh_cache=refresh_cache,
    )
    errors: list[str] = []
    for provider in providers:
        cache_identity = None
        if cache_source_sha256 and cache_source_suffix:
            cache_identity = _transcription_cache_identity(
                operation=cache_operation,
                source_sha256=cache_source_sha256,
                source_suffix=cache_source_suffix,
                provider_identity={
                    "resolved_config": _cacheable_resolved_config(resolved_config),
                    "provider": provider.cache_identity(request),
                },
            )
            if use_cache and not should_refresh:
                cached = get_cached_local_media_transcript(cache_identity)
                if cached is not None:
                    _log(f"transcript cache hit for {filename} via {provider.name}")
                    return TranscriptionResult(
                        text=cached,
                        model=provider.name,
                        provider=provider.name,
                    )
        try:
            result = provider.transcribe(request)
        except (
            TranscriptionProviderUnavailableError,
            TranscriptionProviderUnsupportedError,
            TranscriptionProviderAuthError,
        ) as exc:
            if explicit_provider:
                raise RuntimeError(str(exc)) from exc
            errors.append(str(exc))
            continue
        except TranscriptionProviderError as exc:
            raise RuntimeError(str(exc)) from exc

        if cache_identity and use_cache and result.text.strip():
            store_local_media_transcript(
                cache_identity,
                result.text,
                operation=cache_operation,
                source_sha256=cache_source_sha256,
                source_suffix=cache_source_suffix,
            )
            _log(f"stored transcript cache for {filename} via {provider.name}")
        return result

    if errors:
        raise RuntimeError("; ".join(errors))
    raise RuntimeError("No transcription providers could handle the media input")


def _build_request(
    data: bytes,
    *,
    filename: str,
    content_type: str,
    timeout: float,
    plugin_overrides: dict[str, Any] | None,
    resolved_config: dict[str, Any] | None = None,
) -> TranscriptionRequest:
    config = resolved_config or _resolved_transcription_config(
        data=data,
        filename=filename,
        content_type=content_type,
        timeout=timeout,
        plugin_overrides=plugin_overrides,
    )
    prompt_parts = list(config.get("prompt_parts") or [])
    prompt_parts.extend(_read_prompt_files(config.get("prompt_files") or ()))
    speaker_count = config.get("speakers")
    prompt = _build_prompt(prompt_parts, speaker_count=speaker_count)
    bias_terms = _extract_bias_terms(prompt_parts)
    return TranscriptionRequest(
        data=data,
        filename=filename,
        content_type=content_type,
        timeout=timeout,
        prompt=prompt,
        bias_terms=bias_terms,
        diarize=bool(config.get("diarize", False)),
        speaker_count=speaker_count,
    )


def _normalized_transcription_config(
    plugin_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = {}
    if isinstance(plugin_overrides, dict):
        value = plugin_overrides.get("transcribe")
        if isinstance(value, dict):
            raw = dict(value)

    prompt_parts = [
        item.strip()
        for item in raw.get("prompt_parts", []) or ()
        if isinstance(item, str) and item.strip()
    ]
    prompt_files = [
        item.strip()
        for item in raw.get("prompt_files", []) or ()
        if isinstance(item, str) and item.strip()
    ]
    priorities: dict[str, int] = {}
    raw_priorities = raw.get("priorities")
    if isinstance(raw_priorities, dict):
        for name, value in raw_priorities.items():
            try:
                priorities[str(name)] = int(value)
            except (TypeError, ValueError):
                continue
    speakers = raw.get("speakers")
    if speakers is not None:
        try:
            speakers = int(speakers)
        except (TypeError, ValueError):
            speakers = None
    provider = raw.get("provider")
    explicit_provider = isinstance(provider, str) and bool(provider.strip())
    if not isinstance(provider, str):
        provider = None
    provider = (provider or "auto").strip().lower()
    if provider == "whisper":
        provider = "openai"
    if provider not in {"auto", "openai", "mistral"}:
        provider = "auto"
    return {
        "provider": provider,
        "priorities": priorities,
        "prompt_parts": prompt_parts,
        "prompt_files": prompt_files,
        "diarize": bool(raw.get("diarize", False)),
        "speakers": speakers if speakers and speakers > 0 else None,
        "auto_diarize": bool(raw.get("auto_diarize", False)),
        "auto_diarize_provider": str(
            raw.get("auto_diarize_provider") or "mistral"
        ).strip().lower(),
        "explicit_provider": explicit_provider,
        "explicit_diarize": "diarize" in raw,
        "explicit_speakers": "speakers" in raw,
    }


def _resolved_transcription_config(
    *,
    data: bytes,
    filename: str,
    content_type: str,
    timeout: float,
    plugin_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    config = _normalized_transcription_config(plugin_overrides)
    if not config.get("auto_diarize"):
        return config
    if config.get("explicit_diarize") or config.get("explicit_speakers"):
        return config

    decision = _auto_diarization_decision(
        data=data,
        filename=filename,
        content_type=content_type,
        timeout=timeout,
        config=config,
    )
    if not decision.needs_diarization:
        return config

    if not config.get("explicit_provider") and config.get("provider") == "auto":
        provider = str(config.get("auto_diarize_provider") or "mistral").strip().lower()
        if provider in {"mistral", "local"}:
            config["provider"] = provider

    if config.get("provider") in {"mistral", "local"}:
        config["diarize"] = True
        if decision.speaker_count:
            config["speakers"] = decision.speaker_count
    return config


def _auto_diarization_decision(
    *,
    data: bytes,
    filename: str,
    content_type: str,
    timeout: float,
    config: dict[str, Any],
) -> TranscriptionGateDecision:
    gates = list(loaded_transcription_gates())
    if not gates:
        return TranscriptionGateDecision(
            needs_diarization=False,
            metadata={"reason": "no_transcription_gates_loaded"},
        )

    gate = gates[0]
    source_sha256 = _sha256_bytes(data)
    source_suffix = Path(filename).suffix.lower() or ".bin"
    gate_identity = _transcription_cache_identity(
        operation="auto-diarization-gate",
        source_sha256=source_sha256,
        source_suffix=source_suffix,
        provider_identity=gate.cache_identity(config),
    )
    cached = get_cached_local_media_gate_decision(gate_identity)
    if cached is not None:
        _log(f"auto-diarization gate cache hit for {filename}")
        return TranscriptionGateDecision(
            needs_diarization=bool(cached.get("needs_diarization")),
            speaker_count=_as_positive_int(cached.get("speaker_count")),
            confidence=_as_float(cached.get("confidence")),
            metadata=dict(cached.get("metadata") or {}),
        )

    _log(f"auto-diarization gate cache miss for {filename}")
    decision = gate.analyze(data, filename, content_type, timeout, config)
    store_local_media_gate_decision(
        gate_identity,
        {
            "needs_diarization": decision.needs_diarization,
            "speaker_count": decision.speaker_count,
            "confidence": decision.confidence,
            "metadata": dict(decision.metadata),
        },
        operation="auto-diarization-gate",
        source_sha256=source_sha256,
        source_suffix=source_suffix,
    )
    return decision


def _as_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _selected_provider_name(plugin_overrides: dict[str, Any] | None) -> str | None:
    provider = _normalized_transcription_config(plugin_overrides).get("provider")
    if provider in {None, "auto"}:
        return None
    return provider


def _selected_provider_name_from_config(config: dict[str, Any]) -> str | None:
    provider = config.get("provider")
    if provider in {None, "auto"}:
        return None
    return provider


def _routing_cache_identity(
    *,
    filename: str,
    content_type: str,
    plugin_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    request = _build_request(
        b"",
        filename=filename,
        content_type=content_type,
        timeout=0,
        plugin_overrides=plugin_overrides,
    )
    resolved_config = _resolved_transcription_config(
        data=b"",
        filename=filename,
        content_type=content_type,
        timeout=0,
        plugin_overrides=plugin_overrides,
    )
    explicit_provider = _selected_provider_name_from_config(resolved_config)
    providers = _ordered_providers(explicit_provider, resolved_config)
    return {
        "explicit_provider": explicit_provider,
        "resolved_config": _cacheable_resolved_config(resolved_config),
        "providers": [provider.cache_identity(request) for provider in providers],
    }


def _ordered_providers(
    explicit_provider: str | None,
    config: dict[str, Any],
) -> list[TranscriptionProvider]:
    loaded = list(loaded_transcription_providers())
    if explicit_provider:
        return [provider for provider in loaded if provider.name == explicit_provider]

    priorities = dict(_DEFAULT_PROVIDER_PRIORITIES)
    priorities.update(config.get("priorities") or {})
    loaded.sort(
        key=lambda provider: (
            -int(priorities.get(provider.name, provider.priority)),
            provider.name,
        )
    )
    return loaded


def _cacheable_resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": config.get("provider"),
        "priorities": dict(config.get("priorities") or {}),
        "prompt_parts": list(config.get("prompt_parts") or []),
        "prompt_files": list(config.get("prompt_files") or []),
        "diarize": bool(config.get("diarize", False)),
        "speakers": config.get("speakers"),
        "auto_diarize": bool(config.get("auto_diarize", False)),
        "auto_diarize_provider": config.get("auto_diarize_provider"),
    }


def _read_prompt_files(paths: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    parts: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        text = path.read_text(encoding="utf-8").strip()
        if text:
            parts.append(text)
    return tuple(parts)


def _build_prompt(parts: list[str], *, speaker_count: int | None) -> str:
    prompt_parts = [item for item in parts if item]
    if speaker_count:
        labels = ", ".join(f"Speaker {index}" for index in range(1, speaker_count + 1))
        prompt_parts.append(
            "If multiple speakers are present, preserve dialogue and label turns "
            f"using these speaker names when possible: {labels}."
        )
    return "\n\n".join(prompt_parts)


def _extract_bias_terms(parts: list[str]) -> tuple[str, ...]:
    import yaml

    terms: list[str] = []
    seen: set[str] = set()

    def add_term(value: str) -> None:
        normalized = value.strip().strip("-").strip("\"'")
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        terms.append(normalized)

    def flatten_yaml(value: Any) -> None:
        if isinstance(value, str):
            add_term(value)
            return
        if isinstance(value, dict):
            for item in value.values():
                flatten_yaml(item)
            return
        if isinstance(value, list):
            for item in value:
                flatten_yaml(item)

    for part in parts:
        if not part.strip():
            continue
        try:
            parsed = yaml.safe_load(part)
        except Exception:
            parsed = None
        if parsed not in (None, ""):
            flatten_yaml(parsed)
        else:
            for item in part.replace(";", "\n").replace(",", "\n").splitlines():
                add_term(item)
        if len(terms) >= 100:
            return tuple(terms[:100])
    return tuple(terms)


def _guess_audio_content_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return _AUDIO_SUFFIX_TO_MIME.get(suffix, "audio/mpeg")


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
    provider_identity: dict[str, Any],
) -> str:
    payload = {
        "operation": operation,
        "source_sha256": source_sha256,
        "source_suffix": source_suffix,
        "provider_identity": provider_identity,
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
