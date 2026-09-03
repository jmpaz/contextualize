from __future__ import annotations

from contextvars import ContextVar, Token
import os

_REFRESH_IMAGES: ContextVar[bool] = ContextVar(
    "contextualize_refresh_images", default=False
)
_REFRESH_VIDEOS: ContextVar[bool] = ContextVar(
    "contextualize_refresh_videos", default=False
)
_REFRESH_AUDIO: ContextVar[bool] = ContextVar(
    "contextualize_refresh_audio", default=False
)
_REFRESH_CACHE: ContextVar[bool] = ContextVar(
    "contextualize_refresh_cache", default=False
)
_CACHE_ONLY: ContextVar[bool] = ContextVar("contextualize_cache_only", default=False)
_SKIP_MEDIA: ContextVar[bool] = ContextVar("contextualize_skip_media", default=False)
_DESCRIBE_MEDIA: ContextVar[bool] = ContextVar(
    "contextualize_describe_media", default=True
)
_VERBOSE_LOGGING: ContextVar[bool] = ContextVar(
    "contextualize_verbose_logging", default=False
)
_PAYLOAD_SPEC_JOBS: ContextVar[int | None] = ContextVar(
    "contextualize_payload_spec_jobs", default=None
)
_PAYLOAD_MEDIA_JOBS: ContextVar[int | None] = ContextVar(
    "contextualize_payload_media_jobs", default=None
)

_TRANSCRIPTION_JOBS: ContextVar[int | None] = ContextVar(
    "contextualize_transcription_jobs", default=None
)
_MEDIA_DOWNLOAD_JOBS: ContextVar[int | None] = ContextVar(
    "contextualize_media_download_jobs", default=None
)

_DEFAULT_PAYLOAD_SPEC_JOBS = 8
_DEFAULT_PAYLOAD_MEDIA_JOBS = 4
_DEFAULT_TRANSCRIPTION_JOBS = 2
_DEFAULT_MEDIA_DOWNLOAD_JOBS = 2
_MAX_PAYLOAD_JOBS = 64


def normalize_payload_jobs(value: int | str | None, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        parsed = value
    else:
        raw = str(value).strip()
        if not raw:
            return default
        try:
            parsed = int(raw)
        except ValueError:
            return default
    if parsed <= 0:
        return default
    return min(parsed, _MAX_PAYLOAD_JOBS)


def _read_positive_int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    return normalize_payload_jobs(raw, default=default)


def get_refresh_images() -> bool:
    return _REFRESH_IMAGES.get()


def set_refresh_images(enabled: bool) -> Token[bool]:
    return _REFRESH_IMAGES.set(bool(enabled))


def reset_refresh_images(token: Token[bool]) -> None:
    _REFRESH_IMAGES.reset(token)


def get_refresh_videos() -> bool:
    return _REFRESH_VIDEOS.get()


def set_refresh_videos(enabled: bool) -> Token[bool]:
    return _REFRESH_VIDEOS.set(bool(enabled))


def reset_refresh_videos(token: Token[bool]) -> None:
    _REFRESH_VIDEOS.reset(token)


def get_refresh_audio() -> bool:
    return _REFRESH_AUDIO.get()


def set_refresh_audio(enabled: bool) -> Token[bool]:
    return _REFRESH_AUDIO.set(bool(enabled))


def reset_refresh_audio(token: Token[bool]) -> None:
    _REFRESH_AUDIO.reset(token)


def get_refresh_media() -> bool:
    return get_refresh_images() or get_refresh_videos() or get_refresh_audio()


def get_refresh_cache() -> bool:
    return _REFRESH_CACHE.get()


def set_refresh_cache(enabled: bool) -> Token[bool]:
    return _REFRESH_CACHE.set(bool(enabled))


def reset_refresh_cache(token: Token[bool]) -> None:
    _REFRESH_CACHE.reset(token)


def get_cache_only() -> bool:
    return _CACHE_ONLY.get()


def set_cache_only(enabled: bool) -> Token[bool]:
    return _CACHE_ONLY.set(bool(enabled))


def reset_cache_only(token: Token[bool]) -> None:
    _CACHE_ONLY.reset(token)


def get_skip_media() -> bool:
    return _SKIP_MEDIA.get()


def set_skip_media(enabled: bool) -> Token[bool]:
    return _SKIP_MEDIA.set(bool(enabled))


def reset_skip_media(token: Token[bool]) -> None:
    _SKIP_MEDIA.reset(token)


def get_describe_media() -> bool:
    return _DESCRIBE_MEDIA.get()


def set_describe_media(enabled: bool) -> Token[bool]:
    return _DESCRIBE_MEDIA.set(bool(enabled))


def reset_describe_media(token: Token[bool]) -> None:
    _DESCRIBE_MEDIA.reset(token)


def get_verbose_logging() -> bool:
    return _VERBOSE_LOGGING.get()


def set_verbose_logging(enabled: bool) -> Token[bool]:
    return _VERBOSE_LOGGING.set(bool(enabled))


def reset_verbose_logging(token: Token[bool]) -> None:
    _VERBOSE_LOGGING.reset(token)


def get_payload_spec_jobs() -> int:
    override = _PAYLOAD_SPEC_JOBS.get()
    if override is not None:
        return normalize_payload_jobs(override, default=_DEFAULT_PAYLOAD_SPEC_JOBS)
    return _read_positive_int_env(
        "CONTEXTUALIZE_PAYLOAD_SPEC_JOBS", _DEFAULT_PAYLOAD_SPEC_JOBS
    )


def set_payload_spec_jobs(value: int | None) -> Token[int | None]:
    normalized = (
        normalize_payload_jobs(value, default=_DEFAULT_PAYLOAD_SPEC_JOBS)
        if value is not None
        else None
    )
    return _PAYLOAD_SPEC_JOBS.set(normalized)


def reset_payload_spec_jobs(token: Token[int | None]) -> None:
    _PAYLOAD_SPEC_JOBS.reset(token)


def get_payload_media_jobs() -> int:
    override = _PAYLOAD_MEDIA_JOBS.get()
    if override is not None:
        return normalize_payload_jobs(override, default=_DEFAULT_PAYLOAD_MEDIA_JOBS)
    return _read_positive_int_env(
        "CONTEXTUALIZE_PAYLOAD_MEDIA_JOBS", _DEFAULT_PAYLOAD_MEDIA_JOBS
    )


def set_payload_media_jobs(value: int | None) -> Token[int | None]:
    normalized = (
        normalize_payload_jobs(value, default=_DEFAULT_PAYLOAD_MEDIA_JOBS)
        if value is not None
        else None
    )
    return _PAYLOAD_MEDIA_JOBS.set(normalized)


def reset_payload_media_jobs(token: Token[int | None]) -> None:
    _PAYLOAD_MEDIA_JOBS.reset(token)


def get_transcription_jobs() -> int:
    override = _TRANSCRIPTION_JOBS.get()
    if override is not None:
        return normalize_payload_jobs(override, default=_DEFAULT_TRANSCRIPTION_JOBS)
    return _read_positive_int_env(
        "CONTEXTUALIZE_TRANSCRIPTION_JOBS", _DEFAULT_TRANSCRIPTION_JOBS
    )


def set_transcription_jobs(value: int | None) -> Token[int | None]:
    normalized = (
        normalize_payload_jobs(value, default=_DEFAULT_TRANSCRIPTION_JOBS)
        if value is not None
        else None
    )
    return _TRANSCRIPTION_JOBS.set(normalized)


def reset_transcription_jobs(token: Token[int | None]) -> None:
    _TRANSCRIPTION_JOBS.reset(token)


def get_media_download_jobs() -> int:
    override = _MEDIA_DOWNLOAD_JOBS.get()
    if override is not None:
        return normalize_payload_jobs(override, default=_DEFAULT_MEDIA_DOWNLOAD_JOBS)
    return _read_positive_int_env(
        "CONTEXTUALIZE_MEDIA_DOWNLOAD_JOBS", _DEFAULT_MEDIA_DOWNLOAD_JOBS
    )


def set_media_download_jobs(value: int | None) -> Token[int | None]:
    normalized = (
        normalize_payload_jobs(value, default=_DEFAULT_MEDIA_DOWNLOAD_JOBS)
        if value is not None
        else None
    )
    return _MEDIA_DOWNLOAD_JOBS.set(normalized)


def reset_media_download_jobs(token: Token[int | None]) -> None:
    _MEDIA_DOWNLOAD_JOBS.reset(token)
