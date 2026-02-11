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

_DEFAULT_PAYLOAD_SPEC_JOBS = 4
_DEFAULT_PAYLOAD_MEDIA_JOBS = 3
_MAX_PAYLOAD_JOBS = 64


def _read_positive_int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value <= 0:
        return default
    return min(value, _MAX_PAYLOAD_JOBS)


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


def get_payload_spec_jobs() -> int:
    return _read_positive_int_env(
        "CONTEXTUALIZE_PAYLOAD_SPEC_JOBS", _DEFAULT_PAYLOAD_SPEC_JOBS
    )


def get_payload_media_jobs() -> int:
    return _read_positive_int_env(
        "CONTEXTUALIZE_PAYLOAD_MEDIA_JOBS", _DEFAULT_PAYLOAD_MEDIA_JOBS
    )
