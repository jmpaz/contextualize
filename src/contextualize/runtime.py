from __future__ import annotations

from contextvars import ContextVar, Token

_REFRESH_IMAGES: ContextVar[bool] = ContextVar(
    "contextualize_refresh_images", default=False
)


def get_refresh_images() -> bool:
    return _REFRESH_IMAGES.get()


def set_refresh_images(enabled: bool) -> Token[bool]:
    return _REFRESH_IMAGES.set(bool(enabled))


def reset_refresh_images(token: Token[bool]) -> None:
    _REFRESH_IMAGES.reset(token)
