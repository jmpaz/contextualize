from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, quote, urlparse

from ..render.text import process_text
from ..utils import count_tokens

_SOUNDCLOUD_URN_RE = re.compile(
    r"^soundcloud:(?P<kind>tracks|playlists|users):(?P<id>[^/\s?#]+)$",
    flags=re.IGNORECASE,
)
_SOUNDCLOUD_HOSTS = frozenset(
    {
        "soundcloud.com",
        "www.soundcloud.com",
        "on.soundcloud.com",
        "api.soundcloud.com",
    }
)
_SOUNDCLOUD_RESERVED_USERS = frozenset(
    {
        "discover",
        "stream",
        "upload",
        "charts",
        "people",
        "you",
        "settings",
        "search",
        "signup",
        "login",
        "terms-of-use",
    }
)
_VALID_ARTIST_FILTER_MODES = frozenset({"include", "exclude", "only"})
_VALID_COMMENT_SORTS = frozenset(
    {"created-desc", "created-asc", "track-desc", "track-asc"}
)
_DEFAULT_API_BASE = "https://api.soundcloud.com"
_DEFAULT_OAUTH_BASE = "https://secure.soundcloud.com"
_DOWNLOAD_HEADERS = {
    "User-Agent": "contextualize/soundcloud",
    "Accept": "image/*,application/octet-stream,*/*;q=0.8",
}


def _log(message: str) -> None:
    from ..runtime import get_verbose_logging

    if get_verbose_logging():
        print(message, file=sys.stderr, flush=True)


@lru_cache(maxsize=1)
def _load_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except Exception:
        return


@lru_cache(maxsize=1)
def warmup_soundcloud_network_stack() -> None:
    try:
        import requests  # noqa: F401
    except Exception:
        return


def _parse_bool(value: str, *, default: bool) -> bool:
    cleaned = value.strip().lower()
    if not cleaned:
        return default
    return cleaned not in {"0", "false", "no", "off"}


def _parse_positive_int(value: str, *, default: int, minimum: int = 1) -> int:
    cleaned = value.strip()
    if not cleaned:
        return default
    try:
        parsed = int(cleaned)
    except ValueError:
        return default
    if parsed < minimum:
        return default
    return parsed


def _parse_max_items_env(value: str, *, default: int) -> int | None:
    cleaned = value.strip().lower()
    if not cleaned:
        return default
    if cleaned == "all":
        return None
    try:
        parsed = int(cleaned)
    except ValueError:
        return default
    if parsed < 1:
        return default
    return parsed


def _normalize_optional_int_override(
    value: Any,
    *,
    default: int | None,
    field: str,
    minimum: int = 1,
    allow_all: bool = False,
) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        if value < minimum:
            raise ValueError(f"{field} must be >= {minimum}")
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return default
        if allow_all and cleaned == "all":
            return None
        try:
            parsed = int(cleaned)
        except ValueError as exc:
            if allow_all:
                raise ValueError(
                    f"{field} must be an integer >= {minimum} or 'all'"
                ) from exc
            raise ValueError(f"{field} must be an integer >= {minimum}") from exc
        if parsed < minimum:
            raise ValueError(f"{field} must be >= {minimum}")
        return parsed
    if allow_all:
        raise ValueError(f"{field} must be an integer >= {minimum} or 'all'")
    raise ValueError(f"{field} must be an integer >= {minimum}")


def _normalize_artist_filter_override(
    value: Any,
    *,
    default: Literal["include", "exclude", "only"],
    field: str,
) -> Literal["include", "exclude", "only"]:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{field} must be one of: include, exclude, only")
    cleaned = value.strip().lower()
    if cleaned not in _VALID_ARTIST_FILTER_MODES:
        raise ValueError(f"{field} must be one of: include, exclude, only")
    return cleaned  # type: ignore[return-value]


def _normalize_comment_sort_override(
    value: Any,
    *,
    default: str,
    field: str,
) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be one of: {', '.join(sorted(_VALID_COMMENT_SORTS))}"
        )
    cleaned = value.strip().lower()
    if cleaned not in _VALID_COMMENT_SORTS:
        raise ValueError(
            f"{field} must be one of: {', '.join(sorted(_VALID_COMMENT_SORTS))}"
        )
    return cleaned


def _validate_artist_filters(
    *,
    tracks_filter: Literal["include", "exclude", "only"],
    playlists_filter: Literal["include", "exclude", "only"],
    reposts_filter: Literal["include", "exclude", "only"],
) -> None:
    only_modes = [
        mode
        for mode in (tracks_filter, playlists_filter, reposts_filter)
        if mode == "only"
    ]
    if len(only_modes) > 1:
        raise ValueError(
            "At most one of artist_tracks_filter, artist_playlists_filter, artist_reposts_filter can be 'only'"
        )


@dataclass(frozen=True)
class SoundCloudSettings:
    max_items: int | None = 25
    artist_tracks_filter: Literal["include", "exclude", "only"] = "include"
    artist_playlists_filter: Literal["include", "exclude", "only"] = "include"
    artist_reposts_filter: Literal["include", "exclude", "only"] = "exclude"
    include_comments: bool = True
    comments_max_items: int = 15
    comments_sort: str = "created-desc"
    comments_nesting_depth: int = 1
    include_artwork_descriptions: bool = True
    media_mode: str = "describe"


@dataclass(frozen=True)
class SoundCloudTarget:
    kind: Literal["track", "playlist", "artist", "resolve"]
    original: str
    urn: str | None = None
    permalink_url: str | None = None
    username: str | None = None
    resource_id: str | None = None
    secret_token: str | None = None


@dataclass(frozen=True)
class SoundCloudDocument:
    source_url: str
    kind: str
    urn: str
    label: str
    trace_path: str
    context_subpath: str
    rendered: str
    source_created: str | None = None
    source_modified: str | None = None
    scope_id: str | None = None


@dataclass(frozen=True)
class _AuthContext:
    access_token: str | None
    has_user_token: bool

    @property
    def auth_mode(self) -> str:
        if self.has_user_token and self.access_token:
            return "user-token"
        if self.access_token:
            return "app-token"
        return "public"


def _soundcloud_settings_from_env() -> SoundCloudSettings:
    _load_dotenv()
    max_items = _parse_max_items_env(
        os.environ.get("SOUNDCLOUD_MAX_ITEMS", ""), default=25
    )
    tracks_filter = _normalize_artist_filter_override(
        os.environ.get("SOUNDCLOUD_ARTIST_TRACKS", "").strip().lower() or None,
        default="include",
        field="SOUNDCLOUD_ARTIST_TRACKS",
    )
    playlists_filter = _normalize_artist_filter_override(
        os.environ.get("SOUNDCLOUD_ARTIST_PLAYLISTS", "").strip().lower() or None,
        default="include",
        field="SOUNDCLOUD_ARTIST_PLAYLISTS",
    )
    reposts_filter = _normalize_artist_filter_override(
        os.environ.get("SOUNDCLOUD_ARTIST_REPOSTS", "").strip().lower() or None,
        default="exclude",
        field="SOUNDCLOUD_ARTIST_REPOSTS",
    )
    _validate_artist_filters(
        tracks_filter=tracks_filter,
        playlists_filter=playlists_filter,
        reposts_filter=reposts_filter,
    )

    include_comments = _parse_bool(
        os.environ.get("SOUNDCLOUD_COMMENTS", "1"),
        default=True,
    )
    comments_max_items = _parse_positive_int(
        os.environ.get("SOUNDCLOUD_COMMENTS_MAX", "15"),
        default=15,
        minimum=1,
    )
    comments_sort = _normalize_comment_sort_override(
        os.environ.get("SOUNDCLOUD_COMMENTS_SORT", "").strip().lower() or None,
        default="created-desc",
        field="SOUNDCLOUD_COMMENTS_SORT",
    )
    comments_nesting_depth = _parse_positive_int(
        os.environ.get("SOUNDCLOUD_COMMENTS_NESTING_DEPTH", "1"),
        default=1,
        minimum=0,
    )
    if comments_nesting_depth > 1:
        comments_nesting_depth = 1

    include_artwork_descriptions = _parse_bool(
        os.environ.get("SOUNDCLOUD_ARTWORK_DESCRIPTIONS", "1"),
        default=True,
    )
    media_mode = (os.environ.get("SOUNDCLOUD_MEDIA_MODE") or "").strip().lower()
    if media_mode not in {"", "describe"}:
        media_mode = "describe"

    return SoundCloudSettings(
        max_items=max_items,
        artist_tracks_filter=tracks_filter,
        artist_playlists_filter=playlists_filter,
        artist_reposts_filter=reposts_filter,
        include_comments=include_comments,
        comments_max_items=comments_max_items,
        comments_sort=comments_sort,
        comments_nesting_depth=comments_nesting_depth,
        include_artwork_descriptions=include_artwork_descriptions,
        media_mode=media_mode or "describe",
    )


def build_soundcloud_settings(
    overrides: dict[str, Any] | None = None,
) -> SoundCloudSettings:
    env = _soundcloud_settings_from_env()
    if not overrides:
        return env

    max_items = _normalize_optional_int_override(
        overrides.get("max_items", env.max_items),
        default=env.max_items,
        field="max_items",
        minimum=1,
        allow_all=True,
    )
    tracks_filter = _normalize_artist_filter_override(
        overrides.get("artist_tracks_filter", env.artist_tracks_filter),
        default=env.artist_tracks_filter,
        field="artist_tracks_filter",
    )
    playlists_filter = _normalize_artist_filter_override(
        overrides.get("artist_playlists_filter", env.artist_playlists_filter),
        default=env.artist_playlists_filter,
        field="artist_playlists_filter",
    )
    reposts_filter = _normalize_artist_filter_override(
        overrides.get("artist_reposts_filter", env.artist_reposts_filter),
        default=env.artist_reposts_filter,
        field="artist_reposts_filter",
    )
    _validate_artist_filters(
        tracks_filter=tracks_filter,
        playlists_filter=playlists_filter,
        reposts_filter=reposts_filter,
    )

    include_comments = (
        overrides.get("include_comments")
        if isinstance(overrides.get("include_comments"), bool)
        else env.include_comments
    )
    if not isinstance(include_comments, bool):
        raise ValueError("include_comments must be a boolean")

    comments_max_items = _normalize_optional_int_override(
        overrides.get("comments_max_items", env.comments_max_items),
        default=env.comments_max_items,
        field="comments_max_items",
        minimum=1,
    )
    if comments_max_items is None:
        comments_max_items = env.comments_max_items

    comments_sort = _normalize_comment_sort_override(
        overrides.get("comments_sort", env.comments_sort),
        default=env.comments_sort,
        field="comments_sort",
    )

    comments_nesting_depth = _normalize_optional_int_override(
        overrides.get("comments_nesting_depth", env.comments_nesting_depth),
        default=env.comments_nesting_depth,
        field="comments_nesting_depth",
        minimum=0,
    )
    if comments_nesting_depth is None:
        comments_nesting_depth = env.comments_nesting_depth
    comments_nesting_depth = min(1, comments_nesting_depth)

    include_artwork_descriptions = (
        overrides.get("include_artwork_descriptions")
        if isinstance(overrides.get("include_artwork_descriptions"), bool)
        else env.include_artwork_descriptions
    )
    if not isinstance(include_artwork_descriptions, bool):
        raise ValueError("include_artwork_descriptions must be a boolean")

    media_mode = overrides.get("media_mode", env.media_mode)
    if not isinstance(media_mode, str) or media_mode.strip().lower() not in {
        "describe"
    }:
        raise ValueError("media_mode must be 'describe'")
    media_mode = media_mode.strip().lower()

    return SoundCloudSettings(
        max_items=max_items,
        artist_tracks_filter=tracks_filter,
        artist_playlists_filter=playlists_filter,
        artist_reposts_filter=reposts_filter,
        include_comments=include_comments,
        comments_max_items=int(comments_max_items),
        comments_sort=comments_sort,
        comments_nesting_depth=int(comments_nesting_depth),
        include_artwork_descriptions=include_artwork_descriptions,
        media_mode=media_mode,
    )


def soundcloud_settings_cache_key(settings: SoundCloudSettings) -> tuple[Any, ...]:
    return (
        "v1",
        "all" if settings.max_items is None else settings.max_items,
        settings.artist_tracks_filter,
        settings.artist_playlists_filter,
        settings.artist_reposts_filter,
        settings.include_comments,
        settings.comments_max_items,
        settings.comments_sort,
        settings.comments_nesting_depth,
        settings.include_artwork_descriptions,
        settings.media_mode,
    )


def is_soundcloud_urn(value: str) -> bool:
    return bool(_SOUNDCLOUD_URN_RE.match(value.strip()))


def is_soundcloud_url(value: str) -> bool:
    return parse_soundcloud_target(value) is not None


def parse_soundcloud_target(value: str) -> SoundCloudTarget | None:
    raw = value.strip()
    if not raw:
        return None

    urn_match = _SOUNDCLOUD_URN_RE.match(raw)
    if urn_match:
        kind = urn_match.group("kind").lower()
        urn_id = urn_match.group("id").strip()
        resource_id = urn_id if urn_id.isdigit() else None
        if kind == "tracks":
            return SoundCloudTarget(
                kind="track",
                original=value,
                urn=raw,
                resource_id=resource_id,
            )
        if kind == "playlists":
            return SoundCloudTarget(
                kind="playlist",
                original=value,
                urn=raw,
                resource_id=resource_id,
            )
        return SoundCloudTarget(
            kind="artist",
            original=value,
            urn=raw,
            resource_id=resource_id,
        )

    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    if host not in _SOUNDCLOUD_HOSTS:
        return None

    path_parts = [part for part in parsed.path.split("/") if part]
    if host == "api.soundcloud.com":
        parsed_qs = parse_qs(parsed.query, keep_blank_values=False)
        token_values = parsed_qs.get("secret_token")
        secret_token = None
        if token_values:
            token = token_values[0].strip()
            secret_token = token if token else None
        if len(path_parts) >= 2 and path_parts[0] in {"tracks", "playlists", "users"}:
            urn_or_id = path_parts[1]
            resource_id = (
                urn_or_id
                if urn_or_id.isdigit()
                else urn_or_id.split(":")[-1]
                if urn_or_id.split(":")[-1].isdigit()
                else None
            )
            if path_parts[0] == "tracks":
                urn = (
                    urn_or_id
                    if urn_or_id.startswith("soundcloud:tracks:")
                    else f"soundcloud:tracks:{urn_or_id}"
                )
                return SoundCloudTarget(
                    kind="track",
                    original=value,
                    urn=urn,
                    resource_id=resource_id,
                )
            if path_parts[0] == "playlists":
                urn = (
                    urn_or_id
                    if urn_or_id.startswith("soundcloud:playlists:")
                    else f"soundcloud:playlists:{urn_or_id}"
                )
                return SoundCloudTarget(
                    kind="playlist",
                    original=value,
                    urn=urn,
                    resource_id=resource_id,
                    secret_token=secret_token,
                )
            urn = (
                urn_or_id
                if urn_or_id.startswith("soundcloud:users:")
                else f"soundcloud:users:{urn_or_id}"
            )
            return SoundCloudTarget(
                kind="artist",
                original=value,
                urn=urn,
                resource_id=resource_id,
            )
        return SoundCloudTarget(kind="resolve", original=value)

    if host == "on.soundcloud.com":
        return SoundCloudTarget(kind="resolve", original=value, permalink_url=raw)

    if not path_parts:
        return None
    first = path_parts[0].lower()
    if first in _SOUNDCLOUD_RESERVED_USERS:
        return None
    if len(path_parts) == 1:
        return SoundCloudTarget(kind="artist", original=value, username=path_parts[0])
    if len(path_parts) >= 3 and path_parts[1].lower() == "sets":
        secret_token = None
        if len(path_parts) >= 4 and path_parts[3].startswith("s-"):
            secret_token = path_parts[3]
        if not secret_token:
            parsed_qs = parse_qs(parsed.query, keep_blank_values=False)
            token_values = parsed_qs.get("secret_token")
            if token_values:
                token = token_values[0].strip()
                secret_token = token if token else None
        return SoundCloudTarget(
            kind="playlist",
            original=value,
            permalink_url=raw,
            secret_token=secret_token,
        )
    return SoundCloudTarget(kind="track", original=value, permalink_url=raw)


def _api_timeout_seconds() -> float:
    raw = (os.environ.get("SOUNDCLOUD_API_TIMEOUT") or "").strip()
    if not raw:
        return 30.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def _auth_identity(client_id: str) -> str:
    return f"soundcloud-client:{client_id.strip()}"


def _exchange_client_credentials_token(
    *,
    client_id: str,
    client_secret: str,
) -> tuple[str | None, int]:
    import requests

    basic = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode(
        "ascii"
    )
    headers = {
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json; charset=utf-8",
    }
    url = f"{_DEFAULT_OAUTH_BASE}/oauth/token"
    response = requests.post(
        url,
        data={"grant_type": "client_credentials"},
        headers=headers,
        timeout=_api_timeout_seconds(),
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    expires_in = payload.get("expires_in")
    if not isinstance(token, str) or not token.strip():
        return None, 0
    if not isinstance(expires_in, int):
        expires_in = 3600
    return token.strip(), max(60, expires_in)


def _build_auth_context() -> _AuthContext:
    _load_dotenv()
    token = (os.environ.get("SOUNDCLOUD_ACCESS_TOKEN") or "").strip()
    if token:
        return _AuthContext(access_token=token, has_user_token=True)

    client_id = (os.environ.get("SOUNDCLOUD_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("SOUNDCLOUD_CLIENT_SECRET") or "").strip()
    if not client_id or not client_secret:
        return _AuthContext(access_token=None, has_user_token=False)

    from ..cache.soundcloud import get_cached_client_token, store_client_token

    identity = _auth_identity(client_id)
    cached = get_cached_client_token(identity)
    if cached:
        return _AuthContext(access_token=cached, has_user_token=False)

    try:
        token_value, expires_in = _exchange_client_credentials_token(
            client_id=client_id,
            client_secret=client_secret,
        )
    except Exception as exc:
        _log(f"  soundcloud client_credentials exchange failed: {exc}")
        return _AuthContext(access_token=None, has_user_token=False)

    if not token_value:
        return _AuthContext(access_token=None, has_user_token=False)
    store_client_token(
        identity,
        access_token=token_value,
        expires_in_seconds=expires_in,
    )
    return _AuthContext(access_token=token_value, has_user_token=False)


def _build_headers(
    auth: _AuthContext,
    *,
    bearer: bool,
) -> dict[str, str]:
    headers = {
        "Accept": "application/json; charset=utf-8",
        "User-Agent": "contextualize/soundcloud",
    }
    if auth.access_token:
        scheme = "Bearer" if bearer else "OAuth"
        headers["Authorization"] = f"{scheme} {auth.access_token}"
    return headers


def _request_json(
    url_or_path: str,
    *,
    auth: _AuthContext,
    params: dict[str, Any] | None = None,
) -> Any:
    import requests

    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        url = url_or_path
    else:
        path = url_or_path if url_or_path.startswith("/") else f"/{url_or_path}"
        url = f"{_DEFAULT_API_BASE}{path}"

    timeout = _api_timeout_seconds()
    if auth.access_token:
        for attempt in (False, True):
            response = requests.get(
                url,
                params=params,
                headers=_build_headers(auth, bearer=attempt),
                timeout=timeout,
            )
            if response.status_code != 401:
                response.raise_for_status()
                return response.json()
        raise ValueError(f"SoundCloud API unauthorized for {url}")

    response = requests.get(
        url, params=params, headers=_build_headers(auth, bearer=True), timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def _api_get(
    path_or_url: str,
    *,
    auth: _AuthContext,
    params: dict[str, Any] | None = None,
) -> Any:
    try:
        return _request_json(path_or_url, auth=auth, params=params)
    except Exception as exc:
        raise ValueError(f"SoundCloud request failed for {path_or_url}: {exc}") from exc


def _resolve_cache_identity(
    url: str,
    settings: SoundCloudSettings,
    auth: _AuthContext,
) -> str:
    payload = {
        "v": 1,
        "url": url,
        "settings": soundcloud_settings_cache_key(settings),
        "auth_mode": auth.auth_mode,
    }
    return "soundcloud-resolve:" + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )


def _documents_from_cached_payload(payload: Any) -> list[SoundCloudDocument] | None:
    if not isinstance(payload, list):
        return None
    docs: list[SoundCloudDocument] = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        try:
            docs.append(SoundCloudDocument(**item))
        except TypeError:
            return None
    return docs


def _normalize_urn(kind: str, value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith(f"soundcloud:{kind}:"):
        return cleaned
    return f"soundcloud:{kind}:{cleaned}"


def _extract_secret_token_from_url(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlparse(value.strip())
    parsed_qs = parse_qs(parsed.query, keep_blank_values=False)
    token_values = parsed_qs.get("secret_token")
    if token_values:
        candidate = token_values[0].strip()
        if candidate:
            return candidate
    path_parts = [part for part in parsed.path.split("/") if part]
    for part in reversed(path_parts):
        if part.startswith("s-"):
            return part
    return None


def _extract_resource_id(value: Any) -> str | None:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip().isdigit():
        return value.strip()
    return None


def _extract_target_secret_token(
    *,
    payload: dict[str, Any],
    parsed: SoundCloudTarget,
    raw: str,
) -> str | None:
    direct = payload.get("secret_token")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    for key in ("uri", "permalink_url", "secret_uri"):
        token = _extract_secret_token_from_url(payload.get(key))
        if token:
            return token

    if parsed.secret_token:
        return parsed.secret_token
    return _extract_secret_token_from_url(raw)


def _resource_id_from_payload_or_urn(
    payload: dict[str, Any], urn: str | None
) -> str | None:
    from_payload = _extract_resource_id(payload.get("id"))
    if from_payload:
        return from_payload
    if isinstance(urn, str):
        leaf = urn.split(":")[-1].strip()
        if leaf.isdigit():
            return leaf
    return None


def _resolve_target(
    raw: str,
    *,
    parsed: SoundCloudTarget,
    auth: _AuthContext,
) -> SoundCloudTarget:
    if parsed.kind in {"track", "playlist", "artist"} and parsed.urn:
        return parsed

    if parsed.kind == "resolve" or parsed.permalink_url:
        payload = _api_get(
            "/resolve",
            auth=auth,
            params={"url": parsed.permalink_url or raw},
        )
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected /resolve payload for {raw}")
        kind = str(payload.get("kind") or "").strip().lower()
        urn = payload.get("urn")
        permalink = payload.get("permalink_url")
        username = payload.get("username")
        if kind == "track" and isinstance(urn, str):
            normalized_urn = (
                _normalize_urn("tracks", urn.split(":")[-1])
                if not urn.startswith("soundcloud:tracks:")
                else urn
            )
            return SoundCloudTarget(
                kind="track",
                original=raw,
                urn=normalized_urn,
                permalink_url=permalink if isinstance(permalink, str) else None,
                username=username if isinstance(username, str) else None,
                resource_id=_resource_id_from_payload_or_urn(payload, normalized_urn),
            )
        if kind == "playlist" and isinstance(urn, str):
            normalized_urn = (
                _normalize_urn("playlists", urn.split(":")[-1])
                if not urn.startswith("soundcloud:playlists:")
                else urn
            )
            return SoundCloudTarget(
                kind="playlist",
                original=raw,
                urn=normalized_urn,
                permalink_url=permalink if isinstance(permalink, str) else None,
                username=username if isinstance(username, str) else None,
                resource_id=_resource_id_from_payload_or_urn(payload, normalized_urn),
                secret_token=_extract_target_secret_token(
                    payload=payload,
                    parsed=parsed,
                    raw=raw,
                ),
            )
        if kind == "user" and isinstance(urn, str):
            normalized_urn = (
                _normalize_urn("users", urn.split(":")[-1])
                if not urn.startswith("soundcloud:users:")
                else urn
            )
            return SoundCloudTarget(
                kind="artist",
                original=raw,
                urn=normalized_urn,
                permalink_url=permalink if isinstance(permalink, str) else None,
                username=username if isinstance(username, str) else None,
                resource_id=_resource_id_from_payload_or_urn(payload, normalized_urn),
            )

    if parsed.kind == "artist" and parsed.username:
        payload = _api_get(
            "/resolve",
            auth=auth,
            params={"url": f"https://soundcloud.com/{parsed.username}"},
        )
        if isinstance(payload, dict):
            urn = payload.get("urn")
            if isinstance(urn, str):
                norm = (
                    urn
                    if urn.startswith("soundcloud:users:")
                    else _normalize_urn("users", urn.split(":")[-1])
                )
                return SoundCloudTarget(
                    kind="artist",
                    original=raw,
                    urn=norm,
                    permalink_url=payload.get("permalink_url")
                    if isinstance(payload.get("permalink_url"), str)
                    else None,
                    username=payload.get("username")
                    if isinstance(payload.get("username"), str)
                    else parsed.username,
                    resource_id=_resource_id_from_payload_or_urn(payload, norm),
                )
    raise ValueError(f"Could not resolve SoundCloud target: {raw}")


def _safe_slug(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _urn_leaf(urn: str) -> str:
    parts = urn.split(":")
    if len(parts) >= 3:
        return parts[-1]
    return urn


def _resource_slug(resource: dict[str, Any], *, fallback: str) -> str:
    for key in ("permalink", "title", "username", "urn"):
        value = resource.get(key)
        if isinstance(value, str) and value.strip():
            return _safe_slug(value, fallback)
    return fallback


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None
    if parsed is None:
        for fmt in (
            "%Y/%m/%d %H:%M:%S %z",
            "%Y/%m/%d %H:%M:%S",
        ):
            try:
                parsed = datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue
            break
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_display(value: Any) -> str:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return str(value).strip() if value is not None else ""
    return parsed.strftime("%Y-%m-%dT%H:%MZ")


def _format_duration_ms(value: Any) -> str:
    try:
        raw_ms = float(str(value))
    except (TypeError, ValueError):
        return ""
    if raw_ms < 0:
        return ""
    total_seconds = int(raw_ms // 1000)
    if total_seconds < 60:
        return f"{total_seconds}s"
    if total_seconds < 3600:
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _comment_identifier(comment: dict[str, Any]) -> str | None:
    for key in ("urn", "id", "uri"):
        value = comment.get(key)
        if isinstance(value, str) and value.strip():
            if key == "urn" and value.startswith("soundcloud:comments:"):
                return value.rsplit(":", 1)[-1]
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return None


def _comment_parent_identifier(comment: dict[str, Any]) -> str | None:
    for key in (
        "parent_comment_urn",
        "parent_urn",
        "parent_comment_id",
        "parent_id",
        "in_reply_to_comment_urn",
        "in_reply_to",
    ):
        value = comment.get(key)
        if isinstance(value, str) and value.strip():
            if value.startswith("soundcloud:comments:"):
                return value.rsplit(":", 1)[-1]
            return value.strip()
        if isinstance(value, int):
            return str(value)
    parent = comment.get("parent")
    if isinstance(parent, dict):
        return _comment_identifier(parent)
    return None


def _comment_identity(comment: dict[str, Any]) -> tuple[str, str | None]:
    user = comment.get("user")
    if isinstance(user, dict):
        display_name = None
        username = None

        full_name = user.get("full_name")
        if isinstance(full_name, str) and full_name.strip():
            display_name = full_name.strip()

        permalink = user.get("permalink")
        if isinstance(permalink, str) and permalink.strip():
            username = permalink.strip().lstrip("@")

        username_value = user.get("username")
        if isinstance(username_value, str) and username_value.strip():
            if not display_name:
                display_name = username_value.strip()
            if not username:
                username = username_value.strip().lstrip("@")

        if not username:
            permalink_url = user.get("permalink_url")
            if isinstance(permalink_url, str) and permalink_url.strip():
                leaf = permalink_url.strip().rstrip("/").split("/")[-1]
                if leaf:
                    username = leaf.lstrip("@")

        if not display_name and username:
            display_name = username
        if display_name:
            return display_name, username if username else None
    return "Unknown", None


def _comment_body(comment: dict[str, Any]) -> str:
    body = comment.get("body")
    if not isinstance(body, str):
        return ""
    return body.replace("\r\n", "\n").replace("\r", "\n").strip()


def _comment_created(comment: dict[str, Any]) -> datetime:
    parsed = _parse_iso_datetime(comment.get("created_at"))
    return parsed or datetime.fromtimestamp(0, tz=timezone.utc)


def _comment_track_ts(comment: dict[str, Any]) -> float:
    value = comment.get("timestamp")
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return -1.0


def _sort_top_level_comments(
    comments: list[dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "created-asc":
        return sorted(
            comments,
            key=lambda comment: (
                _comment_created(comment),
                _comment_identifier(comment) or "",
            ),
        )
    if mode == "track-desc":
        return sorted(
            comments,
            key=lambda comment: (
                _comment_track_ts(comment),
                _comment_created(comment),
                _comment_identifier(comment) or "",
            ),
            reverse=True,
        )
    if mode == "track-asc":
        return sorted(
            comments,
            key=lambda comment: (
                (
                    _comment_track_ts(comment)
                    if _comment_track_ts(comment) >= 0
                    else math.inf
                ),
                _comment_created(comment),
                _comment_identifier(comment) or "",
            ),
        )
    return sorted(
        comments,
        key=lambda comment: (
            _comment_created(comment),
            _comment_identifier(comment) or "",
        ),
        reverse=True,
    )


def _sort_child_comments(comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        comments,
        key=lambda comment: (
            _comment_created(comment),
            _comment_identifier(comment) or "",
        ),
    )


def _render_comment_line(comment: dict[str, Any], *, prefix: str = "- ") -> str:
    display_name, username = _comment_identity(comment)
    created = _iso_display(comment.get("created_at"))
    track_offset = _format_duration_ms(comment.get("timestamp"))
    identity = f"[{display_name}]"
    if username:
        identity = f"{identity} (@{username})"
    header_parts = [identity]
    if created:
        header_parts.append(f"at {created}")
    if track_offset:
        header_parts.append(f"[t={track_offset}]")
    header = " ".join(header_parts).strip()
    body = _comment_body(comment)
    if not body:
        return f"{prefix}{header}"
    body_lines = body.splitlines()
    first = f"{prefix}{header}: {body_lines[0]}"
    tail = [f"  {line}" for line in body_lines[1:]]
    return "\n".join([first, *tail])


def _render_comments_section(
    comments: list[dict[str, Any]],
    *,
    total_count: int | None,
    settings: SoundCloudSettings,
) -> str:
    if not comments:
        return ""

    page_count = 1
    if isinstance(total_count, int) and total_count > 0:
        page_count = max(
            1, math.ceil(total_count / max(1, settings.comments_max_items))
        )

    lines = [f"## Comments (Page 1/{page_count})", ""]
    if isinstance(total_count, int) and total_count > len(comments):
        lines.append(f"Showing {len(comments)} of {total_count} comments.")
        lines.append("")

    id_map: dict[str, dict[str, Any]] = {}
    for comment in comments:
        cid = _comment_identifier(comment)
        if cid:
            id_map[cid] = comment

    top_level: list[dict[str, Any]] = []
    children: dict[str, list[dict[str, Any]]] = {}
    for comment in comments:
        parent_id = _comment_parent_identifier(comment)
        if (
            settings.comments_nesting_depth > 0
            and parent_id
            and parent_id in id_map
            and parent_id != _comment_identifier(comment)
        ):
            children.setdefault(parent_id, []).append(comment)
        else:
            top_level.append(comment)

    sorted_top = _sort_top_level_comments(top_level, mode=settings.comments_sort)
    rendered: list[str] = []
    for comment in sorted_top:
        rendered.append(_render_comment_line(comment))
        cid = _comment_identifier(comment)
        if cid and settings.comments_nesting_depth > 0:
            sorted_children = _sort_child_comments(children.get(cid, []))
            for child in sorted_children:
                rendered.append(_render_comment_line(child, prefix="  - "))
        rendered.append("")

    while rendered and not rendered[-1].strip():
        rendered.pop()

    lines.extend(rendered)
    return "\n".join(lines).strip()


def _normalize_llm_description(markdown: str) -> str:
    text = markdown.strip()
    if not text:
        return ""
    lines = text.splitlines()
    if lines:
        first = lines[0].strip().lower()
        if first in {
            "# description (auto-generated):",
            "## description (auto-generated):",
            "# description:",
            "## description:",
        }:
            lines = lines[1:]
            while lines and not lines[0].strip():
                lines = lines[1:]
    return "\n".join(lines).strip()


def _media_render_cache_identity(
    *,
    media_cache_identity: str,
    mode: str,
) -> str:
    payload = {
        "media_cache_identity": media_cache_identity,
        "mode": mode,
    }
    return "soundcloud-media-render:" + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )


def _describe_artwork(
    artwork_url_candidates: list[str],
    *,
    cache_identity: str,
    settings: SoundCloudSettings,
) -> str | None:
    if not settings.include_artwork_descriptions:
        return None

    from ..cache.soundcloud import (
        get_cached_media_bytes,
        get_cached_rendered,
        store_media_bytes,
        store_rendered,
    )
    from ..render.markitdown import convert_path_to_markdown
    from ..runtime import get_refresh_images
    from .media import download_cached_media_to_temp

    render_identity = _media_render_cache_identity(
        media_cache_identity=cache_identity,
        mode=settings.media_mode,
    )
    if not get_refresh_images():
        cached = get_cached_rendered(render_identity)
        if isinstance(cached, str) and cached.strip():
            return cached.strip()

    if not artwork_url_candidates:
        return None

    for idx, artwork_url in enumerate(artwork_url_candidates):
        candidate_identity = (
            cache_identity if idx == 0 else f"{cache_identity}:candidate-{idx}"
        )
        candidate_render_identity = _media_render_cache_identity(
            media_cache_identity=candidate_identity,
            mode=settings.media_mode,
        )
        if not get_refresh_images():
            cached = get_cached_rendered(candidate_render_identity)
            if isinstance(cached, str) and cached.strip():
                return cached.strip()

        suffix = Path(urlparse(artwork_url).path).suffix or ".jpg"
        tmp = download_cached_media_to_temp(
            artwork_url,
            suffix=suffix,
            headers=_DOWNLOAD_HEADERS,
            cache_identity=candidate_identity,
            get_cached_media_bytes=get_cached_media_bytes,
            store_media_bytes=store_media_bytes,
            refresh_cache=get_refresh_images(),
        )
        if tmp is None:
            continue
        try:
            result = convert_path_to_markdown(
                tmp,
                refresh_images=get_refresh_images(),
                source_url=artwork_url,
            )
            cleaned = _normalize_llm_description(result.markdown)
            if not cleaned:
                continue
            store_rendered(candidate_render_identity, cleaned)
            return cleaned
        except Exception as exc:
            _log(f"  soundcloud artwork description failed for {artwork_url}: {exc}")
            continue
        finally:
            tmp.unlink(missing_ok=True)
    return None


def _replace_artwork_size_variant(url: str, variant: str) -> str:
    path = urlparse(url).path
    file_name = path.rsplit("/", 1)[-1]
    if not file_name:
        return url
    replaced = re.sub(
        r"-(?:large|t\d+x\d+|crop|original)\.",
        f"-{variant}.",
        file_name,
        count=1,
    )
    if replaced == file_name:
        if "." not in file_name:
            return url
        stem, ext = file_name.rsplit(".", 1)
        replaced = f"{stem}-{variant}.{ext}"
    return url.replace(file_name, replaced)


def _artwork_variant_candidates(url: str) -> list[str]:
    variants = ["original", "t500x500", "crop", "large"]
    ordered = [_replace_artwork_size_variant(url, variant) for variant in variants]
    ordered.append(url)
    deduped: list[str] = []
    for candidate in ordered:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _resource_artwork_candidates(
    resource: dict[str, Any],
    *,
    include_avatar_fallback: bool,
) -> list[str]:
    candidates: list[str] = []

    visuals = resource.get("visuals")
    if isinstance(visuals, dict):
        visuals_items = visuals.get("visuals")
        if isinstance(visuals_items, list):
            for item in visuals_items:
                if not isinstance(item, dict):
                    continue
                visual_url = item.get("visual_url")
                if isinstance(visual_url, str) and visual_url.strip():
                    candidates.append(visual_url.strip())

    artwork_url = resource.get("artwork_url")
    if isinstance(artwork_url, str) and artwork_url.strip():
        candidates.extend(_artwork_variant_candidates(artwork_url.strip()))

    if include_avatar_fallback:
        user = resource.get("user")
        if isinstance(user, dict):
            avatar_url = user.get("avatar_url")
            if isinstance(avatar_url, str) and avatar_url.strip():
                candidates.extend(_artwork_variant_candidates(avatar_url.strip()))

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _track_source_fields(track: dict[str, Any]) -> tuple[str | None, str | None]:
    created = track.get("created_at")
    modified = track.get("last_modified")
    return (
        created if isinstance(created, str) else None,
        modified if isinstance(modified, str) else None,
    )


def _render_track_document(
    track: dict[str, Any],
    *,
    source_url: str,
    settings: SoundCloudSettings,
    comments: list[dict[str, Any]] | None = None,
) -> str:
    title = str(track.get("title") or "Untitled track")
    urn = str(track.get("urn") or "")
    permalink = str(track.get("permalink_url") or "")
    user = track.get("user") if isinstance(track.get("user"), dict) else {}
    artist_name = (
        str(user.get("username") or user.get("full_name") or "").strip() if user else ""
    )

    lines = [f"# Track: {title}", ""]
    if permalink:
        lines.append(f"- url: {permalink}")
    lines.append(f"- source_url: {source_url}")
    if artist_name:
        lines.append(f"- artist: {artist_name}")
    created_at = _iso_display(track.get("created_at"))
    if created_at:
        lines.append(f"- created_at: {created_at}")
    last_modified = _iso_display(track.get("last_modified"))
    if last_modified:
        lines.append(f"- last_modified: {last_modified}")
    duration = _format_duration_ms(track.get("duration"))
    if duration:
        lines.append(f"- duration: {duration}")
    access = track.get("access")
    if isinstance(access, str) and access:
        lines.append(f"- access: {access}")
    lines.append("")

    description = track.get("description")
    if isinstance(description, str) and description.strip():
        lines.append("## Description")
        lines.append("")
        lines.append(description.strip())
        lines.append("")

    artwork_description = _describe_artwork(
        _resource_artwork_candidates(track, include_avatar_fallback=True),
        cache_identity=f"soundcloud:track:{urn}:artwork",
        settings=settings,
    )
    if artwork_description:
        lines.append("## Artwork Description (auto-generated)")
        lines.append("")
        lines.append(artwork_description)
        lines.append("")

    if comments:
        total_comments = track.get("comment_count")
        total_count = total_comments if isinstance(total_comments, int) else None
        rendered_comments = _render_comments_section(
            comments,
            total_count=total_count,
            settings=settings,
        )
        if rendered_comments:
            lines.append(rendered_comments)
            lines.append("")

    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _render_playlist_summary(
    playlist: dict[str, Any],
    *,
    source_url: str,
    settings: SoundCloudSettings,
    heading_prefix: str = "Playlist",
) -> str:
    title = str(playlist.get("title") or "Untitled playlist")
    urn = str(playlist.get("urn") or "")
    permalink = str(playlist.get("permalink_url") or "")
    playlist_type = str(playlist.get("playlist_type") or playlist.get("type") or "")
    is_album = playlist_type.lower() in {"album", "ep single", "single"}
    heading_kind = "Album" if is_album else heading_prefix

    lines = [f"# {heading_kind}: {title}", ""]
    if permalink:
        lines.append(f"- url: {permalink}")
    lines.append(f"- source_url: {source_url}")
    if playlist_type:
        lines.append(f"- playlist_type: {playlist_type}")
    track_count = playlist.get("track_count")
    if isinstance(track_count, int):
        lines.append(f"- track_count: {track_count}")
    created_at = _iso_display(playlist.get("created_at"))
    if created_at:
        lines.append(f"- created_at: {created_at}")
    last_modified = _iso_display(playlist.get("last_modified"))
    if last_modified:
        lines.append(f"- last_modified: {last_modified}")
    lines.append("")

    description = playlist.get("description")
    if isinstance(description, str) and description.strip():
        lines.append("## Description")
        lines.append("")
        lines.append(description.strip())
        lines.append("")

    artwork_description = _describe_artwork(
        _resource_artwork_candidates(playlist, include_avatar_fallback=True),
        cache_identity=f"soundcloud:playlist:{urn}:artwork",
        settings=settings,
    )
    if artwork_description:
        lines.append("## Artwork Description (auto-generated)")
        lines.append("")
        lines.append(artwork_description)
        lines.append("")

    tracks = playlist.get("tracks")
    if isinstance(tracks, list) and tracks:
        lines.append("## Tracks")
        lines.append("")
        for track in tracks:
            if not isinstance(track, dict):
                continue
            track_title = str(track.get("title") or "untitled").strip()
            track_urn = str(track.get("urn") or "").strip()
            if track_urn:
                lines.append(f"- {track_title} (`{track_urn}`)")
            else:
                lines.append(f"- {track_title}")
        lines.append("")

    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _render_artist_summary(
    user: dict[str, Any],
    *,
    source_url: str,
    settings: SoundCloudSettings,
    counts: dict[str, Any],
    repost_note: str | None,
) -> str:
    username = str(user.get("username") or user.get("permalink") or "unknown")
    urn = str(user.get("urn") or "")
    permalink = str(user.get("permalink_url") or "")
    lines = [f"# Artist: {username}", ""]
    if permalink:
        lines.append(f"- url: {permalink}")
    lines.append(f"- source_url: {source_url}")
    created_at = _iso_display(user.get("created_at"))
    if created_at:
        lines.append(f"- created_at: {created_at}")
    last_modified = _iso_display(user.get("last_modified"))
    if last_modified:
        lines.append(f"- last_modified: {last_modified}")
    for key in ("tracks", "playlists", "reposts"):
        value = counts.get(key)
        if isinstance(value, int):
            lines.append(f"- included_{key}: {value}")
    lines.append("")

    description = user.get("description")
    if isinstance(description, str) and description.strip():
        lines.append("## Bio")
        lines.append("")
        lines.append(description.strip())
        lines.append("")

    artwork_description = _describe_artwork(
        _resource_artwork_candidates(user, include_avatar_fallback=True),
        cache_identity=f"soundcloud:user:{urn}:avatar",
        settings=settings,
    )
    if artwork_description:
        lines.append("## Avatar Description (auto-generated)")
        lines.append("")
        lines.append(artwork_description)
        lines.append("")

    if repost_note:
        lines.append("## Reposts")
        lines.append("")
        lines.append(repost_note.strip())
        lines.append("")

    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _collection_items(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)], None
    if isinstance(payload, dict):
        collection = payload.get("collection")
        next_href = payload.get("next_href")
        if isinstance(collection, list):
            return [item for item in collection if isinstance(item, dict)], (
                next_href if isinstance(next_href, str) and next_href else None
            )
    return [], None


def _fetch_paginated_collection(
    path: str,
    *,
    auth: _AuthContext,
    max_items: int | None,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    next_path: str | None = path
    query = dict(params or {})
    if "linked_partitioning" not in query:
        query["linked_partitioning"] = "true"
    if "limit" not in query:
        query["limit"] = 200

    while next_path and (max_items is None or len(items) < max_items):
        payload = _api_get(
            next_path, auth=auth, params=query if next_path == path else None
        )
        page_items, next_href = _collection_items(payload)
        if not page_items:
            break
        remaining = (
            len(page_items) if max_items is None else max(0, max_items - len(items))
        )
        items.extend(page_items[:remaining])
        if max_items is not None and len(items) >= max_items:
            break
        next_path = next_href
    return items


def _fetch_track(
    track_urn: str,
    *,
    auth: _AuthContext,
) -> dict[str, Any]:
    encoded = quote(track_urn, safe="")
    payload = _api_get(f"/tracks/{encoded}", auth=auth)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected SoundCloud track payload for {track_urn}")
    return payload


def _fetch_playlist(
    target: SoundCloudTarget,
    *,
    auth: _AuthContext,
) -> dict[str, Any]:
    if not target.urn:
        raise ValueError("SoundCloud playlist target is missing urn")

    params: dict[str, Any] = {"show_tracks": "true"}
    if target.secret_token:
        params["secret_token"] = target.secret_token

    attempts: list[str] = []
    if target.resource_id:
        attempts.append(target.resource_id)
    attempts.append(target.urn)

    deduped_attempts: list[str] = []
    for attempt in attempts:
        if attempt not in deduped_attempts:
            deduped_attempts.append(attempt)

    last_exc: Exception | None = None
    for attempt in deduped_attempts:
        encoded = quote(attempt, safe="")
        try:
            payload = _api_get(
                f"/playlists/{encoded}",
                auth=auth,
                params=params,
            )
        except Exception as exc:
            last_exc = exc
            continue
        if not isinstance(payload, dict):
            last_exc = ValueError(
                f"Unexpected SoundCloud playlist payload for {target.urn}"
            )
            continue
        return payload

    hint = ""
    if target.secret_token:
        hint = " Verify the playlist secret token is valid for this playlist URL."
    else:
        hint = " This may be a private playlist; use the playlist URL that includes the `/s-...` secret token segment."
    if last_exc is not None:
        raise ValueError(
            f"Unable to fetch SoundCloud playlist {target.urn}.{hint}"
        ) from last_exc
    raise ValueError(f"Unable to fetch SoundCloud playlist {target.urn}.{hint}")


def _fetch_user(
    user_urn: str,
    *,
    auth: _AuthContext,
) -> dict[str, Any]:
    encoded = quote(user_urn, safe="")
    payload = _api_get(f"/users/{encoded}", auth=auth)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected SoundCloud user payload for {user_urn}")
    return payload


def _fetch_track_comments(
    track_urn: str,
    *,
    auth: _AuthContext,
    settings: SoundCloudSettings,
) -> list[dict[str, Any]]:
    if settings.comments_max_items <= 0:
        return []
    encoded = quote(track_urn, safe="")
    return _fetch_paginated_collection(
        f"/tracks/{encoded}/comments",
        auth=auth,
        max_items=settings.comments_max_items,
        params={
            "limit": min(200, settings.comments_max_items),
            "linked_partitioning": "true",
        },
    )


def _track_docs_from_playlist(
    playlist: dict[str, Any],
    *,
    source_url: str,
    settings: SoundCloudSettings,
    root_dir: str,
) -> list[SoundCloudDocument]:
    docs: list[SoundCloudDocument] = []
    tracks = playlist.get("tracks")
    if not isinstance(tracks, list):
        return docs
    for index, track in enumerate(tracks, start=1):
        if not isinstance(track, dict):
            continue
        track_urn = track.get("urn")
        if not isinstance(track_urn, str) or not track_urn:
            continue
        track_slug = _resource_slug(track, fallback=f"track-{index:03d}")
        rendered = _render_track_document(
            track,
            source_url=source_url,
            settings=settings,
            comments=None,
        )
        created, modified = _track_source_fields(track)
        docs.append(
            SoundCloudDocument(
                source_url=source_url,
                kind="track",
                urn=track_urn,
                label=f"{root_dir}/tracks/{track_slug}",
                trace_path=f"{root_dir}/tracks/{track_slug}.md",
                context_subpath=f"{root_dir}/tracks/{track_slug}.md",
                rendered=rendered,
                source_created=created,
                source_modified=modified,
                scope_id=playlist.get("urn")
                if isinstance(playlist.get("urn"), str)
                else None,
            )
        )
    return docs


def _resolve_track_documents(
    target: SoundCloudTarget,
    *,
    source_url: str,
    auth: _AuthContext,
    settings: SoundCloudSettings,
) -> list[SoundCloudDocument]:
    assert target.urn is not None
    track = _fetch_track(target.urn, auth=auth)
    track_urn = (
        str(track.get("urn")) if isinstance(track.get("urn"), str) else target.urn
    )
    track_slug = _resource_slug(
        track, fallback=_safe_slug(_urn_leaf(track_urn), "track")
    )
    comments: list[dict[str, Any]] | None = None
    if settings.include_comments:
        try:
            comments = _fetch_track_comments(track_urn, auth=auth, settings=settings)
        except Exception as exc:
            _log(f"  failed to fetch SoundCloud comments for {track_urn}: {exc}")
            comments = None
    rendered = _render_track_document(
        track,
        source_url=source_url,
        settings=settings,
        comments=comments,
    )
    created, modified = _track_source_fields(track)
    root_label = f"soundcloud/tracks/{track_slug}"
    return [
        SoundCloudDocument(
            source_url=source_url,
            kind="track",
            urn=track_urn,
            label=root_label,
            trace_path=f"{root_label}.md",
            context_subpath=f"{root_label}.md",
            rendered=rendered,
            source_created=created,
            source_modified=modified,
            scope_id=track_urn,
        )
    ]


def _resolve_playlist_documents(
    target: SoundCloudTarget,
    *,
    source_url: str,
    auth: _AuthContext,
    settings: SoundCloudSettings,
) -> list[SoundCloudDocument]:
    assert target.urn is not None
    playlist = _fetch_playlist(target, auth=auth)
    playlist_urn = (
        str(playlist.get("urn")) if isinstance(playlist.get("urn"), str) else target.urn
    )
    playlist_slug = _resource_slug(
        playlist,
        fallback=_safe_slug(_urn_leaf(playlist_urn), "playlist"),
    )
    root_dir = f"soundcloud/playlists/{playlist_slug}"
    summary = SoundCloudDocument(
        source_url=source_url,
        kind="playlist",
        urn=playlist_urn,
        label=f"{root_dir}/summary",
        trace_path=f"{root_dir}/summary.md",
        context_subpath=f"{root_dir}/summary.md",
        rendered=_render_playlist_summary(
            playlist,
            source_url=source_url,
            settings=settings,
        ),
        source_created=playlist.get("created_at")
        if isinstance(playlist.get("created_at"), str)
        else None,
        source_modified=playlist.get("last_modified")
        if isinstance(playlist.get("last_modified"), str)
        else None,
        scope_id=playlist_urn,
    )
    track_docs = _track_docs_from_playlist(
        playlist,
        source_url=source_url,
        settings=settings,
        root_dir=root_dir,
    )
    return [summary, *track_docs]


def _auth_user_urn(auth: _AuthContext) -> str | None:
    if not auth.has_user_token:
        return None
    try:
        payload = _api_get("/me", auth=auth)
    except Exception:
        return None
    if isinstance(payload, dict):
        urn = payload.get("urn")
        if isinstance(urn, str) and urn:
            return urn
    return None


def _is_authenticated_self(user_urn: str, auth: _AuthContext) -> bool:
    me_urn = _auth_user_urn(auth)
    return bool(me_urn and me_urn == user_urn)


def _parse_activity_origin(entry: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    entry_type = str(entry.get("type") or "").lower()
    if "repost" not in entry_type:
        return None
    origin = entry.get("origin")
    if not isinstance(origin, dict):
        return None
    kind = str(origin.get("kind") or "").lower()
    if kind == "track":
        return "track", origin
    if kind == "playlist":
        return "playlist", origin
    return None


def _fetch_artist_reposts(
    *,
    auth: _AuthContext,
    max_items: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = _fetch_paginated_collection(
        "/me/feed",
        auth=auth,
        max_items=max_items,
        params={
            "limit": 200 if max_items is None else min(200, max_items),
            "linked_partitioning": "true",
        },
    )
    repost_tracks: list[dict[str, Any]] = []
    repost_playlists: list[dict[str, Any]] = []
    for entry in entries:
        parsed = _parse_activity_origin(entry)
        if parsed is None:
            continue
        kind, origin = parsed
        if kind == "track":
            repost_tracks.append(origin)
        else:
            repost_playlists.append(origin)
    return repost_tracks, repost_playlists


def _resolve_artist_documents(
    target: SoundCloudTarget,
    *,
    source_url: str,
    auth: _AuthContext,
    settings: SoundCloudSettings,
) -> list[SoundCloudDocument]:
    assert target.urn is not None
    user = _fetch_user(target.urn, auth=auth)
    user_urn = str(user.get("urn") or target.urn)
    user_slug = _resource_slug(user, fallback=_safe_slug(_urn_leaf(user_urn), "artist"))
    root_dir = f"soundcloud/artists/{user_slug}"

    include_tracks = settings.artist_tracks_filter != "exclude"
    include_playlists = settings.artist_playlists_filter != "exclude"
    include_reposts = settings.artist_reposts_filter != "exclude"
    if settings.artist_tracks_filter == "only":
        include_playlists = False
        include_reposts = False
    if settings.artist_playlists_filter == "only":
        include_tracks = False
        include_reposts = False
    if settings.artist_reposts_filter == "only":
        include_tracks = False
        include_playlists = False

    track_items: list[dict[str, Any]] = []
    playlist_items: list[dict[str, Any]] = []
    repost_track_items: list[dict[str, Any]] = []
    repost_playlist_items: list[dict[str, Any]] = []
    repost_note: str | None = None

    max_items = settings.max_items
    encoded_user = quote(user_urn, safe="")
    if include_tracks:
        track_items = _fetch_paginated_collection(
            f"/users/{encoded_user}/tracks",
            auth=auth,
            max_items=max_items,
            params={
                "limit": 200 if max_items is None else min(200, max_items),
                "linked_partitioning": "true",
            },
        )

    if include_playlists:
        playlist_items = _fetch_paginated_collection(
            f"/users/{encoded_user}/playlists",
            auth=auth,
            max_items=max_items,
            params={
                "show_tracks": "true",
                "limit": 200 if max_items is None else min(200, max_items),
                "linked_partitioning": "true",
            },
        )

    if include_reposts:
        if _is_authenticated_self(user_urn, auth):
            repost_track_items, repost_playlist_items = _fetch_artist_reposts(
                auth=auth,
                max_items=max_items,
            )
        else:
            repost_note = (
                "Repost listing is only available for the authenticated self artist via `/me/feed`; "
                "the SoundCloud public API does not expose a generic user repost listing endpoint."
            )

    docs: list[SoundCloudDocument] = []
    summary = SoundCloudDocument(
        source_url=source_url,
        kind="artist",
        urn=user_urn,
        label=f"{root_dir}/summary",
        trace_path=f"{root_dir}/summary.md",
        context_subpath=f"{root_dir}/summary.md",
        rendered=_render_artist_summary(
            user,
            source_url=source_url,
            settings=settings,
            counts={
                "tracks": len(track_items),
                "playlists": len(playlist_items),
                "reposts": len(repost_track_items) + len(repost_playlist_items),
            },
            repost_note=repost_note,
        ),
        source_created=user.get("created_at")
        if isinstance(user.get("created_at"), str)
        else None,
        source_modified=user.get("last_modified")
        if isinstance(user.get("last_modified"), str)
        else None,
        scope_id=user_urn,
    )
    docs.append(summary)

    seen_paths: set[str] = {summary.context_subpath}

    for idx, track in enumerate(track_items, start=1):
        if not isinstance(track, dict):
            continue
        track_urn = track.get("urn")
        if not isinstance(track_urn, str):
            continue
        track_slug = _resource_slug(track, fallback=f"track-{idx:03d}")
        context_subpath = f"{root_dir}/tracks/{track_slug}.md"
        if context_subpath in seen_paths:
            continue
        seen_paths.add(context_subpath)
        rendered = _render_track_document(
            track,
            source_url=source_url,
            settings=settings,
            comments=None,
        )
        created, modified = _track_source_fields(track)
        docs.append(
            SoundCloudDocument(
                source_url=source_url,
                kind="track",
                urn=track_urn,
                label=context_subpath[:-3],
                trace_path=context_subpath,
                context_subpath=context_subpath,
                rendered=rendered,
                source_created=created,
                source_modified=modified,
                scope_id=user_urn,
            )
        )

    for idx, playlist in enumerate(playlist_items, start=1):
        if not isinstance(playlist, dict):
            continue
        playlist_urn = playlist.get("urn")
        if not isinstance(playlist_urn, str):
            continue
        playlist_slug = _resource_slug(playlist, fallback=f"playlist-{idx:03d}")
        playlist_root = f"{root_dir}/playlists/{playlist_slug}"
        summary_path = f"{playlist_root}/summary.md"
        if summary_path not in seen_paths:
            seen_paths.add(summary_path)
            docs.append(
                SoundCloudDocument(
                    source_url=source_url,
                    kind="playlist",
                    urn=playlist_urn,
                    label=summary_path[:-3],
                    trace_path=summary_path,
                    context_subpath=summary_path,
                    rendered=_render_playlist_summary(
                        playlist,
                        source_url=source_url,
                        settings=settings,
                        heading_prefix="Playlist",
                    ),
                    source_created=playlist.get("created_at")
                    if isinstance(playlist.get("created_at"), str)
                    else None,
                    source_modified=playlist.get("last_modified")
                    if isinstance(playlist.get("last_modified"), str)
                    else None,
                    scope_id=user_urn,
                )
            )
        for track_doc in _track_docs_from_playlist(
            playlist,
            source_url=source_url,
            settings=settings,
            root_dir=playlist_root,
        ):
            if track_doc.context_subpath in seen_paths:
                continue
            seen_paths.add(track_doc.context_subpath)
            docs.append(track_doc)

    for idx, track in enumerate(repost_track_items, start=1):
        if not isinstance(track, dict):
            continue
        track_urn = track.get("urn")
        if not isinstance(track_urn, str):
            continue
        track_slug = _resource_slug(track, fallback=f"repost-track-{idx:03d}")
        context_subpath = f"{root_dir}/reposts/tracks/{track_slug}.md"
        if context_subpath in seen_paths:
            continue
        seen_paths.add(context_subpath)
        rendered = _render_track_document(
            track,
            source_url=source_url,
            settings=settings,
            comments=None,
        )
        created, modified = _track_source_fields(track)
        docs.append(
            SoundCloudDocument(
                source_url=source_url,
                kind="repost-track",
                urn=track_urn,
                label=context_subpath[:-3],
                trace_path=context_subpath,
                context_subpath=context_subpath,
                rendered=rendered,
                source_created=created,
                source_modified=modified,
                scope_id=user_urn,
            )
        )

    for idx, playlist in enumerate(repost_playlist_items, start=1):
        if not isinstance(playlist, dict):
            continue
        playlist_urn = playlist.get("urn")
        if not isinstance(playlist_urn, str):
            continue
        playlist_slug = _resource_slug(playlist, fallback=f"repost-playlist-{idx:03d}")
        playlist_root = f"{root_dir}/reposts/playlists/{playlist_slug}"
        summary_path = f"{playlist_root}/summary.md"
        if summary_path in seen_paths:
            continue
        seen_paths.add(summary_path)
        docs.append(
            SoundCloudDocument(
                source_url=source_url,
                kind="repost-playlist",
                urn=playlist_urn,
                label=summary_path[:-3],
                trace_path=summary_path,
                context_subpath=summary_path,
                rendered=_render_playlist_summary(
                    playlist,
                    source_url=source_url,
                    settings=settings,
                    heading_prefix="Reposted Playlist",
                ),
                source_created=playlist.get("created_at")
                if isinstance(playlist.get("created_at"), str)
                else None,
                source_modified=playlist.get("last_modified")
                if isinstance(playlist.get("last_modified"), str)
                else None,
                scope_id=user_urn,
            )
        )

    return docs


def resolve_soundcloud_url(
    url: str,
    *,
    settings: SoundCloudSettings | None = None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> list[SoundCloudDocument]:
    from ..cache.soundcloud import get_cached_api_json, store_api_json
    from ..runtime import get_refresh_media

    warmup_soundcloud_network_stack()
    parsed = parse_soundcloud_target(url)
    if parsed is None:
        raise ValueError(f"Not a SoundCloud target: {url}")

    effective_settings = (
        settings if settings is not None else _soundcloud_settings_from_env()
    )
    auth = _build_auth_context()
    _log(f"Resolving soundcloud target: {url} (auth={auth.auth_mode})")

    refresh_resolution_cache = refresh_cache or get_refresh_media()
    cache_identity = _resolve_cache_identity(url, effective_settings, auth)
    if use_cache and not refresh_resolution_cache:
        cached = get_cached_api_json(cache_identity, ttl=cache_ttl)
        cached_docs = _documents_from_cached_payload(cached)
        if cached_docs is not None:
            _log(f"  soundcloud resolution cache hit: {url}")
            return cached_docs

    target = _resolve_target(url, parsed=parsed, auth=auth)
    if target.kind == "track":
        docs = _resolve_track_documents(
            target,
            source_url=url,
            auth=auth,
            settings=effective_settings,
        )
    elif target.kind == "playlist":
        docs = _resolve_playlist_documents(
            target,
            source_url=url,
            auth=auth,
            settings=effective_settings,
        )
    elif target.kind == "artist":
        docs = _resolve_artist_documents(
            target,
            source_url=url,
            auth=auth,
            settings=effective_settings,
        )
    else:
        raise ValueError(f"Unsupported SoundCloud target kind: {target.kind}")

    if use_cache:
        store_api_json(cache_identity, [asdict(document) for document in docs])
    return docs


@dataclass
class SoundCloudReference:
    url: str
    document: SoundCloudDocument
    format: str = "md"
    label: str = "relative"
    token_target: str = "cl100k_base"
    include_token_count: bool = False
    label_suffix: str | None = None
    inject: bool = False
    depth: int = 5
    trace_collector: list | None = None

    def __post_init__(self) -> None:
        self.file_content = self.document.rendered
        self.original_file_content = self.document.rendered
        self.output = self._get_contents()

    @property
    def path(self) -> str:
        return self.url

    @property
    def trace_path(self) -> str:
        return self.document.trace_path

    def read(self) -> str:
        return self.original_file_content

    def exists(self) -> bool:
        return True

    def token_count(self, encoding: str = "cl100k_base") -> int:
        return count_tokens(self.original_file_content, target=encoding)["count"]

    def get_label(self) -> str:
        if self.label == "relative":
            return self.document.label
        if self.label == "name":
            return self.document.label.rsplit("/", 1)[-1]
        if self.label == "ext":
            return ""
        return self.label

    def _get_contents(self) -> str:
        text = self.file_content
        if self.inject and text:
            from ..render.inject import inject_content_in_text

            text = inject_content_in_text(
                text,
                self.depth,
                self.trace_collector,
                self.url,
            )
            self.file_content = text
        return process_text(
            text,
            format=self.format,
            label=self.get_label(),
            label_suffix=self.label_suffix,
            token_target=self.token_target,
            include_token_count=self.include_token_count,
        )
