from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..render.text import process_text
from ..utils import count_tokens

_DISCORD_MESSAGE_RE = re.compile(
    r"^https?://(?:ptb\.|canary\.)?discord\.com/channels/"
    r"(?P<guild_id>[^/]+)/(?P<channel_id>\d+)/(?P<message_id>\d+)"
    r"(?:[/?#].*)?$"
)
_DISCORD_CHANNEL_RE = re.compile(
    r"^https?://(?:ptb\.|canary\.)?discord\.com/channels/"
    r"(?P<guild_id>[^/]+)/(?P<channel_id>\d+)"
    r"(?:[/?#].*)?$"
)

_DISCORD_THREAD_TYPES = frozenset({10, 11, 12})
_NON_SYSTEM_MESSAGE_TYPES = frozenset({0, 19, 20, 21})
_DISCORD_THREAD_STARTER_MESSAGE_TYPE = 21
_REPLY_QUOTE_MAX_CHARS = 600
_VALID_FORMATS = frozenset({"idiomatic", "yaml"})
_MEDIA_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".heic", ".heif"}
)
_MEDIA_VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg", ".m4v"}
)
_MEDIA_AUDIO_SUFFIXES = frozenset(
    {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".aiff"}
)
_MARKDOWN_ESCAPED_PUNCT_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|])")
_MEDIA_DOWNLOAD_HEADERS = {
    "User-Agent": "contextualize/discord",
    "Accept": "image/*,video/*,audio/*,application/pdf,application/octet-stream",
}


def _log(message: str) -> None:
    from ..runtime import get_verbose_logging

    if get_verbose_logging():
        print(message, file=sys.stderr, flush=True)


@lru_cache(maxsize=1)
def warmup_discord_network_stack() -> None:
    import ssl
    import requests

    ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    _ = requests.__version__


@lru_cache(maxsize=1)
def _load_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except Exception:
        return


def is_discord_url(url: str) -> bool:
    return is_discord_message_url(url) or is_discord_channel_or_thread_url(url)


def is_discord_message_url(url: str) -> bool:
    return bool(_DISCORD_MESSAGE_RE.match(url))


def is_discord_channel_or_thread_url(url: str) -> bool:
    if not _DISCORD_CHANNEL_RE.match(url):
        return False
    return not is_discord_message_url(url)


def parse_discord_url(url: str) -> dict[str, str] | None:
    msg_match = _DISCORD_MESSAGE_RE.match(url)
    if msg_match:
        return {
            "kind": "message",
            "guild_id": msg_match.group("guild_id"),
            "channel_id": msg_match.group("channel_id"),
            "message_id": msg_match.group("message_id"),
        }
    channel_match = _DISCORD_CHANNEL_RE.match(url)
    if channel_match:
        return {
            "kind": "channel",
            "guild_id": channel_match.group("guild_id"),
            "channel_id": channel_match.group("channel_id"),
        }
    return None


def _parse_bool(value: str, *, default: bool) -> bool:
    cleaned = value.strip().lower()
    if not cleaned:
        return default
    return cleaned not in {"0", "false", "no", "off"}


def _parse_optional_int(value: str) -> int | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        parsed = int(cleaned)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def _parse_iso_datetime(value: str) -> datetime | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.isdigit():
        try:
            ts = int(cleaned)
        except ValueError:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    try:
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _parse_duration_compound(raw: str) -> timedelta | None:
    value = raw.strip().lower()
    if not value:
        return None
    if value.isdigit():
        return timedelta(seconds=int(value))
    parts = re.findall(r"(\d+)\s*([ymdhis])", value)
    if not parts:
        return None
    cursor = 0
    for amount, unit in parts:
        token = f"{amount}{unit}"
        idx = value.find(token, cursor)
        if idx < 0:
            return None
        cursor = idx + len(token)
    if value[cursor:].strip():
        return None

    seconds = 0
    for amount_raw, unit in parts:
        amount = int(amount_raw)
        if unit == "y":
            seconds += amount * 365 * 24 * 60 * 60
        elif unit == "m":
            seconds += amount * 30 * 24 * 60 * 60
        elif unit == "d":
            seconds += amount * 24 * 60 * 60
        elif unit == "h":
            seconds += amount * 60 * 60
        elif unit == "i":
            seconds += amount * 60
        elif unit == "s":
            seconds += amount
    return timedelta(seconds=seconds)


def _parse_gap_threshold(raw: str, *, default: timedelta) -> timedelta:
    parsed = _parse_duration_compound(raw)
    if parsed is None or parsed.total_seconds() <= 0:
        return default
    return parsed


def _format_duration_human(value: timedelta) -> str:
    total = int(value.total_seconds())
    if total < 60:
        return f"{total}s"
    if total < 3600:
        minutes, seconds = divmod(total, 60)
        return f"{minutes}m{seconds}s" if seconds else f"{minutes}m"
    if total < 86400:
        hours, rest = divmod(total, 3600)
        minutes = rest // 60
        return f"{hours}h{minutes}m" if minutes else f"{hours}h"
    days, rest = divmod(total, 86400)
    hours = rest // 3600
    return f"{days}d{hours}h" if hours else f"{days}d"


def _parse_media_kind(*, filename: str, content_type: str) -> str:
    ctype = (content_type or "").lower().strip()
    if ctype.startswith("image/"):
        return "image"
    if ctype.startswith("video/"):
        return "video"
    if ctype.startswith("audio/"):
        return "audio"

    suffix = Path(filename).suffix.lower()
    if suffix in _MEDIA_IMAGE_SUFFIXES:
        return "image"
    if suffix in _MEDIA_VIDEO_SUFFIXES:
        return "video"
    if suffix in _MEDIA_AUDIO_SUFFIXES:
        return "audio"
    return "file"


def _suffix_from_media_kind(kind: str) -> str:
    if kind == "image":
        return ".jpg"
    if kind == "video":
        return ".mp4"
    if kind == "audio":
        return ".wav"
    return ".bin"


def _suffix_from_content_type(content_type: str) -> str | None:
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    if not ctype:
        return None
    if ctype.startswith("image/"):
        subtype = ctype.removeprefix("image/")
        if subtype in {"jpeg", "pjpeg"}:
            return ".jpg"
        if subtype:
            return f".{subtype}"
    if ctype.startswith("video/"):
        subtype = ctype.removeprefix("video/")
        return f".{subtype}" if subtype else ".mp4"
    if ctype.startswith("audio/"):
        subtype = ctype.removeprefix("audio/")
        return f".{subtype}" if subtype else ".wav"
    if ctype == "application/pdf":
        return ".pdf"
    return None


def _media_suffix(
    *,
    filename: str,
    content_type: str,
    kind: str,
    url: str | None = None,
) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix:
        return suffix
    if url:
        url_suffix = Path(url.split("?", 1)[0]).suffix.lower()
        if url_suffix:
            return url_suffix
    content_type_suffix = _suffix_from_content_type(content_type)
    if content_type_suffix:
        return content_type_suffix
    return _suffix_from_media_kind(kind)


def _attachment_media_cache_identity(
    *,
    message_id: str | None,
    attachment_id: str | None,
    filename: str,
    url: str,
) -> str:
    if not message_id:
        return url
    attachment_key = attachment_id or filename or "attachment"
    return f"discord:message:{message_id}:attachment:{attachment_key}:{url}"


def _embed_media_cache_identity(
    *,
    message_id: str | None,
    embed_index: int,
    role: str,
    url: str,
) -> str:
    if not message_id:
        return url
    return f"discord:message:{message_id}:embed:{embed_index}:{role}:{url}"


def _normalize_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class DiscordSettings:
    format: str = "idiomatic"
    include_system: bool = True
    include_thread_starters: bool = True
    expand_threads: bool = False
    gap_threshold: timedelta = timedelta(hours=6)
    before_messages: int | None = 0
    after_messages: int | None = 0
    around_messages: int | None = None
    message_context: int | None = None
    channel_limit: int = 200
    start: datetime | None = None
    end: datetime | None = None
    before_duration: timedelta | None = None
    after_duration: timedelta | None = None
    around_duration: timedelta | None = None
    include_media_descriptions: bool = True
    include_embed_media_descriptions: bool = True
    hard_min_timestamp: datetime | None = None
    hard_max_timestamp: datetime | None = None


def _distribute_message_context(
    *,
    message_context: int | None,
    before_messages: int | None,
    after_messages: int | None,
) -> tuple[int | None, int | None]:
    if message_context is None:
        return before_messages, after_messages

    total = max(0, int(message_context))
    if before_messages is None and after_messages is None:
        before = (total + 1) // 2
        after = total // 2
        return before, after

    if before_messages is None:
        after = max(0, int(after_messages or 0))
        return max(0, total - after), after

    if after_messages is None:
        before = max(0, int(before_messages))
        return before, max(0, total - before)

    return before_messages, after_messages


@dataclass(frozen=True)
class DiscordDocument:
    source_url: str
    label: str
    trace_path: str
    guild_id: str
    channel_id: str
    channel_name: str | None
    channel_type: int | None
    thread_id: str | None
    thread_name: str | None
    parent_channel_id: str | None
    kind: str
    messages: list[dict[str, Any]]
    rendered: str
    guild_name: str | None = None
    parent_channel_name: str | None = None


@dataclass
class DiscordReference:
    url: str
    document: DiscordDocument
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
                text, self.depth, self.trace_collector, self.url
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

    @property
    def trace_path(self) -> str:
        return self.document.trace_path

    def get_contents(self) -> str:
        return self.output


def _discord_settings_from_env() -> DiscordSettings:
    _load_dotenv()

    raw_format = (os.environ.get("DISCORD_FORMAT") or "idiomatic").strip().lower()
    out_format = raw_format if raw_format in _VALID_FORMATS else "idiomatic"

    include_system = _parse_bool(
        os.environ.get("DISCORD_INCLUDE_SYSTEM", "1"), default=True
    )
    include_thread_starters = _parse_bool(
        os.environ.get("DISCORD_INCLUDE_THREAD_STARTERS", "1"),
        default=True,
    )
    expand_threads = _parse_bool(
        os.environ.get("DISCORD_EXPAND_THREADS", "0"), default=False
    )
    media_desc = _parse_bool(os.environ.get("DISCORD_MEDIA_DESC", "1"), default=True)
    embed_media_desc = _parse_bool(
        os.environ.get("DISCORD_EMBED_MEDIA_DESC", "1"),
        default=True,
    )

    around_messages = _parse_optional_int(os.environ.get("DISCORD_AROUND_MESSAGES", ""))
    before_messages = _parse_optional_int(os.environ.get("DISCORD_BEFORE_MESSAGES", ""))
    after_messages = _parse_optional_int(os.environ.get("DISCORD_AFTER_MESSAGES", ""))
    message_context = _parse_optional_int(os.environ.get("DISCORD_MESSAGE_CONTEXT", ""))
    if around_messages is not None:
        before_messages = around_messages
        after_messages = around_messages
    else:
        before_messages, after_messages = _distribute_message_context(
            message_context=message_context,
            before_messages=before_messages,
            after_messages=after_messages,
        )

    start = _parse_iso_datetime(os.environ.get("DISCORD_START_TIMESTAMP", ""))
    end = _parse_iso_datetime(os.environ.get("DISCORD_END_TIMESTAMP", ""))
    hard_min_timestamp = _parse_iso_datetime(
        os.environ.get("DISCORD_MIN_TIMESTAMP", "")
    )
    hard_max_timestamp = _parse_iso_datetime(
        os.environ.get("DISCORD_MAX_TIMESTAMP", "")
    )

    before_duration = _parse_duration_compound(
        os.environ.get("DISCORD_BEFORE_DURATION", "")
    )
    after_duration = _parse_duration_compound(
        os.environ.get("DISCORD_AFTER_DURATION", "")
    )
    around_duration = _parse_duration_compound(
        os.environ.get("DISCORD_AROUND_DURATION", "")
    )

    gap_threshold = _parse_gap_threshold(
        os.environ.get("DISCORD_GAP_THRESHOLD", "6h"),
        default=timedelta(hours=6),
    )

    channel_limit_raw = _parse_optional_int(os.environ.get("DISCORD_CHANNEL_LIMIT", ""))
    channel_limit = (
        channel_limit_raw if channel_limit_raw and channel_limit_raw > 0 else 200
    )

    settings = DiscordSettings(
        format=out_format,
        include_system=include_system,
        include_thread_starters=include_thread_starters,
        expand_threads=expand_threads,
        gap_threshold=gap_threshold,
        before_messages=before_messages if before_messages is not None else 0,
        after_messages=after_messages if after_messages is not None else 0,
        around_messages=around_messages,
        message_context=message_context,
        channel_limit=channel_limit,
        start=_normalize_datetime(start),
        end=_normalize_datetime(end),
        before_duration=before_duration,
        after_duration=after_duration,
        around_duration=around_duration,
        include_media_descriptions=media_desc,
        include_embed_media_descriptions=embed_media_desc,
        hard_min_timestamp=_normalize_datetime(hard_min_timestamp),
        hard_max_timestamp=_normalize_datetime(hard_max_timestamp),
    )
    return _clamp_settings(settings)


def _clamp_settings(settings: DiscordSettings) -> DiscordSettings:
    start = settings.start
    end = settings.end
    hard_min = settings.hard_min_timestamp
    hard_max = settings.hard_max_timestamp

    if hard_min and (start is None or start < hard_min):
        start = hard_min
    if hard_max and (end is None or end > hard_max):
        end = hard_max
    if start and end and start > end:
        start = end

    return DiscordSettings(
        format=settings.format,
        include_system=settings.include_system,
        include_thread_starters=settings.include_thread_starters,
        expand_threads=settings.expand_threads,
        gap_threshold=settings.gap_threshold,
        before_messages=settings.before_messages,
        after_messages=settings.after_messages,
        around_messages=settings.around_messages,
        message_context=settings.message_context,
        channel_limit=settings.channel_limit,
        start=start,
        end=end,
        before_duration=settings.before_duration,
        after_duration=settings.after_duration,
        around_duration=settings.around_duration,
        include_media_descriptions=settings.include_media_descriptions,
        include_embed_media_descriptions=settings.include_embed_media_descriptions,
        hard_min_timestamp=hard_min,
        hard_max_timestamp=hard_max,
    )


def build_discord_settings(overrides: dict[str, Any] | None = None) -> DiscordSettings:
    env = _discord_settings_from_env()
    if not overrides:
        return env

    format_value = str(overrides.get("format", env.format)).strip().lower()
    if format_value not in _VALID_FORMATS:
        format_value = env.format

    include_system = bool(overrides.get("include_system", env.include_system))
    include_thread_starters = bool(
        overrides.get("include_thread_starters", env.include_thread_starters)
    )
    expand_threads = bool(overrides.get("expand_threads", env.expand_threads))
    include_media_descriptions = bool(
        overrides.get("include_media_descriptions", env.include_media_descriptions)
    )
    include_embed_media_descriptions = bool(
        overrides.get(
            "include_embed_media_descriptions",
            env.include_embed_media_descriptions,
        )
    )

    before_messages = overrides.get("before_messages", env.before_messages)
    after_messages = overrides.get("after_messages", env.after_messages)
    around_messages = overrides.get("around_messages", env.around_messages)
    message_context = overrides.get("message_context", env.message_context)
    if around_messages is not None:
        before_messages = around_messages
        after_messages = around_messages
    else:
        before_messages, after_messages = _distribute_message_context(
            message_context=message_context,
            before_messages=before_messages,
            after_messages=after_messages,
        )

    channel_limit = overrides.get("channel_limit", env.channel_limit)
    try:
        channel_limit_int = int(channel_limit)
    except (TypeError, ValueError):
        channel_limit_int = env.channel_limit
    if channel_limit_int <= 0:
        channel_limit_int = env.channel_limit

    gap_threshold = overrides.get("gap_threshold", env.gap_threshold)
    if not isinstance(gap_threshold, timedelta) or gap_threshold.total_seconds() <= 0:
        gap_threshold = env.gap_threshold

    start = overrides.get("start", env.start)
    end = overrides.get("end", env.end)
    start = _normalize_datetime(start)
    end = _normalize_datetime(end)

    before_duration = overrides.get("before_duration", env.before_duration)
    after_duration = overrides.get("after_duration", env.after_duration)
    around_duration = overrides.get("around_duration", env.around_duration)

    result = DiscordSettings(
        format=format_value,
        include_system=include_system,
        include_thread_starters=include_thread_starters,
        expand_threads=expand_threads,
        gap_threshold=gap_threshold,
        before_messages=before_messages,
        after_messages=after_messages,
        around_messages=around_messages,
        message_context=message_context,
        channel_limit=channel_limit_int,
        start=start,
        end=end,
        before_duration=before_duration,
        after_duration=after_duration,
        around_duration=around_duration,
        include_media_descriptions=include_media_descriptions,
        include_embed_media_descriptions=include_embed_media_descriptions,
        hard_min_timestamp=env.hard_min_timestamp,
        hard_max_timestamp=env.hard_max_timestamp,
    )
    return _clamp_settings(result)


def _api_base() -> str:
    _load_dotenv()
    return (
        (os.environ.get("DISCORD_API_BASE") or "https://discord.com/api/v10")
        .strip()
        .rstrip("/")
    )


def _api_timeout_seconds() -> float:
    raw = (os.environ.get("DISCORD_API_TIMEOUT") or "").strip()
    if not raw:
        return 30.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def _api_max_attempts() -> int:
    raw = (os.environ.get("DISCORD_API_MAX_ATTEMPTS") or "").strip()
    if not raw:
        return 6
    try:
        return max(1, int(raw))
    except ValueError:
        return 6


def _retry_delay_seconds(attempt: int) -> float:
    import random

    base = min(20.0, 1.0 * (2 ** max(0, attempt - 1)))
    return base + random.uniform(0.0, 0.35)


def _retry_after_seconds(resp: object, attempt: int) -> float:
    headers = getattr(resp, "headers", None) or {}
    retry_after_raw = headers.get("Retry-After")
    if retry_after_raw:
        try:
            return max(0.0, float(retry_after_raw))
        except ValueError:
            pass
    return _retry_delay_seconds(attempt)


def _get_auth_headers() -> dict[str, str]:
    _load_dotenv()
    token = (os.environ.get("DISCORD_BOT_TOKEN") or "").strip()
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN is required for Discord URLs")
    return {
        "Authorization": f"Bot {token}",
        "Accept": "application/json",
        "User-Agent": "contextualize/discord (+https://github.com/jmpaz/contextualize)",
    }


def _api_cache_identity(path: str, params: dict[str, Any] | None) -> str:
    payload = {
        "base": _api_base(),
        "path": path,
        "params": dict(sorted((params or {}).items())),
    }
    return "discord-api:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _api_get(
    path: str,
    params: dict[str, Any] | None = None,
    *,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> Any:
    import requests

    from ..cache.discord import get_cached_api_json, store_api_json

    identity = _api_cache_identity(path, params)
    if use_cache and not refresh_cache:
        cached = get_cached_api_json(identity, ttl=cache_ttl)
        if cached is not None:
            return cached

    url = f"{_api_base()}{path}"
    timeout = _api_timeout_seconds()
    max_attempts = _api_max_attempts()
    headers = _get_auth_headers()
    transient_statuses = {429, 500, 502, 503, 504}

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                url, headers=headers, params=params, timeout=timeout
            )
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise
            wait = _retry_delay_seconds(attempt)
            _log(
                f"  Discord request failed ({type(exc).__name__}); retrying in {wait:.1f}s "
                f"(attempt {attempt}/{max_attempts})"
            )
            time.sleep(wait)
            continue

        if response.status_code == 404:
            raise ValueError(f"Discord resource not found: {path}")
        if response.status_code in {401, 403}:
            message = "Discord API authorization failed"
            if response.status_code == 403:
                message = "Discord API forbidden (guild-only mode or insufficient permissions)"
            raise ValueError(f"{message}: {path}")

        if response.status_code in transient_statuses and attempt < max_attempts:
            wait = _retry_after_seconds(response, attempt)
            _log(
                f"  Discord API returned {response.status_code}; retrying in {wait:.1f}s "
                f"(attempt {attempt}/{max_attempts})"
            )
            time.sleep(wait)
            continue

        response.raise_for_status()
        payload = response.json()
        if use_cache:
            store_api_json(identity, payload)
        return payload

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Discord API request failed unexpectedly: {path}")


def _fetch_channel(
    channel_id: str,
    *,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> dict[str, Any]:
    payload = _api_get(
        f"/channels/{channel_id}",
        None,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected channel payload for channel {channel_id}")
    return payload


def _fetch_guild(
    guild_id: str,
    *,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> dict[str, Any]:
    payload = _api_get(
        f"/guilds/{guild_id}",
        None,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected guild payload for guild {guild_id}")
    return payload


def _fetch_message(
    channel_id: str,
    message_id: str,
    *,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> dict[str, Any]:
    payload = _api_get(
        f"/channels/{channel_id}/messages/{message_id}",
        None,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected message payload for message {message_id}")
    return payload


def _fetch_messages_page(
    channel_id: str,
    *,
    limit: int,
    before: str | None,
    after: str | None,
    around: str | None,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": min(100, max(1, int(limit)))}
    if before:
        params["before"] = before
    if after:
        params["after"] = after
    if around:
        params["around"] = around

    payload = _api_get(
        f"/channels/{channel_id}/messages",
        params,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    if not isinstance(payload, list):
        return []
    return [m for m in payload if isinstance(m, dict)]


def _message_datetime(message: dict[str, Any]) -> datetime | None:
    raw = message.get("timestamp")
    if not isinstance(raw, str):
        return None
    return _parse_iso_datetime(raw)


def _sort_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for message in messages:
        mid = str(message.get("id") or "")
        if not mid:
            continue
        deduped[mid] = message

    def _key(msg: dict[str, Any]) -> tuple[float, str]:
        ts = _message_datetime(msg)
        return ((ts.timestamp() if ts else 0.0), str(msg.get("id") or ""))

    return sorted(deduped.values(), key=_key)


def _fetch_previous_messages(
    channel_id: str,
    anchor_id: str,
    count: int,
    *,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    out: list[dict[str, Any]] = []
    before = anchor_id
    while len(out) < count:
        remaining = count - len(out)
        page = _fetch_messages_page(
            channel_id,
            limit=min(100, remaining),
            before=before,
            after=None,
            around=None,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        if not page:
            break
        out.extend(page)
        oldest = page[-1]
        before = str(oldest.get("id") or "")
        if not before:
            break
    return out[:count]


def _fetch_next_messages(
    channel_id: str,
    anchor_id: str,
    count: int,
    *,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    out: list[dict[str, Any]] = []
    after = anchor_id
    while len(out) < count:
        remaining = count - len(out)
        page = _fetch_messages_page(
            channel_id,
            limit=min(100, remaining),
            before=None,
            after=after,
            around=None,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        if not page:
            break
        out.extend(page)
        newest = page[0]
        after = str(newest.get("id") or "")
        if not after:
            break
    return out[:count]


def _fetch_latest_messages(
    channel_id: str,
    limit: int,
    *,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    out: list[dict[str, Any]] = []
    before: str | None = None
    while len(out) < limit:
        remaining = limit - len(out)
        page = _fetch_messages_page(
            channel_id,
            limit=min(100, remaining),
            before=before,
            after=None,
            around=None,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        if not page:
            break
        out.extend(page)
        oldest = page[-1]
        before = str(oldest.get("id") or "")
        if not before:
            break
    return out[:limit]


def _fetch_messages_between(
    channel_id: str,
    start: datetime | None,
    end: datetime | None,
    *,
    max_messages: int,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> list[dict[str, Any]]:
    target_cap = max(1, max_messages)
    out: list[dict[str, Any]] = []
    before: str | None = None

    while len(out) < target_cap:
        page = _fetch_messages_page(
            channel_id,
            limit=min(100, target_cap - len(out)),
            before=before,
            after=None,
            around=None,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        if not page:
            break

        should_stop = False
        for message in page:
            ts = _message_datetime(message)
            if ts is None:
                continue
            if end and ts > end:
                continue
            if start and ts < start:
                should_stop = True
                continue
            out.append(message)
            if len(out) >= target_cap:
                break

        if len(out) >= target_cap:
            break

        oldest = page[-1]
        before = str(oldest.get("id") or "")
        if not before:
            break

        if should_stop:
            break

    return _sort_messages(out)


def _channel_label(channel: dict[str, Any], channel_id: str) -> str:
    name = channel.get("name")
    if isinstance(name, str) and name.strip():
        return _sanitize_text(name).strip()
    return channel_id


def _guild_label(guild: dict[str, Any], guild_id: str) -> str:
    name = guild.get("name")
    if isinstance(name, str) and name.strip():
        return _sanitize_text(name).strip()
    return guild_id


def _sanitize_text(value: str) -> str:
    return "".join(
        ch
        for ch in value
        if ch in {"\n", "\r", "\t"} or (0x20 <= ord(ch) and ord(ch) != 0x7F)
    )


def _normalize_display_text(value: str) -> str:
    return _MARKDOWN_ESCAPED_PUNCT_RE.sub(r"\1", _sanitize_text(value))


def _author_name(message: dict[str, Any]) -> str:
    member = message.get("member") if isinstance(message.get("member"), dict) else {}
    author = message.get("author") if isinstance(message.get("author"), dict) else {}
    for key in ("nick", "global_name", "display_name", "username"):
        value = member.get(key) if key in member else author.get(key)
        if isinstance(value, str) and value.strip():
            return _sanitize_text(value).strip()
    return "unknown"


def _author_id(message: dict[str, Any]) -> str | None:
    author = message.get("author") if isinstance(message.get("author"), dict) else {}
    value = author.get("id")
    if value is None:
        return None
    return str(value)


def _embed_text_blob(embed: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("type", "title", "description", "url", "timestamp"):
        value = embed.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}: {value.strip()}")
    author = embed.get("author") if isinstance(embed.get("author"), dict) else None
    if author:
        name = author.get("name")
        if isinstance(name, str) and name.strip():
            parts.append(f"author: {name.strip()}")
    provider = (
        embed.get("provider") if isinstance(embed.get("provider"), dict) else None
    )
    if provider:
        name = provider.get("name")
        if isinstance(name, str) and name.strip():
            parts.append(f"provider: {name.strip()}")
    footer = embed.get("footer") if isinstance(embed.get("footer"), dict) else None
    if footer:
        text = footer.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(f"footer: {text.strip()}")
    fields = embed.get("fields") if isinstance(embed.get("fields"), list) else []
    for index, field in enumerate(fields, 1):
        if not isinstance(field, dict):
            continue
        name = field.get("name")
        value = field.get("value")
        if isinstance(name, str) and name.strip():
            parts.append(f"field[{index}].name: {name.strip()}")
        if isinstance(value, str) and value.strip():
            parts.append(f"field[{index}].value: {value.strip()}")
    return "\n".join(parts)


def _media_prompt_append_from_embed(embed: dict[str, Any]) -> str | None:
    blob = _embed_text_blob(embed)
    if not blob:
        return None
    return f"\n\nEmbed context:\n```text\n{blob}\n```"


def _describe_remote_media(
    url: str,
    *,
    prompt_append: str | None = None,
    media_cache_identity: str | None = None,
    suffix: str = "",
    send_label: str | None = None,
) -> str | None:
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return None

    from ..cache.discord import get_cached_media_bytes, store_media_bytes
    from ..render.markitdown import MarkItDownConversionError, convert_path_to_markdown
    from ..runtime import get_refresh_cache, get_refresh_images
    from .media import download_cached_media_to_temp

    cache_identity = media_cache_identity or url
    cache_label = send_label or cache_identity
    tmp = download_cached_media_to_temp(
        url,
        suffix=suffix,
        headers=_MEDIA_DOWNLOAD_HEADERS,
        cache_identity=cache_identity,
        get_cached_media_bytes=get_cached_media_bytes,
        store_media_bytes=store_media_bytes,
        refresh_cache=get_refresh_cache(),
        on_cache_hit=lambda _identity: _log(
            f"  discord media cache hit: {cache_label}"
        ),
        on_cache_miss=lambda _identity: _log(
            f"  discord media cache miss: {cache_label}"
        ),
    )
    if tmp is None:
        return None

    try:
        refresh_images = get_refresh_images()
        result = convert_path_to_markdown(
            str(tmp),
            refresh_images=refresh_images,
            prompt_append=prompt_append,
        )
    except MarkItDownConversionError:
        return None
    finally:
        tmp.unlink(missing_ok=True)

    markdown = (result.markdown or "").strip()
    if not markdown:
        return None
    return markdown


def _normalize_attachment_nodes(
    message: dict[str, Any],
    *,
    include_media_descriptions: bool,
    include_embed_media_descriptions: bool,
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    message_id_raw = message.get("id")
    message_id = str(message_id_raw) if message_id_raw is not None else None

    attachments = (
        message.get("attachments")
        if isinstance(message.get("attachments"), list)
        else []
    )
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        filename = str(attachment.get("filename") or "")
        content_type = str(attachment.get("content_type") or "")
        kind = _parse_media_kind(filename=filename, content_type=content_type)
        attachment_url = str(attachment.get("url") or "")
        node: dict[str, Any] = {
            "type": kind,
            "filename": filename,
            "url": attachment_url or None,
            "content_type": content_type or None,
            "size": attachment.get("size"),
            "width": attachment.get("width"),
            "height": attachment.get("height"),
            "description": attachment.get("description"),
        }
        if include_media_descriptions and kind in {"image", "video", "audio", "file"}:
            attachment_id_raw = attachment.get("id")
            attachment_id = (
                str(attachment_id_raw) if attachment_id_raw is not None else None
            )
            media_cache_identity = _attachment_media_cache_identity(
                message_id=message_id,
                attachment_id=attachment_id,
                filename=filename,
                url=attachment_url,
            )
            media_desc = _describe_remote_media(
                attachment_url,
                media_cache_identity=media_cache_identity,
                suffix=_media_suffix(
                    filename=filename,
                    content_type=content_type,
                    kind=kind,
                    url=attachment_url,
                ),
                send_label=(
                    f"msg:{message_id or 'unknown'}:attachment:{attachment_id or filename or kind}"
                ),
            )
            if media_desc:
                node["media_description"] = media_desc
        cleaned = {k: v for k, v in node.items() if v not in (None, "", [])}
        nodes.append(cleaned)

    embeds = message.get("embeds") if isinstance(message.get("embeds"), list) else []
    for embed_index, embed in enumerate(embeds):
        if not isinstance(embed, dict):
            continue
        embed_node: dict[str, Any] = {
            "type": "embed",
            "embed_type": embed.get("type"),
            "url": embed.get("url"),
            "title": embed.get("title"),
            "description": embed.get("description"),
            "timestamp": embed.get("timestamp"),
        }
        media_entries: list[dict[str, Any]] = []
        embed_prompt_append = _media_prompt_append_from_embed(embed)

        for key, media_kind in (
            ("image", "image"),
            ("thumbnail", "image"),
            ("video", "video"),
        ):
            media_obj = embed.get(key) if isinstance(embed.get(key), dict) else None
            if not media_obj:
                continue
            media_url = media_obj.get("url") or media_obj.get("proxy_url")
            if not isinstance(media_url, str) or not media_url:
                continue
            media_entry: dict[str, Any] = {
                "type": media_kind,
                "role": key,
                "url": media_url,
                "width": media_obj.get("width"),
                "height": media_obj.get("height"),
            }
            if include_media_descriptions and include_embed_media_descriptions:
                media_cache_identity = _embed_media_cache_identity(
                    message_id=message_id,
                    embed_index=embed_index,
                    role=key,
                    url=media_url,
                )
                media_desc = _describe_remote_media(
                    media_url,
                    prompt_append=embed_prompt_append,
                    media_cache_identity=media_cache_identity,
                    suffix=_media_suffix(
                        filename="",
                        content_type="",
                        kind=media_kind,
                        url=media_url,
                    ),
                    send_label=(
                        f"msg:{message_id or 'unknown'}:embed:{embed_index}:{key}"
                    ),
                )
                if media_desc:
                    media_entry["media_description"] = media_desc
            media_entries.append(
                {k: v for k, v in media_entry.items() if v not in (None, "", [])}
            )

        fields = embed.get("fields") if isinstance(embed.get("fields"), list) else []
        cleaned_fields: list[dict[str, Any]] = []
        for field in fields:
            if not isinstance(field, dict):
                continue
            cleaned = {
                "name": field.get("name"),
                "value": field.get("value"),
                "inline": field.get("inline"),
            }
            cleaned_fields.append(
                {k: v for k, v in cleaned.items() if v not in (None, "", [])}
            )

        if cleaned_fields:
            embed_node["fields"] = cleaned_fields
        if media_entries:
            embed_node["media"] = media_entries

        author = embed.get("author") if isinstance(embed.get("author"), dict) else None
        provider = (
            embed.get("provider") if isinstance(embed.get("provider"), dict) else None
        )
        footer = embed.get("footer") if isinstance(embed.get("footer"), dict) else None
        if author:
            embed_node["author"] = {
                k: v
                for k, v in {
                    "name": author.get("name"),
                    "url": author.get("url"),
                    "icon_url": author.get("icon_url"),
                }.items()
                if v not in (None, "")
            }
        if provider:
            embed_node["provider"] = {
                k: v
                for k, v in {
                    "name": provider.get("name"),
                    "url": provider.get("url"),
                }.items()
                if v not in (None, "")
            }
        if footer:
            embed_node["footer"] = {
                k: v
                for k, v in {
                    "text": footer.get("text"),
                    "icon_url": footer.get("icon_url"),
                }.items()
                if v not in (None, "")
            }

        nodes.append({k: v for k, v in embed_node.items() if v not in (None, "", [])})

    return nodes


def _system_content(message: dict[str, Any], *, thread_id: str | None) -> str:
    msg_type = int(message.get("type") or 0)
    if msg_type == 19:
        return ""
    if msg_type == _DISCORD_THREAD_STARTER_MESSAGE_TYPE:
        if thread_id:
            return "[created thread]"
        return "[created channel]"
    return f"[system event type {msg_type}]"


def _reply_to_id(message: dict[str, Any]) -> str | None:
    reference = (
        message.get("message_reference")
        if isinstance(message.get("message_reference"), dict)
        else {}
    )
    message_id = reference.get("message_id")
    if message_id is None:
        return None
    return str(message_id)


def _build_message_lookup(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for message in messages:
        message_id = message.get("id")
        if message_id is None:
            continue
        out[str(message_id)] = message
    return out


def _reply_to_content(
    message: dict[str, Any],
    *,
    reply_to_id: str | None,
    message_lookup: dict[str, dict[str, Any]] | None,
) -> str | None:
    if not reply_to_id:
        return None
    referenced = (
        message.get("referenced_message")
        if isinstance(message.get("referenced_message"), dict)
        else None
    )
    if referenced is not None:
        content = _extract_message_content(referenced)
        if content:
            return content
    if message_lookup is None:
        return None
    reply_to_message = message_lookup.get(reply_to_id)
    if reply_to_message is None:
        return None
    content = _extract_message_content(reply_to_message)
    if not content:
        return None
    return content


def _snapshot_messages(message: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots = (
        message.get("message_snapshots")
        if isinstance(message.get("message_snapshots"), list)
        else []
    )
    out: list[dict[str, Any]] = []
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        snapshot_message = snapshot.get("message")
        if isinstance(snapshot_message, dict):
            out.append(snapshot_message)
    return out


def _extract_message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return _normalize_display_text(content)
    for snapshot in _snapshot_messages(message):
        snap_content = snapshot.get("content")
        if isinstance(snap_content, str) and snap_content.strip():
            return _normalize_display_text(snap_content)
    return ""


def _message_with_snapshot_media(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("attachments") or message.get("embeds"):
        return message
    snapshots = _snapshot_messages(message)
    if not snapshots:
        return message

    attachments: list[dict[str, Any]] = []
    embeds: list[dict[str, Any]] = []
    for snapshot in snapshots:
        snapshot_attachments = (
            snapshot.get("attachments")
            if isinstance(snapshot.get("attachments"), list)
            else []
        )
        snapshot_embeds = (
            snapshot.get("embeds") if isinstance(snapshot.get("embeds"), list) else []
        )
        attachments.extend(
            item for item in snapshot_attachments if isinstance(item, dict)
        )
        embeds.extend(item for item in snapshot_embeds if isinstance(item, dict))

    if not attachments and not embeds:
        return message

    merged = dict(message)
    if attachments:
        merged["attachments"] = attachments
    if embeds:
        merged["embeds"] = embeds
    return merged


def _forward_reference_fields(
    message: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    reference = (
        message.get("message_reference")
        if isinstance(message.get("message_reference"), dict)
        else {}
    )
    guild_id_raw = reference.get("guild_id")
    channel_id_raw = reference.get("channel_id")
    message_id_raw = reference.get("message_id")
    guild_id = str(guild_id_raw) if guild_id_raw is not None else None
    channel_id = str(channel_id_raw) if channel_id_raw is not None else None
    message_id = str(message_id_raw) if message_id_raw is not None else None
    return guild_id, channel_id, message_id


def _is_forwarded_message(message: dict[str, Any]) -> bool:
    return bool(_snapshot_messages(message))


def _forward_date_from_snapshots(message: dict[str, Any]) -> str | None:
    for snapshot in _snapshot_messages(message):
        timestamp = snapshot.get("timestamp")
        normalized = _format_embed_date_utc(timestamp)
        if normalized:
            return normalized
    return None


def _forward_url(message: dict[str, Any]) -> str | None:
    guild_id, channel_id, message_id = _forward_reference_fields(message)
    if not channel_id or not message_id:
        return None
    guild_segment = guild_id or "@me"
    return f"https://discord.com/channels/{guild_segment}/{channel_id}/{message_id}"


def _build_forward_source_lookup(
    messages: list[dict[str, Any]],
    *,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> dict[tuple[str, str], str]:
    refs: set[tuple[str, str]] = set()
    for message in messages:
        if not _is_forwarded_message(message):
            continue
        guild_id, channel_id, _ = _forward_reference_fields(message)
        if not channel_id:
            continue
        refs.add((guild_id or "", channel_id))

    out: dict[tuple[str, str], str] = {}
    for guild_id, channel_id in refs:
        try:
            channel = _fetch_channel(
                channel_id,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
            )
        except Exception:
            continue
        out[(guild_id, channel_id)] = _channel_label(channel, channel_id)
    return out


def _forward_metadata(
    message: dict[str, Any],
    *,
    forward_source_lookup: dict[tuple[str, str], str] | None,
) -> dict[str, str] | None:
    if not _is_forwarded_message(message):
        return None

    metadata: dict[str, str] = {}
    date = _forward_date_from_snapshots(message)
    if date:
        metadata["date"] = date

    url = _forward_url(message)
    if url:
        metadata["url"] = url

    guild_id, channel_id, _ = _forward_reference_fields(message)
    if channel_id and forward_source_lookup is not None:
        src = forward_source_lookup.get((guild_id or "", channel_id))
        if src:
            metadata["src"] = src

    return metadata


def _normalize_message(
    message: dict[str, Any],
    *,
    guild_id: str,
    channel_id: str,
    thread_id: str | None,
    include_media_descriptions: bool,
    include_embed_media_descriptions: bool,
    message_lookup: dict[str, dict[str, Any]] | None = None,
    forward_source_lookup: dict[tuple[str, str], str] | None = None,
) -> dict[str, Any] | None:
    message_id = message.get("id")
    if not message_id:
        return None
    timestamp_raw = message.get("timestamp")
    timestamp = (
        _parse_iso_datetime(timestamp_raw) if isinstance(timestamp_raw, str) else None
    )
    if timestamp is None:
        return None

    content = _extract_message_content(message)
    message_type = int(message.get("type") or 0)
    if not content and message_type != 0:
        content = _system_content(message, thread_id=thread_id)

    message_with_media = _message_with_snapshot_media(message)
    attachments = _normalize_attachment_nodes(
        message_with_media,
        include_media_descriptions=include_media_descriptions,
        include_embed_media_descriptions=include_embed_media_descriptions,
    )

    reply_to_id = _reply_to_id(message)
    reply_to_content = _reply_to_content(
        message,
        reply_to_id=reply_to_id,
        message_lookup=message_lookup,
    )
    thread_data = (
        message.get("thread") if isinstance(message.get("thread"), dict) else {}
    )
    forward = _forward_metadata(
        message,
        forward_source_lookup=forward_source_lookup,
    )

    normalized = {
        "id": str(message_id),
        "timestamp": timestamp.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "discord_type": message_type,
        "sender": _author_name(message),
        "sender_id": _author_id(message),
        "content": _sanitize_text(content),
        "reply_to_id": reply_to_id,
        "reply_to_content": _sanitize_text(reply_to_content)
        if reply_to_content
        else None,
        "guild_id": guild_id,
        "channel_id": channel_id,
        "thread_id": thread_id,
        "attachments": attachments,
        "forward": forward,
        "is_thread_starter": bool(thread_data.get("id")),
        "thread_starter_id": str(thread_data.get("id"))
        if thread_data.get("id")
        else None,
    }
    return {k: v for k, v in normalized.items() if v not in (None, [])}


def _is_system_message(message: dict[str, Any]) -> bool:
    msg_type = int(message.get("type") or 0)
    return msg_type not in _NON_SYSTEM_MESSAGE_TYPES


def _filter_messages(
    messages: list[dict[str, Any]], *, settings: DiscordSettings
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if not settings.include_system and _is_system_message(message):
            continue
        if not settings.include_thread_starters:
            thread_data = (
                message.get("thread")
                if isinstance(message.get("thread"), dict)
                else None
            )
            if thread_data and thread_data.get("id"):
                continue
        out.append(message)
    return _sort_messages(out)


def _time_window_enabled(settings: DiscordSettings) -> bool:
    return any(
        value is not None
        for value in (
            settings.start,
            settings.end,
            settings.before_duration,
            settings.after_duration,
            settings.around_duration,
        )
    )


def _window_bounds_for_target(
    timestamp: datetime, settings: DiscordSettings
) -> tuple[datetime | None, datetime | None]:
    start = settings.start
    end = settings.end

    if settings.around_duration is not None:
        start = timestamp - settings.around_duration
        end = timestamp + settings.around_duration
    else:
        if settings.before_duration is not None:
            candidate = timestamp - settings.before_duration
            start = candidate if start is None or candidate > start else start
        if settings.after_duration is not None:
            candidate = timestamp + settings.after_duration
            end = candidate if end is None or candidate < end else end

    if settings.hard_min_timestamp and (
        start is None or start < settings.hard_min_timestamp
    ):
        start = settings.hard_min_timestamp
    if settings.hard_max_timestamp and (
        end is None or end > settings.hard_max_timestamp
    ):
        end = settings.hard_max_timestamp
    if start and end and start > end:
        start = end

    return start, end


def _window_bounds_no_target(
    settings: DiscordSettings,
) -> tuple[datetime | None, datetime | None]:
    now = datetime.now(timezone.utc)
    anchor = now
    start = settings.start
    end = settings.end

    if settings.around_duration is not None:
        start = anchor - settings.around_duration
        end = anchor + settings.around_duration
    else:
        if settings.before_duration is not None:
            candidate = anchor - settings.before_duration
            start = candidate if start is None or candidate > start else start
        if settings.after_duration is not None:
            candidate = anchor + settings.after_duration
            end = candidate if end is None or candidate < end else end

    if settings.hard_min_timestamp and (
        start is None or start < settings.hard_min_timestamp
    ):
        start = settings.hard_min_timestamp
    if settings.hard_max_timestamp and (
        end is None or end > settings.hard_max_timestamp
    ):
        end = settings.hard_max_timestamp
    if start and end and start > end:
        start = end

    return start, end


def _message_window_counts(settings: DiscordSettings) -> tuple[int, int]:
    before = settings.before_messages if settings.before_messages is not None else 0
    after = settings.after_messages if settings.after_messages is not None else 0
    before = max(0, int(before))
    after = max(0, int(after))
    return before, after


def _escape_xml_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;")


def _format_embed_date_utc(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    parsed = _parse_iso_datetime(raw)
    if parsed is None:
        return None
    return (
        parsed.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _format_dimensions(media: dict[str, Any]) -> str | None:
    width = media.get("width")
    height = media.get("height")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return f"{width}x{height}"
    return None


def _normalize_media_description(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and re.fullmatch(r"ImageSize:\s*\d+x\d+", lines[0].strip(), flags=re.I):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip().lower() in {
        "# description (auto-generated):",
        "description (auto-generated):",
    }:
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return _normalize_display_text("\n".join(lines))


def _format_scope_header(document: DiscordDocument) -> str:
    guild = _sanitize_text(document.guild_name or document.guild_id).strip()
    if document.thread_id:
        parent = _sanitize_text(
            document.parent_channel_name
            or document.parent_channel_id
            or document.channel_name
            or document.channel_id
        ).strip()
        thread = _sanitize_text(document.thread_name or document.thread_id).strip()
        escaped_thread = thread.replace("\\", "\\\\").replace('"', '\\"')
        return f'{guild} → #{parent} → "{escaped_thread}"'
    channel = _sanitize_text(document.channel_name or document.channel_id).strip()
    return f"{guild} → #{channel}"


def _format_forward_open_tag(forward: dict[str, Any]) -> str:
    attrs: list[str] = []
    for key in ("date", "src", "url"):
        value = forward.get(key)
        if isinstance(value, str) and value.strip():
            attrs.append(f'{key}="{_escape_xml_attr(value.strip())}"')
    if not attrs:
        return "<forwarded-msg>"
    return f"<forwarded-msg {' '.join(attrs)}>"


def _format_attachment_idiomatic(
    node: dict[str, Any], *, indent: str = ""
) -> list[str]:
    lines: list[str] = []
    node_type = str(node.get("type") or "attachment")

    if node_type == "embed":
        url = node.get("url") or ""
        attrs: list[str] = []
        if isinstance(url, str) and url:
            attrs.append(f'url="{_escape_xml_attr(url)}"')
        embed_date = _format_embed_date_utc(node.get("timestamp"))
        if embed_date:
            attrs.append(f'date="{embed_date}"')
        attr_suffix = f" {' '.join(attrs)}" if attrs else ""
        lines.append(f"{indent}<embed{attr_suffix}>")
        title = node.get("title")
        if isinstance(title, str) and title.strip():
            lines.append(f"{indent}{_normalize_display_text(title).strip()}")
        description = node.get("description")
        if isinstance(description, str) and description.strip():
            lines.append(f"{indent}{_normalize_display_text(description).strip()}")
        media_items = node.get("media") if isinstance(node.get("media"), list) else []
        for media in media_items:
            if not isinstance(media, dict):
                continue
            media_type = media.get("type") or "media"
            media_url = media.get("url")
            dimensions = _format_dimensions(media)
            dimensions_attr = f' dimensions="{dimensions}"' if dimensions else ""
            media_desc = media.get("media_description")
            normalized_desc = (
                _normalize_media_description(media_desc)
                if isinstance(media_desc, str) and media_desc.strip()
                else ""
            )

            if media_type == "image":
                if normalized_desc:
                    lines.append(f"{indent}<image{dimensions_attr}>")
                    lines.extend(
                        f"{indent}{line}" for line in normalized_desc.splitlines()
                    )
                    lines.append(f"{indent}</image>")
                    continue
                attrs = []
                if isinstance(media_url, str) and media_url:
                    attrs.append(f'url="{_escape_xml_attr(media_url)}"')
                if dimensions:
                    attrs.append(f'dimensions="{dimensions}"')
                attr_suffix = f" {' '.join(attrs)}" if attrs else ""
                lines.append(f"{indent}<image{attr_suffix} />")
                continue

            attrs = []
            if isinstance(media_url, str) and media_url:
                attrs.append(f'url="{_escape_xml_attr(media_url)}"')
            if dimensions:
                attrs.append(f'dimensions="{dimensions}"')
            attr_suffix = f" {' '.join(attrs)}" if attrs else ""
            lines.append(f"{indent}<{media_type}{attr_suffix} />")
            if normalized_desc:
                lines.extend(f"{indent}{line}" for line in normalized_desc.splitlines())
        lines.append(f"{indent}</embed>")
        return lines

    url = str(node.get("url") or "")
    filename = str(node.get("filename") or "")
    dimensions = _format_dimensions(node)
    media_desc = node.get("media_description")
    normalized_desc = (
        _normalize_media_description(media_desc)
        if isinstance(media_desc, str) and media_desc.strip()
        else ""
    )

    attrs = [f'type="{_escape_xml_attr(node_type)}"']
    if node_type != "image" or not normalized_desc:
        if url:
            attrs.append(f'url="{_escape_xml_attr(url)}"')
    if filename:
        attrs.append(f'filename="{_escape_xml_attr(filename)}"')
    if dimensions:
        attrs.append(f'dimensions="{dimensions}"')

    if normalized_desc:
        lines.append(f"{indent}<attachment {' '.join(attrs)}>")
        lines.extend(f"{indent}{line}" for line in normalized_desc.splitlines())
        lines.append(f"{indent}</attachment>")
        return lines

    lines.append(f"{indent}<attachment {' '.join(attrs)} />")
    return lines


def _format_reply_quote(content: str) -> list[str]:
    normalized = _normalize_display_text(content).strip()
    if not normalized:
        return []
    if len(normalized) > _REPLY_QUOTE_MAX_CHARS:
        normalized = normalized[: _REPLY_QUOTE_MAX_CHARS - 1].rstrip() + "…"
    return [f"> {line}" if line else ">" for line in normalized.splitlines()]


def _render_idiomatic(document: DiscordDocument, settings: DiscordSettings) -> str:
    lines: list[str] = []
    lines.append(_format_scope_header(document))
    lines.append("---")

    previous_date: str | None = None
    previous_dt: datetime | None = None

    for message in document.messages:
        timestamp_raw = message.get("timestamp")
        timestamp = _parse_iso_datetime(str(timestamp_raw or ""))
        if timestamp is None:
            continue

        date_str = timestamp.date().isoformat()
        if previous_date != date_str:
            lines.append("")
            lines.append(f"<{date_str}>")
            previous_date = date_str
        elif previous_dt is not None:
            gap = timestamp - previous_dt
            if gap >= settings.gap_threshold:
                lines.append("")
                lines.append("---")

        sender = str(message.get("sender") or "unknown")
        content = str(message.get("content") or "")
        reply_to_content = (
            message.get("reply_to_content")
            if isinstance(message.get("reply_to_content"), str)
            else ""
        )
        attachments = (
            message.get("attachments")
            if isinstance(message.get("attachments"), list)
            else []
        )
        forward = (
            message.get("forward") if isinstance(message.get("forward"), dict) else None
        )
        lines.append(f"[{sender}]")
        if forward is not None:
            lines.append(_format_forward_open_tag(forward))
        if reply_to_content:
            lines.extend(_format_reply_quote(reply_to_content))
        if content:
            lines.append(content)
        elif not attachments and not reply_to_content:
            lines.append("")
        for node in attachments:
            if isinstance(node, dict):
                lines.extend(_format_attachment_idiomatic(node))
        if forward is not None:
            lines.append("</forwarded-msg>")

        lines.append("")
        previous_dt = timestamp

    rendered = "\n".join(lines).strip()
    return rendered + "\n"


class _LiteralString(str):
    pass


def _render_yaml(document: DiscordDocument, settings: DiscordSettings) -> str:
    import yaml

    class _DiscordDumper(yaml.SafeDumper):
        pass

    def _repr_literal(dumper, value):
        return dumper.represent_scalar("tag:yaml.org,2002:str", str(value), style="|")

    _DiscordDumper.add_representer(_LiteralString, _repr_literal)

    participants = sorted(
        {
            str(message.get("sender"))
            for message in document.messages
            if isinstance(message.get("sender"), str)
        }
    )

    stream: list[dict[str, Any]] = []
    previous_date: str | None = None
    previous_dt: datetime | None = None

    for message in document.messages:
        timestamp_raw = message.get("timestamp")
        timestamp = _parse_iso_datetime(str(timestamp_raw or ""))
        if timestamp is None:
            continue

        date_str = timestamp.date().isoformat()
        if previous_date != date_str:
            stream.append({"type": "timestamp", "date": date_str})
            previous_date = date_str
        elif previous_dt is not None:
            gap = timestamp - previous_dt
            if gap >= settings.gap_threshold:
                stream.append(
                    {
                        "type": "gap",
                        "duration": _format_duration_human(gap),
                    }
                )

        entry: dict[str, Any] = {
            "type": "message",
            "id": message.get("id"),
            "timestamp": message.get("timestamp"),
            "discord_type": message.get("discord_type"),
            "sender": message.get("sender"),
            "sender_id": message.get("sender_id"),
            "guild_id": message.get("guild_id"),
            "channel_id": message.get("channel_id"),
            "thread_id": message.get("thread_id"),
            "reply_to_id": message.get("reply_to_id"),
            "reply_to_content": message.get("reply_to_content"),
            "forward": message.get("forward"),
            "content": _LiteralString(str(message.get("content") or "")),
        }
        attachments = message.get("attachments")
        if isinstance(attachments, list) and attachments:
            entry["attachments"] = attachments
        stream.append({k: v for k, v in entry.items() if v is not None})
        previous_dt = timestamp

    payload: dict[str, Any] = {
        "platform": "Discord",
        "discord": {
            "guild_id": document.guild_id,
            "channel": {
                "id": document.channel_id,
                "name": document.channel_name,
                "type": document.channel_type,
            },
            "thread": (
                {
                    "id": document.thread_id,
                    "name": document.thread_name,
                    "parent_channel_id": document.parent_channel_id,
                }
                if document.thread_id
                else None
            ),
        },
        "participants": [{"alias": participant} for participant in participants],
        "stream": stream,
    }

    cleaned = {
        **payload,
        "discord": {
            "guild_id": document.guild_id,
            "channel": {
                k: v
                for k, v in payload["discord"]["channel"].items()
                if v not in (None, "")
            },
            **(
                {
                    "thread": {
                        k: v
                        for k, v in payload["discord"]["thread"].items()
                        if v not in (None, "")
                    }
                }
                if payload["discord"].get("thread")
                else {}
            ),
        },
    }

    return yaml.dump(
        cleaned,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        Dumper=_DiscordDumper,
    )


def _render_document(document: DiscordDocument, settings: DiscordSettings) -> str:
    if settings.format == "yaml":
        return _render_yaml(document, settings)
    return _render_idiomatic(document, settings)


def _document_label(
    *,
    guild_id: str,
    channel_id: str,
    parent_channel_id: str | None,
    thread_id: str | None,
    message_id: str | None,
) -> str:
    core: str
    if thread_id:
        parent = parent_channel_id or channel_id
        core = f"{guild_id}-{parent}/{thread_id}"
    else:
        core = f"{guild_id}-{channel_id}"
    if message_id:
        core = f"{core}/{message_id}"
    return f"discord:{core}"


def _document_trace_path(
    *,
    guild_id: str,
    channel_id: str,
    parent_channel_id: str | None,
    thread_id: str | None,
) -> str:
    if thread_id:
        parent = parent_channel_id or channel_id
        return f"discord/{guild_id}/{parent}/{thread_id}"
    return f"discord/{guild_id}/{channel_id}"


def _validate_guild_scope(guild_id: str) -> None:
    if guild_id == "@me":
        raise ValueError("Discord DM URLs are out of scope in guild-only mode")


def _resolve_message_url(
    url: str,
    parsed: dict[str, str],
    *,
    settings: DiscordSettings,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> list[DiscordDocument]:
    guild_id = parsed["guild_id"]
    channel_id = parsed["channel_id"]
    message_id = parsed["message_id"]

    _log(
        f"Resolving Discord message URL (guild={guild_id}, channel={channel_id}, message={message_id})"
    )
    _validate_guild_scope(guild_id)
    guild_name: str | None = None
    try:
        guild = _fetch_guild(
            guild_id,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        guild_name = _guild_label(guild, guild_id)
    except Exception:
        guild_name = None
    if guild_name:
        _log(f"  Resolved guild: {guild_name} ({guild_id})")
    else:
        _log(f"  Guild metadata unavailable for {guild_id}")

    channel = _fetch_channel(
        channel_id,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    channel_name = _channel_label(channel, channel_id)
    _log(f"  Resolved channel: #{channel_name} ({channel_id})")
    target = _fetch_message(
        channel_id,
        message_id,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    target_ts = _message_datetime(target)
    if target_ts is None:
        raise ValueError(f"Missing timestamp for Discord message {message_id}")

    if _time_window_enabled(settings):
        _log("  Fetching message context via configured time window")
        start, end = _window_bounds_for_target(target_ts, settings)
        fetched = _fetch_messages_between(
            channel_id,
            start,
            end,
            max_messages=max(200, settings.channel_limit * 2),
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        fetched_by_id = {str(msg.get("id") or ""): msg for msg in fetched}
        if message_id not in fetched_by_id:
            fetched.append(target)
        messages = _sort_messages(fetched)
    else:
        before_count, after_count = _message_window_counts(settings)
        _log(
            f"  Fetching message context around anchor (before={before_count}, after={after_count})"
        )
        previous = _fetch_previous_messages(
            channel_id,
            message_id,
            before_count,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        following = _fetch_next_messages(
            channel_id,
            message_id,
            after_count,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        messages = _sort_messages([*previous, target, *following])

    _log(f"  Retrieved {len(messages)} context message(s) before filtering")
    filtered = _filter_messages(messages, settings=settings)
    _log(f"  Kept {len(filtered)} context message(s) after filtering")

    thread_id: str | None = None
    parent_channel_id: str | None = None
    parent_channel_name: str | None = None
    thread_name: str | None = None
    channel_type_raw = channel.get("type")
    channel_type = int(channel_type_raw) if isinstance(channel_type_raw, int) else None
    if channel_type in _DISCORD_THREAD_TYPES:
        thread_id = channel_id
        parent_channel_id = str(channel.get("parent_id") or "") or None
        raw_name = channel.get("name")
        thread_name = raw_name.strip() if isinstance(raw_name, str) else None
        if parent_channel_id:
            try:
                parent_channel = _fetch_channel(
                    parent_channel_id,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                )
                parent_channel_name = _channel_label(parent_channel, parent_channel_id)
            except Exception:
                parent_channel_name = None
        _log(f"  Message belongs to thread {thread_id}")

    forward_source_lookup = _build_forward_source_lookup(
        filtered,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    message_lookup = _build_message_lookup(filtered)

    normalized_messages = [
        _normalize_message(
            message,
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=thread_id,
            include_media_descriptions=settings.include_media_descriptions,
            include_embed_media_descriptions=settings.include_embed_media_descriptions,
            message_lookup=message_lookup,
            forward_source_lookup=forward_source_lookup,
        )
        for message in filtered
    ]
    cleaned_messages = [
        message for message in normalized_messages if isinstance(message, dict)
    ]

    label = _document_label(
        guild_id=guild_id,
        channel_id=channel_id,
        parent_channel_id=parent_channel_id,
        thread_id=thread_id,
        message_id=message_id,
    )
    trace_path = _document_trace_path(
        guild_id=guild_id,
        channel_id=channel_id,
        parent_channel_id=parent_channel_id,
        thread_id=thread_id,
    )

    base = DiscordDocument(
        source_url=url,
        label=label,
        trace_path=trace_path,
        guild_id=guild_id,
        channel_id=channel_id,
        channel_name=channel_name,
        channel_type=channel_type,
        thread_id=thread_id,
        thread_name=thread_name,
        parent_channel_id=parent_channel_id,
        kind="thread" if thread_id else "message",
        messages=cleaned_messages,
        rendered="",
        guild_name=guild_name,
        parent_channel_name=parent_channel_name,
    )
    _log(
        f"  Built message document '{base.label}' with {len(cleaned_messages)} message(s)"
    )
    rendered = _render_document(base, settings)
    return [
        DiscordDocument(
            source_url=base.source_url,
            label=base.label,
            trace_path=base.trace_path,
            guild_id=base.guild_id,
            channel_id=base.channel_id,
            channel_name=base.channel_name,
            channel_type=base.channel_type,
            thread_id=base.thread_id,
            thread_name=base.thread_name,
            parent_channel_id=base.parent_channel_id,
            kind=base.kind,
            messages=base.messages,
            rendered=rendered,
            guild_name=base.guild_name,
            parent_channel_name=base.parent_channel_name,
        )
    ]


def _resolve_channel_messages(
    channel_id: str,
    *,
    settings: DiscordSettings,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> list[dict[str, Any]]:
    if _time_window_enabled(settings):
        target_cap = max(200, settings.channel_limit)
        _log(
            f"    Fetching channel {channel_id} messages via time window (cap={target_cap})"
        )
        start, end = _window_bounds_no_target(settings)
        messages = _fetch_messages_between(
            channel_id,
            start,
            end,
            max_messages=target_cap,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        _log(f"    Retrieved {len(messages)} message(s) for channel {channel_id}")
        return messages
    _log(
        f"    Fetching latest channel messages for {channel_id} (limit={settings.channel_limit})"
    )
    messages = _sort_messages(
        _fetch_latest_messages(
            channel_id,
            settings.channel_limit,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
    )
    _log(f"    Retrieved {len(messages)} message(s) for channel {channel_id}")
    return messages


def _thread_ids_from_messages(
    messages: list[dict[str, Any]],
) -> list[tuple[str, str | None]]:
    seen: set[str] = set()
    out: list[tuple[str, str | None]] = []
    for message in messages:
        thread_data = (
            message.get("thread") if isinstance(message.get("thread"), dict) else None
        )
        if not thread_data:
            continue
        raw_thread_id = thread_data.get("id")
        if raw_thread_id is None:
            continue
        thread_id = str(raw_thread_id)
        if thread_id in seen:
            continue
        seen.add(thread_id)
        thread_name = thread_data.get("name")
        out.append((thread_id, thread_name if isinstance(thread_name, str) else None))
    return out


def _build_document(
    *,
    source_url: str,
    guild_id: str,
    channel: dict[str, Any],
    messages: list[dict[str, Any]],
    kind: str,
    thread_id: str | None,
    thread_name: str | None,
    parent_channel_id: str | None,
    guild_name: str | None,
    parent_channel_name: str | None,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
    settings: DiscordSettings,
) -> DiscordDocument:
    channel_id = str(channel.get("id") or "")
    channel_type_raw = channel.get("type")
    channel_type = int(channel_type_raw) if isinstance(channel_type_raw, int) else None

    forward_source_lookup = _build_forward_source_lookup(
        messages,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    message_lookup = _build_message_lookup(messages)

    normalized_messages = [
        _normalize_message(
            message,
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=thread_id,
            include_media_descriptions=settings.include_media_descriptions,
            include_embed_media_descriptions=settings.include_embed_media_descriptions,
            message_lookup=message_lookup,
            forward_source_lookup=forward_source_lookup,
        )
        for message in messages
    ]
    cleaned_messages = [item for item in normalized_messages if isinstance(item, dict)]

    label = _document_label(
        guild_id=guild_id,
        channel_id=channel_id,
        parent_channel_id=parent_channel_id,
        thread_id=thread_id,
        message_id=None,
    )
    trace_path = _document_trace_path(
        guild_id=guild_id,
        channel_id=channel_id,
        parent_channel_id=parent_channel_id,
        thread_id=thread_id,
    )

    base = DiscordDocument(
        source_url=source_url,
        label=label,
        trace_path=trace_path,
        guild_id=guild_id,
        channel_id=channel_id,
        channel_name=_channel_label(channel, channel_id),
        channel_type=channel_type,
        thread_id=thread_id,
        thread_name=thread_name,
        parent_channel_id=parent_channel_id,
        kind=kind,
        messages=cleaned_messages,
        rendered="",
        guild_name=guild_name,
        parent_channel_name=parent_channel_name,
    )
    rendered = _render_document(base, settings)
    _log(
        f"  Rendered {kind} document '{base.label}' with {len(cleaned_messages)} message(s)"
    )
    return DiscordDocument(
        source_url=base.source_url,
        label=base.label,
        trace_path=base.trace_path,
        guild_id=base.guild_id,
        channel_id=base.channel_id,
        channel_name=base.channel_name,
        channel_type=base.channel_type,
        thread_id=base.thread_id,
        thread_name=base.thread_name,
        parent_channel_id=base.parent_channel_id,
        kind=base.kind,
        messages=base.messages,
        rendered=rendered,
        guild_name=base.guild_name,
        parent_channel_name=base.parent_channel_name,
    )


def _resolve_channel_or_thread_url(
    url: str,
    parsed: dict[str, str],
    *,
    settings: DiscordSettings,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> list[DiscordDocument]:
    guild_id = parsed["guild_id"]
    channel_id = parsed["channel_id"]

    _log(f"Resolving Discord channel URL (guild={guild_id}, channel={channel_id})")
    _validate_guild_scope(guild_id)
    guild_name: str | None = None
    try:
        guild = _fetch_guild(
            guild_id,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        guild_name = _guild_label(guild, guild_id)
    except Exception:
        guild_name = None
    if guild_name:
        _log(f"  Resolved guild: {guild_name} ({guild_id})")
    else:
        _log(f"  Guild metadata unavailable for {guild_id}")

    channel = _fetch_channel(
        channel_id,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    channel_name = _channel_label(channel, channel_id)
    _log(f"  Resolved channel: #{channel_name} ({channel_id})")
    channel_type_raw = channel.get("type")
    channel_type = int(channel_type_raw) if isinstance(channel_type_raw, int) else None

    messages = _resolve_channel_messages(
        channel_id,
        settings=settings,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    fetched_count = len(messages)
    thread_targets = (
        _thread_ids_from_messages(messages) if settings.expand_threads else []
    )
    messages = _filter_messages(messages, settings=settings)
    _log(
        f"  Channel summary: fetched={fetched_count}, kept={len(messages)}, discovered_threads={len(thread_targets)}"
    )

    documents: list[DiscordDocument] = []

    if channel_type in _DISCORD_THREAD_TYPES:
        thread_name = (
            channel.get("name") if isinstance(channel.get("name"), str) else None
        )
        parent_channel_id = str(channel.get("parent_id") or "") or None
        parent_channel_name: str | None = None
        if parent_channel_id:
            try:
                parent_channel = _fetch_channel(
                    parent_channel_id,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                )
                parent_channel_name = _channel_label(parent_channel, parent_channel_id)
            except Exception:
                parent_channel_name = None
        documents.append(
            _build_document(
                source_url=url,
                guild_id=guild_id,
                channel=channel,
                messages=messages,
                kind="thread",
                thread_id=channel_id,
                thread_name=thread_name,
                parent_channel_id=parent_channel_id,
                guild_name=guild_name,
                parent_channel_name=parent_channel_name,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
                settings=settings,
            )
        )
        _log("  URL is a thread channel; returning one thread document")
        return documents

    documents.append(
        _build_document(
            source_url=url,
            guild_id=guild_id,
            channel=channel,
            messages=messages,
            kind="channel",
            thread_id=None,
            thread_name=None,
            parent_channel_id=None,
            guild_name=guild_name,
            parent_channel_name=None,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            settings=settings,
        )
    )

    if not settings.expand_threads:
        _log("  Thread expansion disabled; returning one channel document")
        return documents

    _log(f"  Expanding {len(thread_targets)} discovered thread(s)")
    for thread_id, thread_name in thread_targets:
        thread_label = (
            thread_name.strip() if thread_name and thread_name.strip() else None
        )
        _log(f"    Resolving thread {thread_label or thread_id} ({thread_id})")
        try:
            thread_channel = _fetch_channel(
                thread_id,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
            )
        except Exception:
            _log(f"    Skipping thread {thread_id}: unable to fetch channel metadata")
            continue

        thread_messages = _resolve_channel_messages(
            thread_id,
            settings=settings,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        thread_messages = _filter_messages(thread_messages, settings=settings)
        _log(f"    Thread {thread_id} kept {len(thread_messages)} message(s)")

        thread_doc = _build_document(
            source_url=url,
            guild_id=guild_id,
            channel=thread_channel,
            messages=thread_messages,
            kind="thread",
            thread_id=thread_id,
            thread_name=thread_name,
            parent_channel_id=channel_id,
            guild_name=guild_name,
            parent_channel_name=_channel_label(channel, channel_id),
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            settings=settings,
        )
        documents.append(thread_doc)

    _log(f"  Built {len(documents)} document(s) from channel URL")
    return documents


def resolve_discord_url(
    url: str,
    *,
    settings: DiscordSettings | None = None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> list[DiscordDocument]:
    warmup_discord_network_stack()

    _log(f"Resolving Discord URL: {url}")
    cache_mode = "refresh" if refresh_cache else "reuse"
    _log(f"  Cache mode: enabled={use_cache}, mode={cache_mode}")
    parsed = parse_discord_url(url)
    if not parsed:
        raise ValueError(f"Not a Discord URL: {url}")
    _log(f"  Parsed URL kind: {parsed['kind']}")

    effective_settings = (
        settings if settings is not None else _discord_settings_from_env()
    )

    if parsed["kind"] == "message":
        documents = _resolve_message_url(
            url,
            parsed,
            settings=effective_settings,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
    else:
        documents = _resolve_channel_or_thread_url(
            url,
            parsed,
            settings=effective_settings,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
    _log(f"Resolved Discord URL to {len(documents)} document(s)")
    return documents


def _parse_window_mapping(raw: Any, *, prefix: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix}.window must be a mapping")

    allowed_keys = {
        "before-messages",
        "after-messages",
        "around-messages",
        "message-context",
        "start",
        "end",
        "before-duration",
        "after-duration",
        "around-duration",
        "channel-limit",
    }
    unknown = sorted(str(key) for key in raw.keys() if key not in allowed_keys)
    if unknown:
        raise ValueError(f"{prefix}.window has invalid keys: {', '.join(unknown)}")

    result: dict[str, Any] = {}

    for config_key, result_key in (
        ("before-messages", "before_messages"),
        ("after-messages", "after_messages"),
        ("around-messages", "around_messages"),
        ("message-context", "message_context"),
        ("channel-limit", "channel_limit"),
    ):
        if config_key not in raw:
            continue
        value = raw.get(config_key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"{prefix}.window.{config_key} must be a non-negative integer"
            )
        result[result_key] = value

    for config_key, result_key in (("start", "start"), ("end", "end")):
        if config_key not in raw:
            continue
        value = raw.get(config_key)
        if not isinstance(value, str):
            raise ValueError(
                f"{prefix}.window.{config_key} must be an ISO timestamp string"
            )
        parsed = _parse_iso_datetime(value)
        if parsed is None:
            raise ValueError(f"{prefix}.window.{config_key} is not a valid timestamp")
        result[result_key] = parsed

    for config_key, result_key in (
        ("before-duration", "before_duration"),
        ("after-duration", "after_duration"),
        ("around-duration", "around_duration"),
    ):
        if config_key not in raw:
            continue
        value = raw.get(config_key)
        if not isinstance(value, str):
            raise ValueError(f"{prefix}.window.{config_key} must be a duration string")
        parsed = _parse_duration_compound(value)
        if parsed is None:
            raise ValueError(
                f"{prefix}.window.{config_key} must use compound units like 2m5d, 6h, 45s"
            )
        result[result_key] = parsed

    return result or None


def parse_discord_config_mapping(raw: Any, *, prefix: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix} must be a mapping")

    allowed_keys = {
        "format",
        "include-system",
        "include-thread-starters",
        "expand-threads",
        "gap-threshold",
        "window",
        "media",
    }
    unknown = sorted(str(key) for key in raw.keys() if key not in allowed_keys)
    if unknown:
        raise ValueError(f"{prefix} has invalid keys: {', '.join(unknown)}")

    result: dict[str, Any] = {}

    if "format" in raw:
        value = raw.get("format")
        if not isinstance(value, str) or value.strip().lower() not in _VALID_FORMATS:
            raise ValueError(
                f"{prefix}.format must be one of: {', '.join(sorted(_VALID_FORMATS))}"
            )
        result["format"] = value.strip().lower()

    for config_key, result_key in (
        ("include-system", "include_system"),
        ("include-thread-starters", "include_thread_starters"),
        ("expand-threads", "expand_threads"),
    ):
        if config_key not in raw:
            continue
        value = raw.get(config_key)
        if not isinstance(value, bool):
            raise ValueError(f"{prefix}.{config_key} must be a boolean")
        result[result_key] = value

    if "gap-threshold" in raw:
        value = raw.get("gap-threshold")
        if not isinstance(value, str):
            raise ValueError(f"{prefix}.gap-threshold must be a duration string")
        parsed = _parse_duration_compound(value)
        if parsed is None:
            raise ValueError(f"{prefix}.gap-threshold must use duration tokens like 6h")
        result["gap_threshold"] = parsed

    window_overrides = _parse_window_mapping(raw.get("window"), prefix=prefix)
    if window_overrides:
        result.update(window_overrides)

    media = raw.get("media")
    if media is not None:
        if not isinstance(media, dict):
            raise ValueError(f"{prefix}.media must be a mapping")
        allowed_media = {"describe", "embed-media-describe"}
        unknown_media = sorted(
            str(key) for key in media.keys() if key not in allowed_media
        )
        if unknown_media:
            raise ValueError(
                f"{prefix}.media has invalid keys: {', '.join(unknown_media)}"
            )

        if "describe" in media:
            describe = media.get("describe")
            if not isinstance(describe, bool):
                raise ValueError(f"{prefix}.media.describe must be a boolean")
            result["include_media_descriptions"] = describe

        if "embed-media-describe" in media:
            embed_describe = media.get("embed-media-describe")
            if not isinstance(embed_describe, bool):
                raise ValueError(
                    f"{prefix}.media.embed-media-describe must be a boolean"
                )
            result["include_embed_media_descriptions"] = embed_describe

    return result or None


def merge_discord_overrides(*overrides: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for item in overrides:
        if item:
            merged.update(item)
    return merged or None


def discord_overrides_cache_key(
    overrides: dict[str, Any] | None,
) -> tuple[Any, ...] | None:
    if not overrides:
        return None

    normalized: list[tuple[str, Any]] = []
    for key, value in sorted(overrides.items()):
        if isinstance(value, datetime):
            normalized.append((key, value.astimezone(timezone.utc).isoformat()))
        elif isinstance(value, timedelta):
            normalized.append((key, int(value.total_seconds())))
        else:
            normalized.append((key, value))
    return tuple(normalized)


def discord_settings_cache_key(settings: DiscordSettings) -> tuple[Any, ...]:
    return (
        settings.format,
        settings.include_system,
        settings.include_thread_starters,
        settings.expand_threads,
        int(settings.gap_threshold.total_seconds()),
        settings.before_messages,
        settings.after_messages,
        settings.around_messages,
        settings.message_context,
        settings.channel_limit,
        settings.start.isoformat() if settings.start else None,
        settings.end.isoformat() if settings.end else None,
        int(settings.before_duration.total_seconds())
        if settings.before_duration
        else None,
        int(settings.after_duration.total_seconds())
        if settings.after_duration
        else None,
        int(settings.around_duration.total_seconds())
        if settings.around_duration
        else None,
        settings.include_media_descriptions,
        settings.include_embed_media_descriptions,
        settings.hard_min_timestamp.isoformat()
        if settings.hard_min_timestamp
        else None,
        settings.hard_max_timestamp.isoformat()
        if settings.hard_max_timestamp
        else None,
    )
