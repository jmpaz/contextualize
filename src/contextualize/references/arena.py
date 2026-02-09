from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from ..render.text import process_text
from ..utils import count_tokens

_ARENA_CHANNEL_RE = re.compile(
    r"^https?://(?:www\.)?are\.na/"
    r"(?:channel/(?P<slug1>[^/?#]+)"
    r"|(?P<user>[^/?#]+)/(?P<slug2>[^/?#]+))$"
)
_ARENA_BLOCK_RE = re.compile(r"^https?://(?:www\.)?are\.na/block/(?P<id>\d+)$")

_API_BASE = "https://api.are.na/v3"


def _log(msg: str) -> None:
    _load_dotenv()
    if os.environ.get("ARENA_VERBOSE", "1").lower() not in ("0", "false", "no"):
        print(msg, file=sys.stderr, flush=True)


_RESERVED_PATHS = frozenset(
    {
        "about",
        "explore",
        "search",
        "settings",
        "notifications",
        "feed",
        "blog",
        "pricing",
        "terms",
        "privacy",
        "sign_in",
        "sign_up",
        "log_in",
        "register",
        "block",
        "channel",
        "api",
    }
)


def is_arena_url(url: str) -> bool:
    return is_arena_channel_url(url) or is_arena_block_url(url)


def is_arena_channel_url(url: str) -> bool:
    match = _ARENA_CHANNEL_RE.match(url)
    if not match:
        return False
    user = match.group("user")
    if user and user.lower() in _RESERVED_PATHS:
        return False
    return True


def is_arena_block_url(url: str) -> bool:
    return bool(_ARENA_BLOCK_RE.match(url))


def extract_channel_slug(url: str) -> str | None:
    match = _ARENA_CHANNEL_RE.match(url)
    if not match:
        return None
    user = match.group("user")
    if user and user.lower() in _RESERVED_PATHS:
        return None
    return match.group("slug1") or match.group("slug2")


def extract_block_id(url: str) -> int | None:
    match = _ARENA_BLOCK_RE.match(url)
    if not match:
        return None
    return int(match.group("id"))


def _load_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except Exception:
        pass


def _get_auth_headers() -> dict[str, str]:
    _load_dotenv()
    token = os.environ.get("ARENA_ACCESS_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _api_timeout_seconds() -> float:
    raw = (os.environ.get("ARENA_API_TIMEOUT") or "").strip()
    if not raw:
        return 30.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def _api_max_attempts() -> int:
    raw = (os.environ.get("ARENA_API_MAX_ATTEMPTS") or "").strip()
    if not raw:
        return 6
    try:
        return max(1, int(raw))
    except ValueError:
        return 6


def _retry_delay_seconds(attempt: int) -> float:
    import random

    base = min(30.0, 1.0 * (2 ** max(0, attempt - 1)))
    return base + random.uniform(0.0, 0.25)


def _server_error_retry_delay_seconds(attempt: int) -> float:
    import random

    base = min(30.0, 5.0 * (2 ** max(0, attempt - 1)))
    return base + random.uniform(0.0, 0.25)


def _retry_after_seconds(resp: object) -> float | None:
    import time

    headers = getattr(resp, "headers", None) or {}
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    reset = headers.get("X-RateLimit-Reset")
    if reset:
        try:
            value = float(reset)
            if value > 10_000_000:
                return max(0.0, value - time.time())
            return max(0.0, value)
        except ValueError:
            pass
    return None


def _api_get(path: str, params: dict | None = None) -> dict:
    import requests
    import time

    url = f"{_API_BASE}{path}"
    headers = {**_get_auth_headers(), "Accept": "application/json"}
    timeout = _api_timeout_seconds()
    max_attempts = _api_max_attempts()
    transient_statuses = {429, 500, 502, 503, 504}

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise
            wait = _retry_delay_seconds(attempt)
            _log(
                f"  Are.na request failed ({type(exc).__name__}); retrying in {wait:.1f}s "
                f"(attempt {attempt}/{max_attempts})"
            )
            time.sleep(wait)
            continue

        if resp.status_code == 404:
            raise ValueError(f"Are.na resource not found: {path}")

        if resp.status_code in transient_statuses and attempt < max_attempts:
            if resp.status_code == 429:
                wait = _retry_after_seconds(resp)
                if wait is None:
                    wait = _retry_delay_seconds(attempt)
            else:
                wait = _server_error_retry_delay_seconds(attempt)
            _log(
                f"  Are.na API returned {resp.status_code}; retrying in {wait:.1f}s "
                f"(attempt {attempt}/{max_attempts})"
            )
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Are.na request failed unexpectedly for {path}")


def _fetch_channel(slug: str) -> dict:
    return _api_get(f"/channels/{slug}")


def _fetch_channel_page(slug: str, page: int, per: int = 100) -> dict:
    return _api_get(f"/channels/{slug}/contents", {"page": page, "per": per})


def _get_max_depth() -> int:
    raw = os.environ.get("ARENA_MAX_DEPTH", "1")
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


def _get_include_descriptions() -> bool:
    raw = os.environ.get("ARENA_INCLUDE_DESCRIPTIONS", "1").lower()
    return raw not in ("0", "false", "no")


def _get_sort_order() -> str:
    return os.environ.get("ARENA_SORT", "desc").lower().strip()


def _get_recurse_users() -> set[str] | None:
    raw = os.environ.get("ARENA_RECURSE_USERS", "").strip()
    if not raw:
        return {"self"}
    if raw.lower() == "all":
        return None
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


VALID_SORT_ORDERS = frozenset(
    {
        "asc",
        "desc",
        "date-asc",
        "date-desc",
        "random",
        "position-asc",
        "position-desc",
    }
)


@dataclass(frozen=True)
class ArenaSettings:
    max_depth: int = 1
    sort_order: str = "desc"
    include_descriptions: bool = True
    recurse_users: set[str] | None = field(default_factory=lambda: {"self"})


def _arena_settings_from_env() -> ArenaSettings:
    return ArenaSettings(
        max_depth=_get_max_depth(),
        sort_order=_get_sort_order(),
        include_descriptions=_get_include_descriptions(),
        recurse_users=_get_recurse_users(),
    )


def build_arena_settings(overrides: dict | None = None) -> ArenaSettings:
    env = _arena_settings_from_env()
    if not overrides:
        return env

    max_depth = overrides.get("max_depth", env.max_depth)
    sort_order = overrides.get("sort_order", env.sort_order)
    include_descriptions = overrides.get(
        "include_descriptions", env.include_descriptions
    )
    recurse_users = overrides.get("recurse_users", env.recurse_users)

    return ArenaSettings(
        max_depth=max_depth,
        sort_order=sort_order,
        include_descriptions=include_descriptions,
        recurse_users=recurse_users,
    )


def _owner_slug(obj: dict) -> str:
    owner = obj.get("owner") or obj.get("user") or {}
    return (owner.get("slug") or "").lower()


def _owner_id(obj: dict) -> int | None:
    owner = obj.get("owner") or obj.get("user") or {}
    return owner.get("id")


def _should_recurse(
    item: dict,
    recurse_users: set[str] | None,
    root_owner_id: int | None,
) -> bool:
    if recurse_users is None:
        return True
    if recurse_users & {"self", "author", "owner"}:
        return _owner_id(item) == root_owner_id
    return _owner_slug(item) in recurse_users


def _fetch_all_channel_contents(
    slug: str,
    *,
    max_depth: int | None = None,
    _depth: int = 0,
    _visited: set[int] | None = None,
    _root_owner_id: int | None = None,
    _recurse_users: set[str] | None = ...,
) -> tuple[dict, list[dict]]:
    if max_depth is None:
        max_depth = _get_max_depth()
    if _recurse_users is ...:
        _recurse_users = _get_recurse_users()
    if _visited is None:
        _visited = set()

    metadata = _fetch_channel(slug)
    channel_id = metadata.get("id")
    channel_title = metadata.get("title") or slug
    if channel_id:
        _visited.add(channel_id)

    if _root_owner_id is None:
        _root_owner_id = _owner_id(metadata)

    indent = "  " * (_depth + 1)
    total_count = (metadata.get("counts") or {}).get("contents") or "?"
    _log(f"{indent}fetching channel: {channel_title} ({total_count} items)")

    all_contents: list[dict] = []
    first_page = _fetch_channel_page(slug, 1)
    all_contents.extend(first_page.get("data", first_page.get("contents", [])))

    meta = first_page.get("meta", {})
    total_pages = meta.get("total_pages", 1)
    for page in range(2, total_pages + 1):
        _log(f"{indent}  page {page}/{total_pages}")
        page_data = _fetch_channel_page(slug, page)
        all_contents.extend(page_data.get("data", page_data.get("contents", [])))

    if _depth < max_depth:
        expanded: list[dict] = []
        for item in all_contents:
            if item.get("base_type") == "Channel" or item.get("type") == "Channel":
                nested_id = item.get("id")
                nested_slug = item.get("slug")
                if nested_id and nested_id in _visited:
                    expanded.append(item)
                    continue
                if nested_slug and _should_recurse(
                    item, _recurse_users, _root_owner_id
                ):
                    _visited.add(nested_id)
                    nested_meta, nested_contents = _fetch_all_channel_contents(
                        nested_slug,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                        _visited=_visited,
                        _root_owner_id=_root_owner_id,
                        _recurse_users=_recurse_users,
                    )
                    item["_nested_metadata"] = nested_meta
                    item["_nested_contents"] = nested_contents
            expanded.append(item)
        all_contents = expanded

    return metadata, all_contents


def _fetch_block(block_id: int) -> dict:
    return _api_get(f"/blocks/{block_id}")


_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; contextualize/1.0)",
    "Accept": "image/*,application/*;q=0.9,*/*;q=0.8",
    "Referer": "https://www.are.na/",
}


def _download_to_temp(
    url: str, suffix: str = "", *, media_cache_identity: str | None = None
) -> Path | None:
    import requests
    from ..cache.arena import get_cached_media_bytes, store_media_bytes
    from ..runtime import get_refresh_cache

    cache_identity = media_cache_identity or url
    cached = None if get_refresh_cache() else get_cached_media_bytes(cache_identity)
    if cached:
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            os.write(fd, cached)
        finally:
            os.close(fd)
        return Path(path)

    try:
        resp = requests.get(url, headers=_DOWNLOAD_HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None
    if not resp.content:
        return None
    store_media_bytes(cache_identity, resp.content)
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, resp.content)
    finally:
        os.close(fd)
    return Path(path)


def _render_block_binary(
    url: str,
    suffix: str,
    *,
    media_cache_identity: str | None = None,
    send_label: str | None = None,
) -> str:
    from ..render.markitdown import MarkItDownConversionError, convert_path_to_markdown

    tmp = _download_to_temp(
        url, suffix=suffix, media_cache_identity=media_cache_identity
    )
    if tmp is None:
        return ""
    try:
        from ..runtime import get_refresh_images

        label = send_label or "arena-media"
        _log(f"  sending to model: {label} ({url})")
        refresh_images = get_refresh_images()
        result = convert_path_to_markdown(str(tmp), refresh_images=refresh_images)
        return result.markdown
    except MarkItDownConversionError as exc:
        _log(f"  image conversion failed for {url}: {exc}")
        return ""
    finally:
        tmp.unlink(missing_ok=True)


def _desc_separator(description: str) -> str:
    max_stars = 0
    for line in description.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"\*{3,}", stripped):
            max_stars = max(max_stars, len(stripped))
    return "*" * (max_stars + 2) if max_stars >= 3 else "***"


def _format_date_line(block: dict) -> str:
    connected = block.get("connected_at") or ""
    created = block.get("created_at") or ""
    c_date = connected[:10] if len(connected) >= 10 else ""
    r_date = created[:10] if len(created) >= 10 else ""
    if c_date and r_date and c_date != r_date:
        return f"connected {c_date} (created {r_date})"
    return c_date or r_date


def _attachment_media_kind(
    *, filename: str, extension: str, content_type: str
) -> str | None:
    ctype = content_type.lower().strip()
    if ctype.startswith("image/"):
        return "image"
    if ctype.startswith("video/"):
        return "video"
    if ctype.startswith("audio/"):
        return "audio"

    suffix = (
        f".{extension.lstrip('.').lower()}"
        if extension
        else Path(filename).suffix.lower()
    )
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif", ".avif"}:
        return "image"
    if suffix in {".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg", ".m4v"}:
        return "video"
    if suffix in {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".aiff"}:
        return "audio"
    return None


def _should_refresh_attachment_media(
    *, filename: str, extension: str, content_type: str
) -> bool:
    from ..runtime import (
        get_refresh_audio,
        get_refresh_images,
        get_refresh_media,
        get_refresh_videos,
    )

    if get_refresh_media():
        return True

    media_kind = _attachment_media_kind(
        filename=filename, extension=extension, content_type=content_type
    )
    if media_kind == "image":
        return get_refresh_images()
    if media_kind == "video":
        return get_refresh_videos()
    if media_kind == "audio":
        return get_refresh_audio()
    return False


def _format_block_output(
    title: str, description: str, content: str, date: str = ""
) -> str | None:
    if not title and not description and not content and not date:
        return None

    parts: list[str] = []
    if title and date:
        parts.append(f"{title}\n{date}\n---")
    elif date:
        parts.append(f"{date}\n---")
    elif title:
        parts.append(f"{title}\n---")

    if description and content:
        sep = _desc_separator(description)
        parts.append(f"{description}\n\n{sep}")
        parts.append(content)
    elif description:
        parts.append(description)
    elif content:
        parts.append(content)

    return "\n\n".join(parts)


def _render_block(
    block: dict, *, include_descriptions: bool | None = None
) -> str | None:
    from ..cache.arena import get_cached_block_render, store_block_render

    block_type = block.get("class") or block.get("type", "")
    state = block.get("state")
    if state == "processing" or block_type == "PendingBlock":
        return None

    block_id = block.get("id")
    updated_at = block.get("updated_at") or ""
    from ..runtime import get_refresh_images, get_refresh_media, get_refresh_videos

    title = block.get("title") or ""
    if include_descriptions is None:
        include_descriptions = _get_include_descriptions()
    if include_descriptions:
        raw_desc = block.get("description") or ""
        if isinstance(raw_desc, dict):
            description = raw_desc.get("markdown") or raw_desc.get("plain") or ""
        else:
            description = raw_desc
    else:
        description = ""

    date = _format_date_line(block)

    if block_type == "Text":
        raw_content = block.get("content") or ""
        if isinstance(raw_content, dict):
            content = raw_content.get("markdown") or raw_content.get("plain") or ""
        else:
            content = raw_content
        if description == content:
            description = ""
        return _format_block_output(title, description, content, date=date)

    if block_type == "Image":
        refresh_image = get_refresh_images() or get_refresh_media()
        if block_id and updated_at and not refresh_image:
            cached = get_cached_block_render(block_id, updated_at)
            if cached is not None:
                return cached
        image = block.get("image") or {}
        image_urls = list(
            dict.fromkeys(
                [
                    u
                    for u in (
                        image.get("src"),
                        (image.get("large") or {}).get("src"),
                        (image.get("original") or {}).get("url"),
                        image.get("url"),
                    )
                    if u
                ]
            )
        )
        for image_url in image_urls:
            suffix = Path(image_url.split("?")[0]).suffix or ".jpg"
            media_cache_identity = (
                f"arena:block:{block_id}:{updated_at}:image:{image_url}"
                if block_id and updated_at
                else image_url
            )
            send_label = f"image:{block_id or 'unknown'}:{(title or 'untitled')[:80]}"
            converted = _render_block_binary(
                image_url,
                suffix,
                media_cache_identity=media_cache_identity,
                send_label=send_label,
            )
            if converted:
                result = _format_block_output(title, description, converted, date=date)
                if result and block_id and updated_at:
                    store_block_render(block_id, updated_at, result)
                return result
        fallback_url = image_urls[0] if image_urls else ""
        fallback = f"[Image: {title or block.get('id')}]"
        if fallback_url:
            fallback += f"\nURL: {fallback_url}"
        if block_id and updated_at:
            store_block_render(block_id, updated_at, fallback)
        return fallback

    if block_type == "Link":
        source = block.get("source") or {}
        source_url = source.get("url") or ""
        raw_content = block.get("content") or ""
        if isinstance(raw_content, dict):
            content = raw_content.get("markdown") or raw_content.get("plain") or ""
        else:
            content = raw_content
        link_parts = []
        if source_url:
            link_parts.append(source_url)
        if content:
            link_parts.append(content)
        return _format_block_output(
            title, description, "\n\n".join(link_parts), date=date
        )

    if block_type == "Attachment":
        attachment = block.get("attachment") or {}
        att_url = attachment.get("url") or ""
        filename = attachment.get("filename") or ""
        content_type = attachment.get("content_type") or ""
        extension = attachment.get("file_extension") or ""
        refresh_attachment = _should_refresh_attachment_media(
            filename=filename,
            extension=extension,
            content_type=content_type,
        )
        if block_id and updated_at and not refresh_attachment:
            cached = get_cached_block_render(block_id, updated_at)
            if cached is not None:
                return cached
        if att_url:
            suffix = f".{extension}" if extension else Path(filename).suffix or ""
            media_cache_identity = (
                f"arena:block:{block_id}:{updated_at}:attachment:{att_url}"
                if block_id and updated_at
                else att_url
            )
            send_label = f"attachment:{block_id or 'unknown'}:{(filename or title or 'untitled')[:80]}"
            converted = _render_block_binary(
                att_url,
                suffix,
                media_cache_identity=media_cache_identity,
                send_label=send_label,
            )
            if converted:
                att_title = title if title != filename else ""
                result = _format_block_output(
                    att_title, description, converted, date=date
                )
                if result and block_id and updated_at:
                    store_block_render(block_id, updated_at, result)
                return result
        fallback = f"[Attachment: {filename or title or block.get('id')}]"
        if content_type:
            fallback += f"\nType: {content_type}"
        if att_url:
            fallback += f"\nURL: {att_url}"
        if block_id and updated_at:
            store_block_render(block_id, updated_at, fallback)
        return fallback

    if block_type == "Embed":
        refresh_embed = (
            get_refresh_images() or get_refresh_media() or get_refresh_videos()
        )
        if block_id and updated_at and not refresh_embed:
            cached = get_cached_block_render(block_id, updated_at)
            if cached is not None:
                return cached

        embed = block.get("embed") or {}
        embed_url = embed.get("url") or ""
        embed_type = embed.get("type") or ""
        embed_parts = []
        if embed_type == "video":
            image = block.get("image") or {}
            image_urls = list(
                dict.fromkeys(
                    [
                        u
                        for u in (
                            image.get("src"),
                            (image.get("large") or {}).get("src"),
                            (image.get("original") or {}).get("url"),
                            image.get("url"),
                        )
                        if u
                    ]
                )
            )
            for image_url in image_urls:
                suffix = Path(image_url.split("?")[0]).suffix or ".jpg"
                media_cache_identity = (
                    f"arena:block:{block_id}:{updated_at}:embed-image:{image_url}"
                    if block_id and updated_at
                    else image_url
                )
                send_label = (
                    f"embed-image:{block_id or 'unknown'}:{(title or 'untitled')[:80]}"
                )
                converted = _render_block_binary(
                    image_url,
                    suffix,
                    media_cache_identity=media_cache_identity,
                    send_label=send_label,
                )
                if converted:
                    embed_parts.append(converted)
                    break
        if embed_url:
            embed_parts.append(embed_url)
        if embed_type:
            embed_parts.append(f"Type: {embed_type}")
        result = _format_block_output(
            title, description, "\n\n".join(embed_parts), date=date
        )
        if result and block_id and updated_at:
            store_block_render(block_id, updated_at, result)
        return result

    return _format_block_output(title, description, "", date=date)


def _render_channel_stub(item: dict) -> str:
    ch_title = item.get("title") or item.get("slug") or "Untitled"
    ch_owner = (item.get("owner") or {}).get("name") or ""
    ch_slug = item.get("slug") or ""
    ch_counts = item.get("counts") or {}
    ch_blocks = ch_counts.get("contents") or "?"
    parts = [f"[Channel: {ch_title}]"]
    if ch_owner:
        parts.append(f"Owner: {ch_owner}")
    parts.append(f"Blocks: {ch_blocks}")
    if ch_slug:
        parts.append(f"https://www.are.na/channel/{ch_slug}")
    return "\n".join(parts)


def _block_label(block: dict, channel_slug: str, channel_path: str = "") -> str:
    block_id = block.get("id", "unknown")
    prefix = channel_path or channel_slug or "are.na/block"
    return f"{prefix}/{block_id}"


def _flatten_channel_blocks(
    contents: list[dict],
    channel_slug: str,
    channel_path: str = "",
) -> list[tuple[str, dict]]:
    result: list[tuple[str, dict]] = []
    path = channel_path or channel_slug

    for item in contents:
        if item.get("base_type") == "Channel" or item.get("type") == "Channel":
            nested_contents = item.get("_nested_contents")
            if nested_contents is not None:
                nested_meta = item.get("_nested_metadata", item)
                nested_slug = nested_meta.get("slug") or item.get("slug") or ""
                nested_title = (
                    nested_meta.get("title") or item.get("title") or "channel"
                )
                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", nested_title)[:60].strip(
                    "-"
                )
                sub_path = f"{path}/{safe_name or nested_slug}"
                result.extend(
                    _flatten_channel_blocks(nested_contents, nested_slug, sub_path)
                )
            else:
                result.append((path, item))
        else:
            result.append((path, item))

    return result


def _block_identity(block: dict) -> str | None:
    block_id = block.get("id")
    if block_id is not None:
        return f"id:{block_id}"
    slug = block.get("slug")
    if slug:
        return f"slug:{slug}"
    return None


def _dedupe_flat_blocks(flat: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    deduped: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for path, block in flat:
        identity = _block_identity(block)
        if identity is not None:
            if identity in seen:
                continue
            seen.add(identity)
        deduped.append((path, block))
    return deduped


def _sort_blocks(flat: list[tuple[str, dict]], order: str) -> list[tuple[str, dict]]:
    if order == "position-asc" or order == "asc":
        return flat
    if order == "position-desc" or order == "desc":
        return list(reversed(flat))
    if order == "random":
        import random

        shuffled = list(flat)
        random.shuffle(shuffled)
        return shuffled
    use_reverse = order == "date-desc"
    return sorted(
        flat,
        key=lambda pair: pair[1].get("connected_at") or pair[1].get("created_at") or "",
        reverse=use_reverse,
    )


def resolve_channel(
    slug: str,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    settings: ArenaSettings | None = None,
) -> tuple[dict, list[tuple[str, dict]]]:
    if settings is None:
        settings = _arena_settings_from_env()
    max_depth = settings.max_depth
    recurse_users = settings.recurse_users
    sort_order = settings.sort_order
    ru_key = ",".join(sorted(recurse_users)) if recurse_users else "all"
    cache_key = f"{slug}:d={max_depth}:u={ru_key}:s={sort_order}"

    if use_cache and not refresh_cache:
        from ..cache.arena import get_cached_channel

        cached = get_cached_channel(cache_key, cache_ttl)
        if cached is not None:
            data = json.loads(cached)
            metadata = data["metadata"]
            flat = [(path, block) for path, block in data["blocks"]]
            channel_title = metadata.get("title") or slug
            _log(f"  using cached channel: {channel_title} ({len(flat)} items)")
            return metadata, flat

    metadata, contents = _fetch_all_channel_contents(
        slug,
        max_depth=max_depth,
        _recurse_users=recurse_users,
    )
    flat = _flatten_channel_blocks(contents, slug)
    flat = _sort_blocks(flat, sort_order)

    if use_cache:
        from ..cache.arena import store_channel

        data = json.dumps({"metadata": metadata, "blocks": flat}, ensure_ascii=False)
        block_count = len([b for _, b in flat if b.get("type") != "Channel"])
        store_channel(cache_key, data, block_count)

    return metadata, flat


@dataclass
class ArenaReference:
    url: str
    block: dict
    channel_path: str = ""
    format: str = "md"
    label: str = "relative"
    token_target: str = "cl100k_base"
    include_token_count: bool = False
    label_suffix: str | None = None
    inject: bool = False
    depth: int = 5
    trace_collector: list = None
    include_descriptions: bool | None = None

    def __post_init__(self) -> None:
        self.file_content = ""
        self.original_file_content = ""
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
        block_type = self.block.get("type", "")
        is_channel = block_type == "Channel" or self.block.get("base_type") == "Channel"

        if is_channel:
            slug = self.block.get("slug") or ""
            name = self.block.get("title") or slug or "channel"
        else:
            name = _block_label(self.block, "", self.channel_path)

        if self.label == "relative":
            return name
        if self.label == "name":
            return name.rsplit("/", 1)[-1] if "/" in name else name
        if self.label == "ext":
            return ""
        return self.label

    def _get_contents(self) -> str:
        block_type = self.block.get("type", "")
        is_channel = block_type == "Channel" or self.block.get("base_type") == "Channel"

        if is_channel:
            text = _render_channel_stub(self.block)
        else:
            if block_type in ("Image", "Attachment"):
                block_title = self.block.get("title") or f"block-{self.block.get('id')}"
                _log(f"  resolving {block_type.lower()}: {block_title[:60]}")
            text = (
                _render_block(
                    self.block, include_descriptions=self.include_descriptions
                )
                or ""
            )

        self.original_file_content = text
        self.file_content = text

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

    def get_contents(self) -> str:
        return self.output
