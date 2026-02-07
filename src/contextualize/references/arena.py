from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
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


def _api_get(path: str, params: dict | None = None) -> dict:
    import requests

    url = f"{_API_BASE}{path}"
    headers = {**_get_auth_headers(), "Accept": "application/json"}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code == 429:
        import time

        reset = resp.headers.get("X-RateLimit-Reset")
        wait = int(reset) if reset else 5
        print(f"Are.na rate limited, waiting {wait}s...", file=sys.stderr)
        time.sleep(wait)
        resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code == 404:
        raise ValueError(f"Are.na resource not found: {path}")
    resp.raise_for_status()
    return resp.json()


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


def _get_recurse_users() -> set[str] | None:
    raw = os.environ.get("ARENA_RECURSE_USERS", "").strip()
    if not raw:
        return {"self"}
    if raw.lower() == "all":
        return None
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


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


def _download_to_temp(url: str, suffix: str = "") -> Path | None:
    import requests

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None
    if not resp.content:
        return None
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, resp.content)
    finally:
        os.close(fd)
    return Path(path)


def _render_block_binary(url: str, suffix: str) -> str:
    from ..render.markitdown import MarkItDownConversionError, convert_path_to_markdown

    tmp = _download_to_temp(url, suffix=suffix)
    if tmp is None:
        return ""
    try:
        result = convert_path_to_markdown(str(tmp))
        return result.markdown
    except MarkItDownConversionError:
        return ""
    finally:
        tmp.unlink(missing_ok=True)


def _render_block(block: dict) -> str | None:
    block_type = block.get("class") or block.get("type", "")
    state = block.get("state")
    if state == "processing" or block_type == "PendingBlock":
        return None

    title = block.get("title") or ""
    if _get_include_descriptions():
        raw_desc = block.get("description") or ""
        if isinstance(raw_desc, dict):
            description = raw_desc.get("markdown") or raw_desc.get("plain") or ""
        else:
            description = raw_desc
    else:
        description = ""

    if block_type == "Text":
        raw_content = block.get("content") or ""
        if isinstance(raw_content, dict):
            content = raw_content.get("markdown") or raw_content.get("plain") or ""
        else:
            content = raw_content
        parts = []
        if title:
            parts.append(title)
        if content:
            parts.append(content)
        if description and description != content:
            parts.append(description)
        return "\n\n".join(parts) if parts else None

    if block_type == "Image":
        image = block.get("image") or {}
        image_urls = [
            u
            for u in (
                image.get("src"),
                (image.get("large") or {}).get("src"),
                (image.get("original") or {}).get("url"),
                image.get("url"),
            )
            if u
        ]
        for image_url in image_urls:
            suffix = Path(image_url.split("?")[0]).suffix or ".jpg"
            converted = _render_block_binary(image_url, suffix)
            if converted:
                parts = []
                if title:
                    parts.append(title)
                parts.append(converted)
                return "\n\n".join(parts)
        fallback_url = image_urls[0] if image_urls else ""
        parts = [f"[Image: {title or block.get('id')}]"]
        if fallback_url:
            parts.append(f"URL: {fallback_url}")
        return "\n".join(parts)

    if block_type == "Link":
        source = block.get("source") or {}
        source_url = source.get("url") or ""
        parts = []
        if title:
            parts.append(title)
        if source_url:
            parts.append(source_url)
        raw_content = block.get("content") or ""
        if isinstance(raw_content, dict):
            content = raw_content.get("markdown") or raw_content.get("plain") or ""
        else:
            content = raw_content
        if content:
            parts.append(content)
        elif description:
            parts.append(description)
        return "\n\n".join(parts) if parts else None

    if block_type == "Attachment":
        attachment = block.get("attachment") or {}
        att_url = attachment.get("url") or ""
        filename = attachment.get("filename") or ""
        content_type = attachment.get("content_type") or ""
        extension = attachment.get("file_extension") or ""
        if att_url:
            suffix = f".{extension}" if extension else Path(filename).suffix or ""
            converted = _render_block_binary(att_url, suffix)
            if converted:
                parts = []
                if title and title != filename:
                    parts.append(title)
                parts.append(converted)
                return "\n\n".join(parts)
        parts = [f"[Attachment: {filename or title or block.get('id')}]"]
        if content_type:
            parts.append(f"Type: {content_type}")
        if att_url:
            parts.append(f"URL: {att_url}")
        return "\n".join(parts)

    if block_type == "Embed":
        embed = block.get("embed") or {}
        embed_url = embed.get("url") or ""
        embed_type = embed.get("type") or ""
        parts = []
        if title:
            parts.append(title)
        if embed_url:
            parts.append(embed_url)
        if embed_type:
            parts.append(f"Type: {embed_type}")
        if description:
            parts.append(description)
        return "\n\n".join(parts) if parts else None

    parts = []
    if title:
        parts.append(title)
    if description:
        parts.append(description)
    return "\n\n".join(parts) if parts else None


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
    block_type = block.get("type", "")
    title = block.get("title") or ""
    block_id = block.get("id", "unknown")

    if block_type == "Attachment":
        attachment = block.get("attachment") or {}
        name = attachment.get("filename") or title or f"block-{block_id}"
    elif title:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", title.strip())[:80].strip("-")
        name = safe or f"block-{block_id}"
    else:
        name = f"block-{block_id}"

    prefix = channel_path or channel_slug
    return f"{prefix}/{name}"


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


def resolve_channel(
    slug: str,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> tuple[dict, list[tuple[str, dict]]]:
    max_depth = _get_max_depth()
    recurse_users = _get_recurse_users()
    ru_key = ",".join(sorted(recurse_users)) if recurse_users else "all"
    cache_key = f"{slug}:d={max_depth}:u={ru_key}"

    if use_cache and not refresh_cache:
        from ..cache.arena import get_cached_channel

        cached = get_cached_channel(cache_key, cache_ttl)
        if cached is not None:
            data = json.loads(cached)
            metadata = data["metadata"]
            flat = [(path, block) for path, block in data["blocks"]]
            return metadata, flat

    metadata, contents = _fetch_all_channel_contents(slug)
    flat = _flatten_channel_blocks(contents, slug)

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
            text = _render_block(self.block) or ""

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
