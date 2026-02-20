from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..render.text import process_text
from ..utils import count_tokens

_AT_URI_RE = re.compile(
    r"^at://(?P<repo>[^/?#]+)"
    r"(?:/(?P<collection>[^/?#]+)"
    r"(?:/(?P<rkey>[^/?#]+))?)?$",
    flags=re.IGNORECASE,
)
_BSKY_APP_RE = re.compile(r"^https?://(?:www\.)?bsky\.app(?:/|$)", flags=re.IGNORECASE)
_DID_RE = re.compile(r"^did:[a-z0-9]+:[A-Za-z0-9._:%-]+$", flags=re.IGNORECASE)
_BSKY_HANDLE_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9-]{1,63})*\.bsky\.social$",
    flags=re.IGNORECASE,
)
_VALID_MEDIA_MODES = frozenset({"describe", "transcribe"})
_MEDIA_DOWNLOAD_HEADERS = {
    "User-Agent": "contextualize/atproto",
    "Accept": "image/*,video/*,audio/*,application/octet-stream,*/*;q=0.8",
}

_KNOWN_COLLECTION_TO_KIND = {
    "app.bsky.feed.post": "post",
    "app.bsky.feed.generator": "feed",
    "app.bsky.graph.list": "list",
    "app.bsky.graph.starterpack": "starter-pack",
    "app.bsky.labeler.service": "labeler",
}
_KNOWN_KIND_TO_COLLECTION = {
    "post": "app.bsky.feed.post",
    "feed": "app.bsky.feed.generator",
    "list": "app.bsky.graph.list",
    "starter-pack": "app.bsky.graph.starterpack",
    "labeler": "app.bsky.labeler.service",
}
_DEFAULT_PUBLIC_APPVIEW = "https://public.api.bsky.app"
_DEFAULT_PDS = "https://bsky.social"


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
def warmup_atproto_network_stack() -> None:
    try:
        import lexrpc  # noqa: F401
        import requests  # noqa: F401
    except Exception:
        return


def is_at_uri(value: str) -> bool:
    return bool(_AT_URI_RE.match(value.strip()))


def is_bsky_app_url(url: str) -> bool:
    return bool(_BSKY_APP_RE.match(url.strip()))


def is_atproto_url(value: str) -> bool:
    return parse_atproto_target(value) is not None


@dataclass(frozen=True)
class AtprotoSettings:
    max_items: int = 25
    thread_parent_height: int = 8
    thread_depth: int = 6
    include_replies: bool = False
    include_media_descriptions: bool = True
    include_embed_media_descriptions: bool = True
    media_mode: str = "describe"
    quote_depth: int = 1
    max_replies: int = 0
    reply_quote_depth: int = 1


@dataclass(frozen=True)
class AtprotoTarget:
    kind: str
    original: str
    actor: str | None = None
    repo: str | None = None
    collection: str | None = None
    rkey: str | None = None
    uri: str | None = None


@dataclass(frozen=True)
class AtprotoDocument:
    source_url: str
    kind: str
    uri: str
    label: str
    trace_path: str
    context_subpath: str
    rendered: str
    source_created: str | None = None
    source_modified: str | None = None


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


def _parse_media_mode(value: str, *, default: str) -> str:
    cleaned = value.strip().lower()
    if cleaned in _VALID_MEDIA_MODES:
        return cleaned
    return default


def _atproto_settings_from_env() -> AtprotoSettings:
    _load_dotenv()
    max_items = _parse_positive_int(
        os.environ.get("ATPROTO_MAX_ITEMS", ""),
        default=25,
        minimum=1,
    )
    thread_parent_height = _parse_positive_int(
        os.environ.get("ATPROTO_THREAD_PARENT_HEIGHT", ""),
        default=8,
        minimum=0,
    )
    thread_depth = _parse_positive_int(
        os.environ.get("ATPROTO_THREAD_DEPTH", ""),
        default=6,
        minimum=0,
    )
    include_replies = _parse_bool(
        os.environ.get("ATPROTO_INCLUDE_REPLIES", "0"),
        default=False,
    )
    include_media_descriptions = _parse_bool(
        os.environ.get("ATPROTO_MEDIA_DESCRIPTIONS", "1"),
        default=True,
    )
    include_embed_media_descriptions = _parse_bool(
        os.environ.get("ATPROTO_EMBED_MEDIA_DESCRIPTIONS", "1"),
        default=True,
    )
    media_mode = _parse_media_mode(
        os.environ.get("ATPROTO_MEDIA_MODE", ""),
        default="describe",
    )
    quote_depth = _parse_positive_int(
        os.environ.get("ATPROTO_QUOTE_DEPTH", ""),
        default=1,
        minimum=0,
    )
    max_replies = _parse_positive_int(
        os.environ.get("ATPROTO_MAX_REPLIES", ""),
        default=0,
        minimum=0,
    )
    reply_quote_depth = _parse_positive_int(
        os.environ.get("ATPROTO_REPLY_QUOTE_DEPTH", ""),
        default=1,
        minimum=0,
    )
    return AtprotoSettings(
        max_items=max_items,
        thread_parent_height=thread_parent_height,
        thread_depth=thread_depth,
        include_replies=include_replies,
        include_media_descriptions=include_media_descriptions,
        include_embed_media_descriptions=include_embed_media_descriptions,
        media_mode=media_mode,
        quote_depth=quote_depth,
        max_replies=max_replies,
        reply_quote_depth=reply_quote_depth,
    )


def build_atproto_settings(overrides: dict[str, Any] | None = None) -> AtprotoSettings:
    env = _atproto_settings_from_env()
    if not overrides:
        return env
    max_items = int(overrides.get("max_items", env.max_items))
    thread_parent_height = int(
        overrides.get("thread_parent_height", env.thread_parent_height)
    )
    thread_depth = int(overrides.get("thread_depth", env.thread_depth))
    include_replies = bool(overrides.get("include_replies", env.include_replies))
    include_media_descriptions = bool(
        overrides.get("include_media_descriptions", env.include_media_descriptions)
    )
    include_embed_media_descriptions = bool(
        overrides.get(
            "include_embed_media_descriptions",
            env.include_embed_media_descriptions,
        )
    )
    media_mode = _parse_media_mode(
        str(overrides.get("media_mode", env.media_mode) or ""),
        default=env.media_mode,
    )
    quote_depth = int(overrides.get("quote_depth", env.quote_depth))
    max_replies = int(overrides.get("max_replies", env.max_replies))
    reply_quote_depth = int(overrides.get("reply_quote_depth", env.reply_quote_depth))
    return AtprotoSettings(
        max_items=max(1, max_items),
        thread_parent_height=max(0, thread_parent_height),
        thread_depth=max(0, thread_depth),
        include_replies=include_replies,
        include_media_descriptions=include_media_descriptions,
        include_embed_media_descriptions=include_embed_media_descriptions,
        media_mode=media_mode,
        quote_depth=max(0, quote_depth),
        max_replies=max(0, max_replies),
        reply_quote_depth=max(0, reply_quote_depth),
    )


def atproto_settings_cache_key(settings: AtprotoSettings) -> tuple[Any, ...]:
    return (
        settings.max_items,
        settings.thread_parent_height,
        settings.thread_depth,
        settings.include_replies,
        settings.include_media_descriptions,
        settings.include_embed_media_descriptions,
        settings.media_mode,
        settings.quote_depth,
        settings.max_replies,
        settings.reply_quote_depth,
    )


def parse_atproto_target(value: str) -> AtprotoTarget | None:
    raw = value.strip()
    if not raw:
        return None

    if _DID_RE.match(raw) or _BSKY_HANDLE_RE.match(raw):
        return AtprotoTarget(
            kind="profile",
            original=value,
            actor=raw,
            repo=raw,
            uri=f"at://{raw}",
        )

    uri_match = _AT_URI_RE.match(raw)
    if uri_match:
        repo = uri_match.group("repo")
        collection = uri_match.group("collection")
        rkey = uri_match.group("rkey")
        kind = _KNOWN_COLLECTION_TO_KIND.get(collection or "", "record")
        if collection is None:
            kind = "profile"
        canonical_uri = f"at://{repo}"
        if collection:
            canonical_uri = f"{canonical_uri}/{collection}"
            if rkey:
                canonical_uri = f"{canonical_uri}/{rkey}"
        return AtprotoTarget(
            kind=kind,
            original=value,
            repo=repo,
            actor=repo,
            collection=collection,
            rkey=rkey,
            uri=canonical_uri,
        )

    if not is_bsky_app_url(raw):
        return None
    parsed = urlparse(raw)
    parts = [segment for segment in parsed.path.split("/") if segment]
    if not parts:
        return None
    if parts[0] == "profile":
        if len(parts) == 2:
            actor = parts[1]
            return AtprotoTarget(
                kind="profile",
                original=value,
                actor=actor,
                repo=actor,
                uri=f"at://{actor}",
            )
        if len(parts) >= 4:
            actor = parts[1]
            scope = parts[2]
            rkey = parts[3]
            if scope == "post":
                collection = "app.bsky.feed.post"
                kind = "post"
            elif scope == "feed":
                collection = "app.bsky.feed.generator"
                kind = "feed"
            elif scope == "lists":
                collection = "app.bsky.graph.list"
                kind = "list"
            elif scope == "labeler":
                collection = "app.bsky.labeler.service"
                kind = "labeler"
            else:
                return None
            return AtprotoTarget(
                kind=kind,
                original=value,
                actor=actor,
                repo=actor,
                collection=collection,
                rkey=rkey,
                uri=f"at://{actor}/{collection}/{rkey}",
            )
    if parts[0] == "starter-pack" and len(parts) >= 3:
        actor = parts[1]
        rkey = parts[2]
        collection = "app.bsky.graph.starterpack"
        return AtprotoTarget(
            kind="starter-pack",
            original=value,
            actor=actor,
            repo=actor,
            collection=collection,
            rkey=rkey,
            uri=f"at://{actor}/{collection}/{rkey}",
        )
    return None


def _documents_from_cached_payload(payload: Any) -> list[AtprotoDocument] | None:
    if not isinstance(payload, list):
        return None
    documents: list[AtprotoDocument] = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        try:
            doc = AtprotoDocument(**item)
        except TypeError:
            return None
        documents.append(doc)
    return documents


def _resolution_cache_identity(
    url: str, settings: AtprotoSettings, auth_mode: str
) -> str:
    payload = {
        "v": 2,
        "url": url,
        "settings": atproto_settings_cache_key(settings),
        "auth_mode": auth_mode,
    }
    return "atproto-resolve:" + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )


def _media_render_cache_identity(
    *,
    media_cache_identity: str,
    prompt_append: str | None,
    suffix: str,
    mode: str,
    media_kind: str,
) -> str:
    payload = {
        "media_cache_identity": media_cache_identity,
        "prompt_append": prompt_append or "",
        "suffix": suffix,
        "mode": mode,
        "media_kind": media_kind,
    }
    return "atproto-media-render:" + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )


@dataclass
class _ClientContext:
    client: Any
    auth_mode: str


def _build_client_context() -> _ClientContext:
    _load_dotenv()
    from lexrpc.client import Client

    identifier = (os.environ.get("ATPROTO_IDENTIFIER") or "").strip()
    password = (os.environ.get("ATPROTO_APP_PASSWORD") or "").strip()

    if identifier and password:
        pds_host = (os.environ.get("ATPROTO_PDS_HOST") or "").strip() or _DEFAULT_PDS
        client = Client(address=pds_host, timeout=30)
        try:
            client.com.atproto.server.createSession(
                {"identifier": identifier, "password": password}
            )
            _log("  atproto auth mode: authenticated")
            return _ClientContext(client=client, auth_mode="auth")
        except Exception as exc:
            _log(f"  atproto auth failed, falling back to public mode: {exc}")

    public_host = (
        os.environ.get("ATPROTO_PUBLIC_HOST") or ""
    ).strip() or _DEFAULT_PUBLIC_APPVIEW
    client = Client(address=public_host, timeout=30)
    _log("  atproto auth mode: public")
    return _ClientContext(client=client, auth_mode="public")


def _client_call(label: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        raise ValueError(f"ATProto {label} failed: {exc}") from exc


def _resolve_actor_to_did(
    actor: str,
    *,
    client: Any,
    use_cache: bool,
    refresh_cache: bool,
) -> str:
    from ..cache.atproto import get_cached_handle_did, store_handle_did

    cleaned = actor.strip().lstrip("@")
    if cleaned.startswith("did:"):
        return cleaned
    if use_cache and not refresh_cache:
        cached = get_cached_handle_did(cleaned)
        if cached:
            return cached
    resolved = _client_call(
        "identity.resolveHandle",
        client.com.atproto.identity.resolveHandle,
        handle=cleaned,
    )
    did = str(resolved.get("did") or "").strip()
    if not did.startswith("did:"):
        raise ValueError(f"Could not resolve handle to DID: {actor}")
    if use_cache:
        store_handle_did(cleaned, did)
    return did


def _canonicalize_target(
    target: AtprotoTarget,
    *,
    client: Any,
    use_cache: bool,
    refresh_cache: bool,
) -> AtprotoTarget:
    repo = target.repo or target.actor
    if repo:
        repo = _resolve_actor_to_did(
            repo,
            client=client,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )
    collection = target.collection
    if collection is None and target.kind in _KNOWN_KIND_TO_COLLECTION and target.rkey:
        collection = _KNOWN_KIND_TO_COLLECTION[target.kind]
    canonical_uri = f"at://{repo}" if repo else (target.uri or "")
    if collection:
        canonical_uri = f"{canonical_uri}/{collection}"
        if target.rkey:
            canonical_uri = f"{canonical_uri}/{target.rkey}"
    return AtprotoTarget(
        kind=target.kind,
        original=target.original,
        actor=target.actor,
        repo=repo,
        collection=collection,
        rkey=target.rkey,
        uri=canonical_uri or target.uri,
    )


def _safe_slug(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _parse_at_uri(uri: str) -> tuple[str | None, str | None, str | None]:
    match = _AT_URI_RE.match(uri.strip())
    if not match:
        return None, None, None
    return match.group("repo"), match.group("collection"), match.group("rkey")


def _uri_to_bsky_url(uri: str, handle: str | None = None) -> str | None:
    repo, collection, rkey = _parse_at_uri(uri)
    if not repo:
        return None
    actor = handle or repo
    if collection is None:
        return f"https://bsky.app/profile/{actor}"
    if collection == "app.bsky.feed.post" and rkey:
        return f"https://bsky.app/profile/{actor}/post/{rkey}"
    if collection == "app.bsky.feed.generator" and rkey:
        return f"https://bsky.app/profile/{actor}/feed/{rkey}"
    if collection == "app.bsky.graph.list" and rkey:
        return f"https://bsky.app/profile/{actor}/lists/{rkey}"
    if collection == "app.bsky.graph.starterpack" and rkey:
        return f"https://bsky.app/starter-pack/{actor}/{rkey}"
    if collection == "app.bsky.labeler.service" and rkey:
        return f"https://bsky.app/profile/{actor}/labeler/{rkey}"
    return None


def _post_label(post: dict[str, Any], fallback_index: int) -> str:
    uri = str(post.get("uri") or "")
    _repo, collection, rkey = _parse_at_uri(uri)
    if collection and rkey:
        return f"{collection}-{rkey}"
    return f"post-{fallback_index:03d}"


def _render_rich_text(text: str, facets: Any) -> str:
    if not text or not isinstance(facets, list):
        return text
    text_bytes = text.encode("utf-8")
    pieces: list[str] = []
    cursor = 0

    def _feature_replacement(snippet: str, feature: dict[str, Any]) -> str:
        ftype = str(feature.get("$type") or "")
        if "facet#link" in ftype or ftype.endswith(".facet#link"):
            uri = feature.get("uri")
            if isinstance(uri, str) and uri:
                return f"[{snippet}]({uri})"
        if "facet#mention" in ftype or ftype.endswith(".facet#mention"):
            did = feature.get("did")
            if isinstance(did, str) and did:
                return f"[{snippet}](https://bsky.app/profile/{did})"
        if "facet#tag" in ftype or ftype.endswith(".facet#tag"):
            tag = feature.get("tag")
            if isinstance(tag, str) and tag:
                return f"[#{tag}](https://bsky.app/hashtag/{tag})"
        return snippet

    sorted_facets = sorted(
        [item for item in facets if isinstance(item, dict)],
        key=lambda item: int((item.get("index") or {}).get("byteStart") or 0),
    )
    for facet in sorted_facets:
        index = facet.get("index")
        if not isinstance(index, dict):
            continue
        start = index.get("byteStart")
        end = index.get("byteEnd")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or start > len(text_bytes):
            continue
        end = min(end, len(text_bytes))
        if start < cursor:
            continue

        pieces.append(text_bytes[cursor:start].decode("utf-8", errors="ignore"))
        snippet = text_bytes[start:end].decode("utf-8", errors="ignore")
        features = facet.get("features")
        replacement = snippet
        if isinstance(features, list):
            first_feature = next(
                (item for item in features if isinstance(item, dict)),
                None,
            )
            if first_feature is not None:
                replacement = _feature_replacement(snippet, first_feature)
        pieces.append(replacement)
        cursor = end

    pieces.append(text_bytes[cursor:].decode("utf-8", errors="ignore"))
    return "".join(pieces)


def _embed_type(embed: Any) -> str:
    if not isinstance(embed, dict):
        return ""
    return str(embed.get("$type") or "")


def _extract_record_embed(record_embed: Any, view_embed: Any) -> tuple[Any, Any]:
    record_media = record_embed
    view_media = view_embed
    if "recordwithmedia" in _embed_type(record_embed).lower():
        record_media = (
            record_embed.get("media") if isinstance(record_embed, dict) else {}
        )
    if "recordwithmedia" in _embed_type(view_embed).lower():
        view_media = view_embed.get("media") if isinstance(view_embed, dict) else {}
    return record_media or {}, view_media or {}


def _media_kind(
    *, mime: str | None = None, url: str | None = None, fallback: str = "file"
) -> str:
    mtype = (mime or "").lower().strip()
    if mtype.startswith("image/"):
        return "image"
    if mtype.startswith("video/"):
        return "video"
    if mtype.startswith("audio/"):
        return "audio"
    suffix = Path((url or "").split("?")[0]).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".heic", ".heif"}:
        return "image"
    if suffix in {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}:
        return "video"
    if suffix in {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".aiff"}:
        return "audio"
    return fallback


def _media_suffix(*, mime: str | None, url: str | None, kind: str) -> str:
    suffix = Path((url or "").split("?")[0]).suffix
    if suffix:
        return suffix
    mtype = (mime or "").lower().strip()
    if mtype == "image/jpeg":
        return ".jpg"
    if mtype == "image/png":
        return ".png"
    if mtype == "image/webp":
        return ".webp"
    if mtype == "video/mp4":
        return ".mp4"
    if mtype == "audio/mpeg":
        return ".mp3"
    if mtype == "audio/wav":
        return ".wav"
    if kind == "image":
        return ".jpg"
    if kind == "video":
        return ".mp4"
    if kind == "audio":
        return ".mp3"
    return ".bin"


def _collect_media_entries(post: dict[str, Any]) -> list[dict[str, Any]]:
    record = post.get("record")
    if not isinstance(record, dict):
        record = post.get("value") if isinstance(post.get("value"), dict) else {}

    record_embed = record.get("embed") if isinstance(record.get("embed"), dict) else {}
    view_embed = post.get("embed") if isinstance(post.get("embed"), dict) else {}
    if not view_embed:
        embeds = post.get("embeds")
        if isinstance(embeds, list):
            first_embed = next(
                (item for item in embeds if isinstance(item, dict)),
                None,
            )
            view_embed = first_embed or {}

    record_media, view_media = _extract_record_embed(record_embed, view_embed)

    media: list[dict[str, Any]] = []
    record_images = (
        record_media.get("images")
        if isinstance(record_media, dict)
        and isinstance(record_media.get("images"), list)
        else []
    )
    view_images = (
        view_media.get("images")
        if isinstance(view_media, dict) and isinstance(view_media.get("images"), list)
        else []
    )
    for index in range(max(len(record_images), len(view_images))):
        rec_img = record_images[index] if index < len(record_images) else {}
        view_img = view_images[index] if index < len(view_images) else {}
        rec_blob = rec_img.get("image") if isinstance(rec_img, dict) else {}
        alt = ""
        if isinstance(rec_img, dict) and isinstance(rec_img.get("alt"), str):
            alt = rec_img.get("alt", "")
        elif isinstance(view_img, dict) and isinstance(view_img.get("alt"), str):
            alt = view_img.get("alt", "")
        ar = None
        if isinstance(rec_img, dict) and isinstance(rec_img.get("aspectRatio"), dict):
            ar = rec_img.get("aspectRatio")
        elif isinstance(view_img, dict) and isinstance(
            view_img.get("aspectRatio"), dict
        ):
            ar = view_img.get("aspectRatio")
        fullsize = view_img.get("fullsize") if isinstance(view_img, dict) else None
        thumb = view_img.get("thumb") if isinstance(view_img, dict) else None
        url = fullsize or thumb
        if not isinstance(url, str) or not url:
            continue
        cid = None
        if isinstance(rec_blob, dict):
            ref = rec_blob.get("ref")
            if isinstance(ref, dict) and isinstance(ref.get("$link"), str):
                cid = ref["$link"]
        mime = rec_blob.get("mimeType") if isinstance(rec_blob, dict) else None
        kind = _media_kind(mime=mime, url=url, fallback="image")
        media.append(
            {
                "kind": kind,
                "url": url,
                "alt": alt or None,
                "mime": mime if isinstance(mime, str) else None,
                "width": ar.get("width") if isinstance(ar, dict) else None,
                "height": ar.get("height") if isinstance(ar, dict) else None,
                "cache_identity": f"atproto:blob:{cid}"
                if cid
                else f"atproto:url:{url}",
                "is_embed_media": True,
            }
        )

    video_url = None
    video_alt = None
    video_ar = None
    video_mime = None
    if isinstance(view_media, dict):
        playlist = view_media.get("playlist")
        thumbnail = view_media.get("thumbnail")
        if isinstance(playlist, str) and playlist:
            video_url = playlist
        elif isinstance(thumbnail, str) and thumbnail:
            video_url = thumbnail
        if isinstance(view_media.get("alt"), str):
            video_alt = view_media.get("alt")
        if isinstance(view_media.get("aspectRatio"), dict):
            video_ar = view_media.get("aspectRatio")
    if isinstance(record_media, dict):
        if isinstance(record_media.get("alt"), str) and not video_alt:
            video_alt = record_media.get("alt")
        if isinstance(record_media.get("aspectRatio"), dict) and not video_ar:
            video_ar = record_media.get("aspectRatio")
        blob = (
            record_media.get("video")
            if isinstance(record_media.get("video"), dict)
            else record_media
        )
        if isinstance(blob, dict):
            maybe_mime = blob.get("mimeType")
            if isinstance(maybe_mime, str):
                video_mime = maybe_mime
    if isinstance(video_url, str) and video_url:
        media.append(
            {
                "kind": _media_kind(mime=video_mime, url=video_url, fallback="video"),
                "url": video_url,
                "alt": video_alt or None,
                "mime": video_mime,
                "width": video_ar.get("width") if isinstance(video_ar, dict) else None,
                "height": video_ar.get("height")
                if isinstance(video_ar, dict)
                else None,
                "cache_identity": f"atproto:url:{video_url}",
                "is_embed_media": True,
            }
        )

    return media


def _extract_external_link(post: dict[str, Any]) -> dict[str, Any] | None:
    record = post.get("record")
    if not isinstance(record, dict):
        record = post.get("value") if isinstance(post.get("value"), dict) else {}
    record_embed = record.get("embed") if isinstance(record.get("embed"), dict) else {}
    view_embed = post.get("embed") if isinstance(post.get("embed"), dict) else {}
    if not view_embed:
        embeds = post.get("embeds")
        if isinstance(embeds, list):
            first = next((item for item in embeds if isinstance(item, dict)), None)
            view_embed = first or {}
    record_media, view_media = _extract_record_embed(record_embed, view_embed)

    external = None
    if isinstance(view_media, dict) and isinstance(view_media.get("external"), dict):
        external = view_media.get("external")
    elif isinstance(record_media, dict) and isinstance(
        record_media.get("external"), dict
    ):
        external = record_media.get("external")
    if not isinstance(external, dict):
        return None
    uri = external.get("uri")
    if not isinstance(uri, str) or not uri:
        return None
    title = external.get("title")
    description = external.get("description")
    thumb = external.get("thumb")
    return {
        "uri": uri,
        "title": title.strip() if isinstance(title, str) and title.strip() else None,
        "description": description.strip()
        if isinstance(description, str) and description.strip()
        else None,
        "thumb": thumb if isinstance(thumb, str) and thumb else None,
    }


def _facet_link_uris(post: dict[str, Any]) -> list[str]:
    record = post.get("record")
    if not isinstance(record, dict):
        record = post.get("value") if isinstance(post.get("value"), dict) else {}
    facets = record.get("facets")
    if not isinstance(facets, list):
        return []
    links: list[str] = []
    seen: set[str] = set()
    for facet in facets:
        if not isinstance(facet, dict):
            continue
        features = facet.get("features")
        if not isinstance(features, list):
            continue
        for feature in features:
            if not isinstance(feature, dict):
                continue
            uri = feature.get("uri")
            if not isinstance(uri, str) or not uri:
                continue
            if uri in seen:
                continue
            seen.add(uri)
            links.append(uri)
    return links


def _extract_quote_view(post: dict[str, Any]) -> dict[str, Any] | None:
    record = post.get("record")
    if not isinstance(record, dict):
        record = post.get("value") if isinstance(post.get("value"), dict) else {}

    record_embed = record.get("embed") if isinstance(record.get("embed"), dict) else {}
    view_embed = post.get("embed") if isinstance(post.get("embed"), dict) else {}
    if not view_embed:
        embeds = post.get("embeds")
        if isinstance(embeds, list):
            view_embed = (
                next((item for item in embeds if isinstance(item, dict)), {}) or {}
            )

    if "recordwithmedia" in _embed_type(view_embed).lower():
        record_part = view_embed.get("record") if isinstance(view_embed, dict) else None
        if isinstance(record_part, dict):
            candidate = record_part.get("record")
            if isinstance(candidate, dict):
                return candidate
            return record_part

    if "embed.record" in _embed_type(view_embed).lower() and isinstance(
        view_embed, dict
    ):
        candidate = view_embed.get("record")
        if isinstance(candidate, dict):
            return candidate

    if "recordwithmedia" in _embed_type(record_embed).lower() and isinstance(
        record_embed, dict
    ):
        record_part = record_embed.get("record")
        if isinstance(record_part, dict):
            candidate = record_part.get("record")
            if isinstance(candidate, dict):
                return candidate
            return record_part
    if "embed.record" in _embed_type(record_embed).lower() and isinstance(
        record_embed, dict
    ):
        candidate = record_embed.get("record")
        if isinstance(candidate, dict):
            return candidate
    return None


def _quote_to_post(quote: dict[str, Any]) -> dict[str, Any] | None:
    qtype = str(quote.get("$type") or "")
    if "blocked" in qtype.lower() or "notfound" in qtype.lower():
        return None
    if not isinstance(quote.get("uri"), str):
        return None
    record = quote.get("value")
    if not isinstance(record, dict):
        record = quote.get("record") if isinstance(quote.get("record"), dict) else {}
    return {
        "uri": quote.get("uri"),
        "cid": quote.get("cid"),
        "author": quote.get("author") if isinstance(quote.get("author"), dict) else {},
        "record": record,
        "embed": quote.get("embed") if isinstance(quote.get("embed"), dict) else {},
        "embeds": quote.get("embeds") if isinstance(quote.get("embeds"), list) else [],
        "indexedAt": quote.get("indexedAt"),
        "replyCount": quote.get("replyCount"),
        "repostCount": quote.get("repostCount"),
        "likeCount": quote.get("likeCount"),
        "quoteCount": quote.get("quoteCount"),
    }


def _render_json_block(payload: Any) -> str:
    return "```json\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n```"


def _should_refresh_kind(kind: str) -> bool:
    from ..runtime import (
        get_refresh_audio,
        get_refresh_images,
        get_refresh_media,
        get_refresh_videos,
    )

    if get_refresh_media():
        return True
    if kind == "image":
        return get_refresh_images()
    if kind == "video":
        return get_refresh_videos()
    if kind == "audio":
        return get_refresh_audio()
    return False


def _describe_media(
    media: dict[str, Any],
    *,
    settings: AtprotoSettings,
) -> str | None:
    from ..cache.atproto import (
        get_cached_media_bytes,
        get_cached_rendered,
        store_media_bytes,
        store_rendered,
    )
    from ..render.markitdown import convert_path_to_markdown
    from .audio_transcription import transcribe_audio_file
    from .media import download_cached_media_to_temp

    url = media.get("url")
    if not isinstance(url, str) or not url:
        return None
    kind = str(media.get("kind") or "file")
    if not settings.include_media_descriptions:
        return None
    if media.get("is_embed_media") and not settings.include_embed_media_descriptions:
        return None
    suffix = _media_suffix(mime=media.get("mime"), url=url, kind=kind)
    prompt_append = media.get("alt") if isinstance(media.get("alt"), str) else None
    cache_identity = str(media.get("cache_identity") or f"atproto:url:{url}")
    render_identity = _media_render_cache_identity(
        media_cache_identity=cache_identity,
        prompt_append=prompt_append,
        suffix=suffix,
        mode=settings.media_mode,
        media_kind=kind,
    )
    refresh_for_kind = _should_refresh_kind(kind)
    if not refresh_for_kind:
        cached = get_cached_rendered(render_identity)
        if isinstance(cached, str) and cached.strip():
            return cached.strip()

    tmp = download_cached_media_to_temp(
        url,
        suffix=suffix,
        headers=_MEDIA_DOWNLOAD_HEADERS,
        cache_identity=cache_identity,
        get_cached_media_bytes=get_cached_media_bytes,
        store_media_bytes=store_media_bytes,
        refresh_cache=refresh_for_kind,
    )
    if tmp is None:
        return None
    try:
        markdown = ""
        if settings.media_mode == "transcribe" and kind == "audio":
            markdown = transcribe_audio_file(tmp)
        if not markdown:
            result = convert_path_to_markdown(
                tmp,
                refresh_images=refresh_for_kind,
                prompt_append=prompt_append,
            )
            markdown = result.markdown
        cleaned = _normalize_llm_description(markdown)
        if not cleaned:
            return None
        store_rendered(render_identity, cleaned)
        return cleaned
    except Exception:
        return None
    finally:
        tmp.unlink(missing_ok=True)


def _normalize_llm_description(markdown: str) -> str:
    text = markdown.strip()
    if not text:
        return ""
    lines = text.splitlines()

    # Drop image size line; dimensions are rendered separately in metadata.
    if lines and lines[0].strip().startswith("ImageSize:"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]

    # Drop generic markitdown heading prefixes.
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


def _append_field_line(
    lines: list[str],
    key: str,
    value: str,
    *,
    literal: bool = False,
    prefix: str = "  - ",
) -> None:
    cleaned = value.strip()
    if not cleaned:
        return
    leading_spaces = len(prefix) - len(prefix.lstrip(" "))
    literal_prefix = " " * (leading_spaces + 4)
    if literal or "\n" in cleaned:
        lines.append(f"{prefix}{key}: |")
        lines.extend(
            f"{literal_prefix}{line}" if line else literal_prefix
            for line in cleaned.splitlines()
        )
        return
    lines.append(f"{prefix}{key}: {cleaned}")


def _render_media_section(
    media_items: list[dict[str, Any]], settings: AtprotoSettings
) -> str:
    lines: list[str] = []
    for media in media_items:
        url = str(media.get("url") or "")
        if not url:
            continue
        kind = str(media.get("kind") or "file")
        lines.append(f"- {url} ({kind})")
        alt = media.get("alt") if isinstance(media.get("alt"), str) else None
        if alt:
            _append_field_line(lines, "alt", alt)
        width = media.get("width")
        height = media.get("height")
        if isinstance(width, int) and isinstance(height, int):
            lines.append(f"  - dimensions: {width}x{height}")
        generated = _describe_media(media, settings=settings)
        if generated:
            _append_field_line(lines, "llm-description", generated, literal=True)
    return "\n".join(lines)


def _author_name(author: Any) -> str:
    if not isinstance(author, dict):
        return "unknown"
    display_name = author.get("displayName")
    handle = author.get("handle")
    did = author.get("did")
    if isinstance(display_name, str) and display_name.strip():
        if isinstance(handle, str) and handle.strip():
            return f"{display_name.strip()} (@{handle.strip()})"
        return display_name.strip()
    if isinstance(handle, str) and handle.strip():
        return f"@{handle.strip()}"
    if isinstance(did, str) and did.strip():
        return did.strip()
    return "unknown"


def _render_markdown_frontmatter(payload: dict[str, Any]) -> str:
    import yaml

    data = {key: value for key, value in payload.items() if value is not None}
    frontmatter = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ).strip()
    return f"---\n{frontmatter}\n---"


def _render_links_section(
    post: dict[str, Any],
    *,
    settings: AtprotoSettings,
) -> str | None:
    blocks: list[str] = []
    seen: set[str] = set()

    external = _extract_external_link(post)
    if external:
        uri = external["uri"]
        seen.add(uri)
        lines = [f"- {uri}"]
        if external.get("title"):
            lines.append(f"  - title: {external['title']}")
        if external.get("description"):
            _append_field_line(
                lines,
                "description",
                str(external["description"]),
                literal=True,
            )
        thumb = external.get("thumb")
        if isinstance(thumb, str) and thumb:
            lines.append(f"  - preview-image: {thumb}")
            preview_media = {
                "kind": "image",
                "url": thumb,
                "alt": external.get("title"),
                "mime": None,
                "width": None,
                "height": None,
                "cache_identity": f"atproto:url:{thumb}",
                "is_embed_media": True,
            }
            generated = _describe_media(preview_media, settings=settings)
            if generated:
                _append_field_line(lines, "llm-description", generated, literal=True)
        blocks.append("\n".join(lines))

    for facet_uri in _facet_link_uris(post):
        if facet_uri in seen:
            continue
        seen.add(facet_uri)
        blocks.append(f"- {facet_uri}")

    joined = "\n\n".join(blocks).strip()
    return joined or None


def _post_metrics(post: dict[str, Any]) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for key, label in (
        ("replyCount", "replies"),
        ("repostCount", "reposts"),
        ("likeCount", "likes"),
        ("quoteCount", "quotes"),
    ):
        value = post.get(key)
        metrics[label] = value if isinstance(value, int) else 0
    return metrics


def _collect_quote_post(
    post: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    quote_view = _extract_quote_view(post)
    if not isinstance(quote_view, dict):
        return None, None
    quote_post = _quote_to_post(quote_view)
    quote_uri = quote_view.get("uri")
    if not isinstance(quote_uri, str) or not quote_uri:
        quote_uri = (
            str(quote_post.get("uri") or "") if isinstance(quote_post, dict) else ""
        )
    return (quote_uri or None), quote_post


def _fetch_direct_reply_posts(
    uri: str,
    *,
    client: Any,
    max_replies: int,
) -> list[dict[str, Any]]:
    if max_replies <= 0:
        return []
    try:
        response = _client_call(
            "feed.getPostThread",
            client.app.bsky.feed.getPostThread,
            uri=uri,
            depth=1,
            parentHeight=0,
        )
    except Exception as exc:
        _log(f"  failed to fetch replies for {uri}: {exc}")
        return []
    thread = response.get("thread")
    if not isinstance(thread, dict):
        return []
    replies = thread.get("replies")
    if not isinstance(replies, list):
        return []

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reply in replies:
        if len(output) >= max_replies:
            break
        if not isinstance(reply, dict):
            continue
        ntype = str(reply.get("$type") or "")
        if "blocked" in ntype.lower() or "notfound" in ntype.lower():
            continue
        post = reply.get("post")
        if not isinstance(post, dict):
            continue
        reply_uri = post.get("uri")
        if not isinstance(reply_uri, str) or not reply_uri:
            continue
        if reply_uri in seen:
            continue
        seen.add(reply_uri)
        output.append(post)
    return output


def _render_post_document(
    *,
    post: dict[str, Any],
    source_url: str,
    settings: AtprotoSettings,
    quote_uris: list[str],
    reply_uris: list[str],
) -> str:
    record = post.get("record")
    if not isinstance(record, dict):
        record = post.get("value") if isinstance(post.get("value"), dict) else {}
    uri = str(post.get("uri") or "")
    author = post.get("author") if isinstance(post.get("author"), dict) else {}
    created_at = None
    for key in ("createdAt", "indexedAt"):
        value = record.get(key) if key in record else post.get(key)
        if isinstance(value, str) and value:
            created_at = value
            break
    text = str(record.get("text") or "")
    facets = record.get("facets") if isinstance(record.get("facets"), list) else []
    rendered_text = _render_rich_text(text, facets)
    web_url = _uri_to_bsky_url(uri, handle=author.get("handle"))
    display_name = author.get("displayName") if isinstance(author, dict) else None
    if not isinstance(display_name, str) or not display_name.strip():
        display_name = (
            author.get("handle")
            if isinstance(author, dict) and isinstance(author.get("handle"), str)
            else None
        )
    metadata = {
        "uri": uri or None,
        "url": web_url,
        "author_name": display_name.strip()
        if isinstance(display_name, str) and display_name.strip()
        else None,
        "created_at": created_at,
        "source_url": source_url,
        "metrics": _post_metrics(post),
    }
    body = rendered_text or "(empty)"
    sections: list[str] = []
    if quote_uris:
        sections.append("## Quote\n\n" + "\n".join(f"- {item}" for item in quote_uris))
    if reply_uris:
        sections.append(
            "## Replies\n\n" + "\n".join(f"- {item}" for item in reply_uris)
        )
    links_section = _render_links_section(post, settings=settings)
    if links_section:
        sections.append("## Links\n\n" + links_section)
    media_items = _collect_media_entries(post)
    media_section = _render_media_section(media_items, settings).strip()
    if media_section:
        sections.append("## Media\n\n" + media_section)

    lines = [
        _render_markdown_frontmatter(metadata),
        body,
    ]
    if sections:
        lines.extend(["***", "\n\n".join(sections)])
    return "\n\n".join(part.strip() for part in lines if part.strip())


def _post_source_timestamps(post: dict[str, Any]) -> tuple[str | None, str | None]:
    record = post.get("record")
    if not isinstance(record, dict):
        record = post.get("value") if isinstance(post.get("value"), dict) else {}
    created = record.get("createdAt")
    source_created = created if isinstance(created, str) else None
    modified = post.get("indexedAt")
    source_modified = modified if isinstance(modified, str) else None
    return source_created, source_modified


def _post_document_label(root: str, index: int, post: dict[str, Any]) -> str:
    return (
        f"{root}/posts/{index:03d}-"
        f"{_safe_slug(_post_label(post, index), f'post-{index:03d}')}"
    )


def _collect_post_documents(
    *,
    post: dict[str, Any],
    source_url: str,
    kind: str,
    root: str,
    settings: AtprotoSettings,
    client: Any,
    index_counter: list[int],
    emitted_uris: set[str],
    quote_depth: int,
    quote_seen: set[str],
    expand_replies: bool,
) -> list[AtprotoDocument]:
    uri = post.get("uri")
    if not isinstance(uri, str) or not uri:
        return []
    if uri in emitted_uris:
        return []
    emitted_uris.add(uri)

    quote_uris: list[str] = []
    nested_docs: list[AtprotoDocument] = []
    if quote_depth > 0:
        quote_uri, quote_post = _collect_quote_post(post)
        if quote_uri:
            quote_uris.append(quote_uri)
        if (
            quote_uri
            and isinstance(quote_post, dict)
            and quote_uri not in quote_seen
            and quote_uri not in emitted_uris
        ):
            nested_docs.extend(
                _collect_post_documents(
                    post=quote_post,
                    source_url=source_url,
                    kind=kind,
                    root=root,
                    settings=settings,
                    client=client,
                    index_counter=index_counter,
                    emitted_uris=emitted_uris,
                    quote_depth=quote_depth - 1,
                    quote_seen={*quote_seen, quote_uri},
                    expand_replies=False,
                )
            )

    reply_posts: list[dict[str, Any]] = []
    reply_uris: list[str] = []
    if expand_replies and settings.max_replies > 0:
        reply_posts = _fetch_direct_reply_posts(
            uri,
            client=client,
            max_replies=settings.max_replies,
        )
        reply_uris = [
            str(item.get("uri"))
            for item in reply_posts
            if isinstance(item.get("uri"), str) and item.get("uri")
        ]

    index_counter[0] += 1
    label = _post_document_label(root, index_counter[0], post)
    rendered = _render_post_document(
        post=post,
        source_url=source_url,
        settings=settings,
        quote_uris=quote_uris,
        reply_uris=reply_uris,
    )
    source_created, source_modified = _post_source_timestamps(post)
    documents = [
        AtprotoDocument(
            source_url=source_url,
            kind=kind,
            uri=uri,
            label=label,
            trace_path=label,
            context_subpath=f"{label}.md",
            rendered=rendered,
            source_created=source_created,
            source_modified=source_modified,
        )
    ]
    documents.extend(nested_docs)

    for reply_post in reply_posts:
        reply_uri = reply_post.get("uri")
        if not isinstance(reply_uri, str) or not reply_uri:
            continue
        if reply_uri in emitted_uris:
            continue
        documents.extend(
            _collect_post_documents(
                post=reply_post,
                source_url=source_url,
                kind=kind,
                root=root,
                settings=settings,
                client=client,
                index_counter=index_counter,
                emitted_uris=emitted_uris,
                quote_depth=settings.reply_quote_depth,
                quote_seen={reply_uri},
                expand_replies=False,
            )
        )
    return documents


def _thread_posts(
    thread: Any, *, include_replies: bool
) -> tuple[list[tuple[dict[str, Any], list[str]]], dict[str, int]]:
    seen: set[str] = set()
    output: list[tuple[dict[str, Any], list[str]]] = []
    stats = {"blocked": 0, "not_found": 0}

    def walk_parent(node: Any) -> None:
        if not isinstance(node, dict):
            return
        ntype = str(node.get("$type") or "")
        if "blocked" in ntype.lower():
            stats["blocked"] += 1
            return
        if "notfound" in ntype.lower():
            stats["not_found"] += 1
            return
        post = node.get("post")
        if isinstance(post, dict):
            uri = str(post.get("uri") or "")
            if uri and uri not in seen:
                seen.add(uri)
                output.append((post, ["parent"]))
            parent = node.get("parent")
            if isinstance(parent, dict):
                walk_parent(parent)

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        ntype = str(node.get("$type") or "")
        if "blocked" in ntype.lower():
            stats["blocked"] += 1
            return
        if "notfound" in ntype.lower():
            stats["not_found"] += 1
            return
        post = node.get("post")
        if not isinstance(post, dict):
            return
        uri = str(post.get("uri") or "")
        if uri and uri not in seen:
            seen.add(uri)
            output.append((post, ["thread"]))
        parent = node.get("parent")
        if isinstance(parent, dict):
            walk_parent(parent)
        if include_replies:
            replies = node.get("replies")
            if isinstance(replies, list):
                for reply in replies:
                    if isinstance(reply, dict):
                        walk(reply)

    walk(thread)
    return output, stats


def _summary_document(
    *,
    source_url: str,
    kind: str,
    uri: str,
    root: str,
    fields: list[tuple[str, Any]],
    source_created: str | None = None,
    source_modified: str | None = None,
) -> AtprotoDocument:
    label = f"{root}/index"
    lines: list[str] = []
    for key, value in fields:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        lines.append(f"{key}: {value}")
    rendered = "\n".join(lines).strip()
    return AtprotoDocument(
        source_url=source_url,
        kind=kind,
        uri=uri,
        label=label,
        trace_path=label,
        context_subpath=f"{label}.md",
        rendered=rendered,
        source_created=source_created,
        source_modified=source_modified,
    )


def _post_documents_from_feed(
    *,
    source_url: str,
    kind: str,
    root: str,
    entries: list[Any],
    settings: AtprotoSettings,
    client: Any,
) -> list[AtprotoDocument]:
    documents: list[AtprotoDocument] = []
    emitted_uris: set[str] = set()
    index_counter = [0]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        post = entry.get("post") if isinstance(entry.get("post"), dict) else None
        if post is None:
            continue
        documents.extend(
            _collect_post_documents(
                post=post,
                source_url=source_url,
                kind=kind,
                root=root,
                settings=settings,
                client=client,
                index_counter=index_counter,
                emitted_uris=emitted_uris,
                quote_depth=settings.quote_depth,
                quote_seen={str(post.get("uri") or "")},
                expand_replies=True,
            )
        )
    return documents


def _resolve_post_documents(
    target: AtprotoTarget,
    *,
    source_url: str,
    client: Any,
    settings: AtprotoSettings,
) -> list[AtprotoDocument]:
    assert target.uri is not None
    response = _client_call(
        "feed.getPostThread",
        client.app.bsky.feed.getPostThread,
        uri=target.uri,
        depth=settings.thread_depth,
        parentHeight=settings.thread_parent_height,
    )
    thread = response.get("thread")
    posts, _ = _thread_posts(thread, include_replies=settings.include_replies)
    repo, _collection, _rkey = _parse_at_uri(target.uri)
    root_actor = target.actor or repo or "profile"
    root = "atproto/profile/" + _safe_slug(str(root_actor), "profile")
    documents: list[AtprotoDocument] = []
    emitted_uris: set[str] = set()
    index_counter = [0]
    for post, _context in posts:
        documents.extend(
            _collect_post_documents(
                post=post,
                source_url=source_url,
                kind="post",
                root=root,
                settings=settings,
                client=client,
                index_counter=index_counter,
                emitted_uris=emitted_uris,
                quote_depth=settings.quote_depth,
                quote_seen={str(post.get("uri") or "")},
                expand_replies=True,
            )
        )
    return documents


def _resolve_profile_documents(
    target: AtprotoTarget,
    *,
    source_url: str,
    client: Any,
    settings: AtprotoSettings,
) -> list[AtprotoDocument]:
    assert target.repo is not None
    profile = _client_call(
        "actor.getProfile",
        client.app.bsky.actor.getProfile,
        actor=target.repo,
    )
    feed_resp = _client_call(
        "feed.getAuthorFeed",
        client.app.bsky.feed.getAuthorFeed,
        actor=target.repo,
        limit=settings.max_items,
    )
    feed_entries = feed_resp.get("feed")
    if not isinstance(feed_entries, list):
        feed_entries = []
    handle = profile.get("handle") if isinstance(profile, dict) else None
    display_name = profile.get("displayName") if isinstance(profile, dict) else None
    root = "atproto/profile/" + _safe_slug(str(handle or target.repo), "profile")
    profile_uri = f"at://{target.repo}"
    profile_url = (
        f"https://bsky.app/profile/{handle}"
        if isinstance(handle, str) and handle.strip()
        else _uri_to_bsky_url(profile_uri)
    )
    documents = _post_documents_from_feed(
        source_url=source_url,
        kind="profile",
        root=root,
        entries=feed_entries,
        settings=settings,
        client=client,
    )
    summary = _summary_document(
        source_url=source_url,
        kind="profile",
        uri=profile_uri,
        root=root,
        fields=[
            ("uri", profile_uri),
            ("url", profile_url),
            (
                "display_name",
                display_name.strip() if isinstance(display_name, str) else None,
            ),
            (
                "followers",
                profile.get("followersCount") if isinstance(profile, dict) else None,
            ),
            (
                "following",
                profile.get("followsCount") if isinstance(profile, dict) else None,
            ),
            (
                "post_count",
                profile.get("postsCount") if isinstance(profile, dict) else None,
            ),
        ],
        source_modified=profile.get("indexedAt")
        if isinstance(profile, dict) and isinstance(profile.get("indexedAt"), str)
        else None,
    )
    return [summary, *documents]


def _resolve_feed_documents(
    target: AtprotoTarget,
    *,
    source_url: str,
    client: Any,
    settings: AtprotoSettings,
) -> list[AtprotoDocument]:
    assert target.uri is not None
    generator_resp = _client_call(
        "feed.getFeedGenerator",
        client.app.bsky.feed.getFeedGenerator,
        feed=target.uri,
    )
    feed_resp = _client_call(
        "feed.getFeed",
        client.app.bsky.feed.getFeed,
        feed=target.uri,
        limit=settings.max_items,
    )
    entries = feed_resp.get("feed")
    if not isinstance(entries, list):
        entries = []
    view = generator_resp.get("view") if isinstance(generator_resp, dict) else {}
    display_name = view.get("displayName") if isinstance(view, dict) else None
    root = "atproto/feed/" + _safe_slug(str(target.rkey or target.uri), "feed")
    creator = view.get("creator") if isinstance(view, dict) else None
    documents = _post_documents_from_feed(
        source_url=source_url,
        kind="feed",
        root=root,
        entries=entries,
        settings=settings,
        client=client,
    )
    summary = _summary_document(
        source_url=source_url,
        kind="feed",
        uri=target.uri,
        root=root,
        fields=[
            ("uri", target.uri),
            (
                "url",
                source_url
                if is_bsky_app_url(source_url)
                else _uri_to_bsky_url(target.uri),
            ),
            (
                "display_name",
                display_name.strip() if isinstance(display_name, str) else None,
            ),
            ("creator", _author_name(creator) if isinstance(creator, dict) else None),
            ("item_count", len(documents)),
        ],
    )
    return [summary, *documents]


def _resolve_list_documents(
    target: AtprotoTarget,
    *,
    source_url: str,
    client: Any,
    settings: AtprotoSettings,
) -> list[AtprotoDocument]:
    assert target.uri is not None
    list_resp = _client_call(
        "graph.getList",
        client.app.bsky.graph.getList,
        list=target.uri,
        limit=min(settings.max_items, 100),
    )
    feed_resp = _client_call(
        "feed.getListFeed",
        client.app.bsky.feed.getListFeed,
        list=target.uri,
        limit=settings.max_items,
    )
    entries = feed_resp.get("feed")
    if not isinstance(entries, list):
        entries = []
    list_view = list_resp.get("list") if isinstance(list_resp, dict) else {}
    root = "atproto/list/" + _safe_slug(str(target.rkey or target.uri), "list")
    documents = _post_documents_from_feed(
        source_url=source_url,
        kind="list",
        root=root,
        entries=entries,
        settings=settings,
        client=client,
    )
    name = list_view.get("name") if isinstance(list_view, dict) else None
    description = list_view.get("description") if isinstance(list_view, dict) else None
    creator = list_view.get("creator") if isinstance(list_view, dict) else None
    list_item_count = (
        list_view.get("listItemCount") if isinstance(list_view, dict) else None
    )
    summary = _summary_document(
        source_url=source_url,
        kind="list",
        uri=target.uri,
        root=root,
        fields=[
            ("uri", target.uri),
            (
                "url",
                source_url
                if is_bsky_app_url(source_url)
                else _uri_to_bsky_url(target.uri),
            ),
            ("name", name.strip() if isinstance(name, str) else None),
            (
                "description",
                description.strip() if isinstance(description, str) else None,
            ),
            ("creator", _author_name(creator) if isinstance(creator, dict) else None),
            (
                "member_count",
                list_item_count if isinstance(list_item_count, int) else None,
            ),
            ("item_count", len(documents)),
        ],
    )
    return [summary, *documents]


def _resolve_starter_pack_documents(
    target: AtprotoTarget,
    *,
    source_url: str,
    client: Any,
    settings: AtprotoSettings,
) -> list[AtprotoDocument]:
    assert target.uri is not None
    pack_resp = _client_call(
        "graph.getStarterPack",
        client.app.bsky.graph.getStarterPack,
        starterPack=target.uri,
    )
    pack = pack_resp.get("starterPack") if isinstance(pack_resp, dict) else {}
    if not isinstance(pack, dict):
        raise ValueError(f"Unexpected starter pack payload for {target.uri}")
    list_view = pack.get("list") if isinstance(pack.get("list"), dict) else {}
    list_uri = list_view.get("uri") if isinstance(list_view.get("uri"), str) else None
    entries: list[Any] = []
    if list_uri:
        try:
            list_feed_resp = _client_call(
                "feed.getListFeed",
                client.app.bsky.feed.getListFeed,
                list=list_uri,
                limit=settings.max_items,
            )
            fetched = list_feed_resp.get("feed")
            if isinstance(fetched, list):
                entries = fetched
        except Exception as exc:
            _log(f"  starter-pack list feed fallback failed: {exc}")
    root = "atproto/starter-pack/" + _safe_slug(str(target.rkey or target.uri), "pack")
    record = pack.get("record") if isinstance(pack.get("record"), dict) else {}
    pack_name = record.get("name") if isinstance(record, dict) else None
    description = record.get("description") if isinstance(record, dict) else None
    creator = pack.get("creator") if isinstance(pack.get("creator"), dict) else None
    list_name = list_view.get("name") if isinstance(list_view, dict) else None
    feeds = pack.get("feeds")
    documents = _post_documents_from_feed(
        source_url=source_url,
        kind="starter-pack",
        root=root,
        entries=entries,
        settings=settings,
        client=client,
    )
    summary = _summary_document(
        source_url=source_url,
        kind="starter-pack",
        uri=target.uri,
        root=root,
        fields=[
            ("uri", target.uri),
            (
                "url",
                source_url
                if is_bsky_app_url(source_url)
                else _uri_to_bsky_url(target.uri),
            ),
            ("name", pack_name.strip() if isinstance(pack_name, str) else None),
            (
                "description",
                description.strip() if isinstance(description, str) else None,
            ),
            ("creator", _author_name(creator) if isinstance(creator, dict) else None),
            ("list_uri", list_uri),
            ("list_name", list_name.strip() if isinstance(list_name, str) else None),
            ("embedded_feeds", len(feeds) if isinstance(feeds, list) else None),
            ("item_count", len(documents)),
        ],
    )
    return [summary, *documents]


def _resolve_labeler_documents(
    target: AtprotoTarget,
    *,
    source_url: str,
    client: Any,
    settings: AtprotoSettings,
) -> list[AtprotoDocument]:
    assert target.repo is not None
    services_resp = _client_call(
        "labeler.getServices",
        client.app.bsky.labeler.getServices,
        dids=[target.repo],
        detailed=True,
    )
    services = services_resp.get("views")
    if not isinstance(services, list):
        services = []
    record_payload = None
    if target.collection and target.rkey:
        try:
            record_payload = _client_call(
                "repo.getRecord",
                client.com.atproto.repo.getRecord,
                repo=target.repo,
                collection=target.collection,
                rkey=target.rkey,
            )
        except Exception:
            record_payload = None

    root = "atproto/labeler/" + _safe_slug(str(target.rkey or target.repo), "labeler")
    fields: list[tuple[str, Any]] = [
        ("uri", target.uri or f"at://{target.repo}"),
        (
            "url",
            source_url
            if is_bsky_app_url(source_url)
            else _uri_to_bsky_url(target.uri or f"at://{target.repo}"),
        ),
        ("did", target.repo),
        ("service_count", len(services)),
    ]
    for index, service in enumerate(services, start=1):
        if not isinstance(service, dict):
            continue
        fields.append((f"service_{index}_uri", service.get("uri")))
    if record_payload is not None:
        fields.append(("record_fetched", True))

    return [
        _summary_document(
            source_url=source_url,
            kind="labeler",
            uri=target.uri or f"at://{target.repo}",
            root=root,
            fields=fields,
        )
    ]


def _resolve_record_documents(
    target: AtprotoTarget,
    *,
    source_url: str,
    client: Any,
    settings: AtprotoSettings,
) -> list[AtprotoDocument]:
    if target.collection == "app.bsky.feed.post" and target.rkey and target.uri:
        return _resolve_post_documents(
            target,
            source_url=source_url,
            client=client,
            settings=settings,
        )
    if not target.repo or not target.collection:
        raise ValueError(f"Incomplete AT URI target: {target.original}")

    root = "atproto/record/" + _safe_slug(
        f"{target.collection}-{target.rkey or target.repo}",
        "record",
    )
    if target.rkey:
        payload = _client_call(
            "repo.getRecord",
            client.com.atproto.repo.getRecord,
            repo=target.repo,
            collection=target.collection,
            rkey=target.rkey,
        )
        label = f"{root}/record"
        rendered = _render_json_block(payload)
        return [
            AtprotoDocument(
                source_url=source_url,
                kind="record",
                uri=target.uri or "",
                label=label,
                trace_path=label,
                context_subpath=f"{label}.md",
                rendered=rendered,
            )
        ]

    payload = _client_call(
        "repo.listRecords",
        client.com.atproto.repo.listRecords,
        repo=target.repo,
        collection=target.collection,
        limit=settings.max_items,
    )
    records = payload.get("records") if isinstance(payload, dict) else []
    if not isinstance(records, list):
        records = []
    documents = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        uri = str(record.get("uri") or target.uri or "")
        label = f"{root}/records/{index:03d}"
        documents.append(
            AtprotoDocument(
                source_url=source_url,
                kind="record",
                uri=uri,
                label=label,
                trace_path=label,
                context_subpath=f"{label}.md",
                rendered=_render_json_block(record),
            )
        )
    summary = _summary_document(
        source_url=source_url,
        kind="record",
        uri=target.uri or "",
        root=root,
        fields=[
            ("uri", target.uri),
            ("source_url", source_url),
            ("collection", target.collection),
            ("record_count", len(documents)),
        ],
    )
    return [summary, *documents]


def resolve_atproto_url(
    url: str,
    *,
    settings: AtprotoSettings | None = None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> list[AtprotoDocument]:
    from ..cache.atproto import get_cached_api_json, store_api_json
    from ..runtime import get_refresh_media

    warmup_atproto_network_stack()
    _log(f"Resolving atproto target: {url}")

    parsed = parse_atproto_target(url)
    if parsed is None:
        raise ValueError(f"Not an atproto target: {url}")
    effective_settings = (
        settings if settings is not None else _atproto_settings_from_env()
    )
    client_ctx = _build_client_context()

    refresh_resolution_cache = refresh_cache or get_refresh_media()
    cache_identity = _resolution_cache_identity(
        url,
        effective_settings,
        client_ctx.auth_mode,
    )
    if use_cache and not refresh_resolution_cache:
        cached_payload = get_cached_api_json(cache_identity, ttl=cache_ttl)
        cached_docs = _documents_from_cached_payload(cached_payload)
        if cached_docs is not None:
            _log(f"  atproto resolution cache hit: {url}")
            return cached_docs

    target = _canonicalize_target(
        parsed,
        client=client_ctx.client,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
    if target.kind == "post":
        documents = _resolve_post_documents(
            target,
            source_url=url,
            client=client_ctx.client,
            settings=effective_settings,
        )
    elif target.kind == "profile":
        documents = _resolve_profile_documents(
            target,
            source_url=url,
            client=client_ctx.client,
            settings=effective_settings,
        )
    elif target.kind == "feed":
        documents = _resolve_feed_documents(
            target,
            source_url=url,
            client=client_ctx.client,
            settings=effective_settings,
        )
    elif target.kind == "list":
        documents = _resolve_list_documents(
            target,
            source_url=url,
            client=client_ctx.client,
            settings=effective_settings,
        )
    elif target.kind == "starter-pack":
        documents = _resolve_starter_pack_documents(
            target,
            source_url=url,
            client=client_ctx.client,
            settings=effective_settings,
        )
    elif target.kind == "labeler":
        documents = _resolve_labeler_documents(
            target,
            source_url=url,
            client=client_ctx.client,
            settings=effective_settings,
        )
    else:
        documents = _resolve_record_documents(
            target,
            source_url=url,
            client=client_ctx.client,
            settings=effective_settings,
        )

    if use_cache:
        store_api_json(cache_identity, [asdict(document) for document in documents])
    return documents


@dataclass
class AtprotoReference:
    url: str
    document: AtprotoDocument
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

    def get_contents(self) -> str:
        return self.output
