from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, TypedDict
from urllib.parse import urlparse, urlsplit, urlunsplit

from ..render.text import process_text
from ..utils import count_tokens
from .audio_transcription import is_audio_suffix, transcribe_audio_bytes
from .helpers import (
    DISALLOWED_CONTENT_TYPES,
    DISALLOWED_EXTENSIONS,
    MARKITDOWN_PREFERRED_EXTENSIONS,
    RAW_PREFIX,
    fetch_gist_files,
    infer_url_suffix,
    looks_like_text_content_type,
    parse_gist_url,
    strip_content_type,
)

_HTML_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_MARKDOWN_CONTENT_TYPES = frozenset({"text/markdown", "text/x-markdown"})
_MARKDOWN_NEW_ENDPOINT = "https://markdown.new/"
_JINA_ENDPOINT = "https://r.jina.ai/"
_MARKDOWN_ESCAPED_PUNCT_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|])")
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_MARKDOWN_LIST_RE = re.compile(r"^\s{0,3}(?:[-*+]\s+\S|\d{1,3}\.\s+\S)")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]\n]{1,200}\]\([^\s)]+[^)]*\)")
_MARKDOWN_BLOB_TOKEN_RE = re.compile(
    r"(?:\[[^\]\n]{1,200}\]\([^\s)]+[^)]*\)|\*\*[^*\n]+\*\*|`[^`\n]+`|#{1,6}\s+\S)"
)
_HTML_DOC_ROOT_RE = re.compile(r"^\s*<!doctype\s+html|^\s*<html\b", re.IGNORECASE)
_HTML_DOC_MARKERS = (
    "<html",
    "<head",
    "<body",
    "<meta ",
    "<title",
    "<script",
    "</html>",
)


class MarkdownQualitySignals(TypedDict):
    char_count: int
    line_count: int
    escaped_punct_count: int
    escaped_punct_ratio: float
    heading_count: int
    list_marker_count: int
    link_count: int
    avg_line_length: float
    newline_density: float
    has_single_line_blob: bool


class MarkdownQualityAssessment(TypedDict):
    requires_fallback: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ConverterCandidate:
    provider_name: str
    converted_text: str
    quality_score: float
    requires_fallback: bool
    fallback_reasons: tuple[str, ...]


def _log(message: str) -> None:
    from ..runtime import get_verbose_logging

    if get_verbose_logging():
        print(f"[url] {message}", file=sys.stderr, flush=True)


def _non_fenced_lines(text: str) -> list[str]:
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        lines.append(line)
    return lines


def _looks_like_html(text: str) -> bool:
    snippet = text.lstrip()
    if not snippet:
        return False
    if _HTML_DOC_ROOT_RE.search(snippet):
        return True

    normalized = snippet[:4000].lower()
    marker_hits = sum(1 for marker in _HTML_DOC_MARKERS if marker in normalized)
    has_root_marker = "<html" in normalized and (
        "<head" in normalized or "<body" in normalized
    )
    return marker_hits >= 4 or has_root_marker


def _quality_signals(markdown: str) -> MarkdownQualitySignals:
    lines = _non_fenced_lines(markdown)
    non_fenced = "\n".join(lines)
    char_count = len(non_fenced)
    line_count = len(lines)
    escaped_punct_count = len(_MARKDOWN_ESCAPED_PUNCT_RE.findall(non_fenced))
    escaped_punct_ratio = (escaped_punct_count / char_count) if char_count > 0 else 0.0
    heading_count = sum(1 for line in lines if _MARKDOWN_HEADING_RE.match(line))
    list_marker_count = sum(1 for line in lines if _MARKDOWN_LIST_RE.match(line))
    link_count = len(_MARKDOWN_LINK_RE.findall(non_fenced))
    avg_line_length = (char_count / line_count) if line_count > 0 else 0.0
    newline_density = ((line_count - 1) / char_count) if char_count > 0 else 0.0
    blob_token_count = len(_MARKDOWN_BLOB_TOKEN_RE.findall(non_fenced))
    has_single_line_blob = (
        line_count <= 2
        and char_count >= 500
        and (
            blob_token_count >= 6
            or link_count >= 4
            or escaped_punct_count >= 8
            or list_marker_count >= 5
        )
    ) or (
        line_count <= 4
        and char_count >= 1400
        and avg_line_length >= 220
        and blob_token_count >= 8
    )
    return {
        "char_count": char_count,
        "line_count": line_count,
        "escaped_punct_count": escaped_punct_count,
        "escaped_punct_ratio": escaped_punct_ratio,
        "heading_count": heading_count,
        "list_marker_count": list_marker_count,
        "link_count": link_count,
        "avg_line_length": avg_line_length,
        "newline_density": newline_density,
        "has_single_line_blob": has_single_line_blob,
    }


def _assess_markdown_quality(
    markdown: str, *, signals: MarkdownQualitySignals | None = None
) -> MarkdownQualityAssessment:
    signal_set = signals or _quality_signals(markdown)
    reasons: list[str] = []
    if _looks_like_html(markdown):
        reasons.append("html_document")
    if signal_set["has_single_line_blob"]:
        reasons.append("single_line_blob")
    if (
        signal_set["escaped_punct_count"] >= 10
        and signal_set["escaped_punct_ratio"] >= 0.003
    ):
        reasons.append("overescaped_markdown")
    if (
        signal_set["line_count"] <= 4
        and signal_set["avg_line_length"] >= 200
        and (
            signal_set["heading_count"]
            + signal_set["list_marker_count"]
            + signal_set["link_count"]
        )
        >= 6
    ):
        reasons.append("dense_markdown_blob")
    return {
        "requires_fallback": bool(reasons),
        "reasons": tuple(reasons),
    }


def _score_markdown_quality(
    markdown: str, *, signals: MarkdownQualitySignals | None = None
) -> float:
    signal_set = signals or _quality_signals(markdown)
    if _looks_like_html(markdown):
        return -100.0
    score = 0.0
    score += min(signal_set["line_count"], 160) * 0.08
    score += min(signal_set["heading_count"], 20) * 1.2
    score += min(signal_set["list_marker_count"], 80) * 0.2
    score += min(signal_set["link_count"], 120) * 0.08
    score += min(signal_set["newline_density"] * 1000.0, 12.0)
    score -= signal_set["escaped_punct_ratio"] * 300.0
    if signal_set["avg_line_length"] > 140.0:
        score -= (signal_set["avg_line_length"] - 140.0) / 30.0
    if signal_set["has_single_line_blob"]:
        score -= 30.0
    return score


def _has_excessive_markdown_escapes(text: str) -> bool:
    signals = _quality_signals(text)
    return (
        signals["escaped_punct_count"] >= 8 and signals["escaped_punct_ratio"] >= 0.003
    )


def _normalize_markdown_converter_output(text: str) -> str:
    normalized = text.strip()
    if not normalized or not _has_excessive_markdown_escapes(normalized):
        return normalized

    out_lines: list[str] = []
    in_fence = False
    for line in normalized.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue
        out_lines.append(_MARKDOWN_ESCAPED_PUNCT_RE.sub(r"\1", line))
    return "\n".join(out_lines)


@dataclass
class URLReference:
    url: str
    format: str = "md"
    label: str = "relative"
    token_target: str = "cl100k_base"
    include_token_count: bool = False
    label_suffix: str | None = None
    inject: bool = False
    depth: int = 5
    trace_collector: list = None
    filename_override: str | None = None
    use_cache: bool = True
    cache_ttl: timedelta | None = None
    refresh_cache: bool = False
    _bypass_markdown_converter: bool = field(default=False, init=False, repr=False)
    _gist_filename: str | None = field(default=None, init=False, repr=False)
    _content_type: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.url.startswith(RAW_PREFIX):
            self.url = self.url[len(RAW_PREFIX) :]
            self._bypass_markdown_converter = True
        if self.filename_override:
            self._gist_filename = self.filename_override
        else:
            gist_id = parse_gist_url(self.url)
            if gist_id:
                gist_files = fetch_gist_files(gist_id)
                if gist_files:
                    self._gist_filename = gist_files[0][0]
        self.file_content = ""
        self.original_file_content = ""
        self.output = self._get_contents()

    @property
    def path(self) -> str:
        return self.url

    def read(self) -> str:
        return self.original_file_content

    def exists(self) -> bool:
        import requests

        try:
            r = requests.head(
                self.url, timeout=10, headers={"User-Agent": "contextualize"}
            )
            return r.status_code < 400
        except Exception:
            return False

    def token_count(self, encoding: str = "cl100k_base") -> int:
        return count_tokens(self.original_file_content, target=encoding)["count"]

    def get_label(self) -> str:
        path = urlparse(self.url).path
        if self.label == "relative":
            if self._gist_filename:
                return self._gist_filename
            return self.url
        if self.label == "name":
            if self._gist_filename:
                return self._gist_filename
            return os.path.basename(path)
        if self.label == "ext":
            if self._gist_filename:
                return os.path.splitext(self._gist_filename)[1]
            return os.path.splitext(path)[1]
        return self.label

    def _fetch_via_markdown_converter(self) -> str:
        import requests

        failures: list[str] = []
        candidates: list[ConverterCandidate] = []

        def run_markdown_new() -> str:
            response = requests.post(
                _MARKDOWN_NEW_ENDPOINT,
                json={"url": self.url},
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/markdown, text/plain;q=0.9, application/json;q=0.8",
                    "User-Agent": "contextualize",
                },
                timeout=30,
            )
            response.raise_for_status()
            content_type = strip_content_type(response.headers.get("Content-Type", ""))
            if content_type and "json" in content_type:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("JSON response was not an object")
                for key in ("markdown", "content", "data"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
                raise ValueError("JSON response did not include markdown text")

            text = response.text.strip()
            if not text:
                raise ValueError("empty markdown response")
            return text

        def run_jina() -> str:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Md-Heading-Style": "atx",
                "X-Md-Bullet-List-Marker": "-",
            }
            api_key = os.environ.get("JINA_API_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            response = requests.post(
                _JINA_ENDPOINT,
                json={"url": self.url},
                headers=headers,
                timeout=30,
            )
            if response.status_code == 429:
                raise ValueError(
                    "Jina Reader rate limit exceeded. "
                    "Set JINA_API_KEY for higher limits: https://jina.ai/?sui=apikey"
                )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 200:
                raise ValueError(f"Jina Reader error payload: {payload}")
            value = payload.get("data", {}).get("content", "")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("empty markdown response")
            return value

        providers: tuple[tuple[str, Callable[[], str]], ...] = (
            ("markdown.new", run_markdown_new),
            ("jina", run_jina),
        )
        for provider_name, provider in providers:
            try:
                converted = _normalize_markdown_converter_output(provider())
            except Exception as exc:
                failures.append(f"{provider_name}: {exc}")
                continue
            if not converted:
                failures.append(f"{provider_name}: empty markdown response")
                continue
            signals = _quality_signals(converted)
            quality_assessment = _assess_markdown_quality(converted, signals=signals)
            requires_fallback = quality_assessment["requires_fallback"]
            fallback_reasons = quality_assessment["reasons"]
            score = _score_markdown_quality(converted, signals=signals)
            candidates.append(
                ConverterCandidate(
                    provider_name=provider_name,
                    converted_text=converted,
                    quality_score=score,
                    requires_fallback=requires_fallback,
                    fallback_reasons=fallback_reasons,
                )
            )
            _log(
                "converter candidate "
                f"provider={provider_name} requires_fallback={requires_fallback} "
                f"reasons={','.join(fallback_reasons) or 'none'} score={score:.2f}"
            )

            if provider_name == "markdown.new":
                if not requires_fallback:
                    return converted
                _log(
                    "markdown.new output failed quality gate "
                    f"({', '.join(fallback_reasons)}); trying jina fallback"
                )
                continue

        if candidates:
            selected_candidate = max(candidates, key=lambda item: item.quality_score)
            _log(
                "selected converter candidate "
                f"provider={selected_candidate.provider_name} "
                f"requires_fallback={selected_candidate.requires_fallback} "
                f"score={selected_candidate.quality_score:.2f}"
            )
            return selected_candidate.converted_text

        failure_text = "; ".join(failures) if failures else "no providers configured"
        raise ValueError(f"Markdown conversion failed for {self.url}: {failure_text}")

    def _try_fetch_markdown(self) -> tuple[str, bool]:
        return self._try_fetch_markdown_url(self.url)

    def _try_fetch_markdown_url(self, url: str) -> tuple[str, bool]:
        import requests

        try:
            r = requests.get(
                url,
                timeout=30,
                headers={
                    "User-Agent": "contextualize",
                    "Accept": "text/markdown, text/x-markdown",
                    "Accept-Encoding": "identity",
                },
                allow_redirects=True,
            )
            r.raise_for_status()
            content_type = strip_content_type(r.headers.get("Content-Type", ""))
            if content_type in _MARKDOWN_CONTENT_TYPES:
                return r.text, True
        except Exception:
            pass
        return "", False

    def _markdown_url_candidate(self) -> str | None:
        split_url = urlsplit(self.url)
        path = split_url.path
        if not path or path.endswith(".md"):
            return None
        if path == "/":
            return None
        normalized_path = path[:-1] if path.endswith("/") else path
        if not normalized_path:
            return None
        candidate = urlunsplit(
            (
                split_url.scheme,
                split_url.netloc,
                f"{normalized_path}.md",
                split_url.query,
                split_url.fragment,
            )
        )
        if candidate == self.url:
            return None
        return candidate

    def _get_contents(self) -> str:
        if self.use_cache and not self.refresh_cache:
            from ..cache import get_cached

            cached = get_cached(self.url, self.cache_ttl)
            if cached is not None:
                self.original_file_content = cached
                self.file_content = cached
                text = cached
                if self.inject:
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

        text = self._fetch_content()

        if self.use_cache:
            from ..cache import store_cached

            store_cached(self.url, text, self._content_type)

        return process_text(
            text,
            format=self.format,
            label=self.get_label(),
            label_suffix=self.label_suffix,
            token_target=self.token_target,
            include_token_count=self.include_token_count,
        )

    def _fetch_content(self) -> str:
        import json

        import requests

        try:
            head_r = requests.head(
                self.url,
                timeout=10,
                headers={
                    "User-Agent": "contextualize",
                    "Accept": "text/markdown, text/x-markdown",
                },
                allow_redirects=True,
            )
            head_r.raise_for_status()
            head_content_type = strip_content_type(
                head_r.headers.get("Content-Type", "")
            )
        except Exception:
            head_content_type = ""

        self._content_type = head_content_type
        try_markdown = not self._bypass_markdown_converter and (
            head_content_type in _HTML_CONTENT_TYPES
            or head_content_type in _MARKDOWN_CONTENT_TYPES
        )

        if try_markdown:
            md_content, got_markdown = self._try_fetch_markdown()
            if not got_markdown:
                candidate_url = self._markdown_url_candidate()
                if candidate_url:
                    md_content, got_markdown = self._try_fetch_markdown_url(
                        candidate_url
                    )
            if got_markdown:
                text = md_content
            elif head_content_type in _HTML_CONTENT_TYPES:
                text = self._fetch_via_markdown_converter()
                self._content_type = "text/markdown"
            else:
                text = md_content
            self.original_file_content = text
            self.file_content = text
            if self.inject:
                from ..render.inject import inject_content_in_text

                text = inject_content_in_text(
                    text, self.depth, self.trace_collector, self.url
                )
                self.file_content = text
            return text

        r = requests.get(self.url, timeout=30, headers={"User-Agent": "contextualize"})
        try:
            r.raise_for_status()
        except requests.HTTPError:
            if not self._bypass_markdown_converter and r.status_code in {401, 403}:
                text = self._fetch_via_markdown_converter()
                self._content_type = "text/markdown"
                self.original_file_content = text
                self.file_content = text
                if self.inject:
                    from ..render.inject import inject_content_in_text

                    text = inject_content_in_text(
                        text, self.depth, self.trace_collector, self.url
                    )
                self.file_content = text
                return text
            raise
        content_type = strip_content_type(r.headers.get("Content-Type", ""))
        self._content_type = content_type
        if content_type in _HTML_CONTENT_TYPES and not self._bypass_markdown_converter:
            text = self._fetch_via_markdown_converter()
            self._content_type = "text/markdown"
            self.original_file_content = text
            self.file_content = text
            if self.inject:
                from ..render.inject import inject_content_in_text

                text = inject_content_in_text(
                    text, self.depth, self.trace_collector, self.url
                )
            self.file_content = text
            return text
        suffix = infer_url_suffix(self.url, dict(r.headers))
        if (suffix and suffix in DISALLOWED_EXTENSIONS) or (
            content_type in DISALLOWED_CONTENT_TYPES
        ):
            raise ValueError(f"Unsupported file type: {self.url}")

        data = r.content
        is_audio = bool(suffix and is_audio_suffix(suffix)) or content_type.startswith(
            "audio/"
        )
        if is_audio:
            audio_name = os.path.basename(urlparse(self.url).path) or "audio"
            if "." not in audio_name and suffix:
                audio_name = f"{audio_name}{suffix}"
            elif "." not in audio_name:
                audio_name = f"{audio_name}.mp3"
            try:
                text = transcribe_audio_bytes(
                    data,
                    filename=audio_name,
                    content_type=content_type or None,
                )
            except RuntimeError as exc:
                raise ValueError(
                    f"Audio transcription failed for {self.url}: {exc}"
                ) from exc
            self.original_file_content = text
            self.file_content = text
            if self.inject:
                from ..render.inject import inject_content_in_text

                text = inject_content_in_text(
                    text, self.depth, self.trace_collector, self.url
                )
            self.file_content = text
            return text
        prefer_markitdown = bool(suffix and suffix in MARKITDOWN_PREFERRED_EXTENSIONS)
        is_text = looks_like_text_content_type(content_type)

        if prefer_markitdown:
            from ..render.markitdown import convert_response_to_markdown

            text = convert_response_to_markdown(r).markdown
            self.original_file_content = text
        elif is_text:
            text = r.text
            self.original_file_content = text
            if "json" in content_type:
                try:
                    text = json.dumps(r.json(), indent=2)
                except Exception:
                    pass
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                from ..render.markitdown import convert_response_to_markdown

                text = convert_response_to_markdown(r).markdown
            self.original_file_content = text
        if self.inject:
            from ..render.inject import inject_content_in_text

            text = inject_content_in_text(
                text, self.depth, self.trace_collector, self.url
            )
        self.file_content = text
        return text

    def get_contents(self) -> str:
        return self.output


def create_url_reference(
    url: str,
    format: str = "md",
    label: str = "relative",
    label_suffix: str | None = None,
    include_token_count: bool = False,
    token_target: str = "cl100k_base",
    inject: bool = False,
    depth: int = 5,
    trace_collector=None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> URLReference:
    return URLReference(
        url=url,
        format=format,
        label=label,
        label_suffix=label_suffix,
        include_token_count=include_token_count,
        token_target=token_target,
        inject=inject,
        depth=depth,
        trace_collector=trace_collector,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
