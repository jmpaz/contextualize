from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from urllib.parse import urlparse, urlsplit, urlunsplit

from ..render.text import process_text
from ..utils import count_tokens
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

_JINA_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_MARKDOWN_CONTENT_TYPES = frozenset({"text/markdown", "text/x-markdown"})
_JINA_ENDPOINT = "https://r.jina.ai/"


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
    _bypass_jina: bool = field(default=False, init=False, repr=False)
    _gist_filename: str | None = field(default=None, init=False, repr=False)
    _content_type: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.url.startswith(RAW_PREFIX):
            self.url = self.url[len(RAW_PREFIX) :]
            self._bypass_jina = True
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

    def _fetch_via_jina(self) -> str:
        import requests

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Md-Heading-Style": "atx",
            "X-Md-Bullet-List-Marker": "-",
        }
        api_key = os.environ.get("JINA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        r = requests.post(
            _JINA_ENDPOINT,
            json={"url": self.url},
            headers=headers,
            timeout=30,
        )
        if r.status_code == 429:
            raise ValueError(
                f"Jina Reader rate limit exceeded for {self.url}. "
                "Set JINA_API_KEY for higher limits: https://jina.ai/?sui=apikey"
            )
        r.raise_for_status()

        data = r.json()
        if data.get("code") != 200:
            raise ValueError(f"Jina Reader error for {self.url}: {data}")

        return data.get("data", {}).get("content", "")

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
                headers={"User-Agent": "contextualize"},
                allow_redirects=True,
            )
            head_r.raise_for_status()
            head_content_type = strip_content_type(
                head_r.headers.get("Content-Type", "")
            )
        except Exception:
            head_content_type = ""

        self._content_type = head_content_type
        use_jina = not self._bypass_jina and head_content_type in _JINA_HTML_TYPES

        if use_jina:
            md_content, got_markdown = self._try_fetch_markdown()
            if got_markdown:
                text = md_content
            else:
                candidate_url = self._markdown_url_candidate()
                if candidate_url:
                    md_content, got_markdown = self._try_fetch_markdown_url(
                        candidate_url
                    )
                text = md_content if got_markdown else self._fetch_via_jina()
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
        r.raise_for_status()
        content_type = strip_content_type(r.headers.get("Content-Type", ""))
        self._content_type = content_type
        suffix = infer_url_suffix(self.url, dict(r.headers))
        if (suffix and suffix in DISALLOWED_EXTENSIONS) or (
            content_type in DISALLOWED_CONTENT_TYPES
        ):
            raise ValueError(f"Unsupported file type: {self.url}")

        data = r.content
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
