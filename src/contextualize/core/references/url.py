import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..render import process_text
from ..utils import count_tokens
from .helpers import (
    DISALLOWED_CONTENT_TYPES,
    DISALLOWED_EXTENSIONS,
    MARKITDOWN_PREFERRED_EXTENSIONS,
    infer_url_suffix,
    looks_like_text_content_type,
    strip_content_type,
)


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
            return self.url
        if self.label == "name":
            return os.path.basename(path)
        if self.label == "ext":
            return os.path.splitext(path)[1]
        return self.label

    def _get_contents(self) -> str:
        import json

        import requests

        r = requests.get(self.url, timeout=30, headers={"User-Agent": "contextualize"})
        r.raise_for_status()
        content_type = strip_content_type(r.headers.get("Content-Type", ""))
        suffix = infer_url_suffix(self.url, dict(r.headers))
        if (suffix and suffix in DISALLOWED_EXTENSIONS) or (
            content_type in DISALLOWED_CONTENT_TYPES
        ):
            raise ValueError(f"Unsupported file type: {self.url}")

        data = r.content
        prefer_markitdown = bool(suffix and suffix in MARKITDOWN_PREFERRED_EXTENSIONS)
        is_text = looks_like_text_content_type(content_type)

        if prefer_markitdown:
            from ..markitdown_adapter import convert_response_to_markdown

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
                from ..markitdown_adapter import convert_response_to_markdown

                text = convert_response_to_markdown(r).markdown
            self.original_file_content = text
        if self.inject:
            from ..links import inject_content_in_text

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
    )
