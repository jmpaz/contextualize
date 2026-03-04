from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..render.text import process_text
from ..utils import count_tokens


@dataclass(frozen=True)
class PluginResolvedDocument:
    source: str
    label: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginReference:
    source: str
    document: PluginResolvedDocument
    format: str = "md"
    label: str = "relative"
    token_target: str = "cl100k_base"
    include_token_count: bool = False
    label_suffix: str | None = None
    inject: bool = False
    depth: int = 5
    trace_collector: list | None = None

    def __post_init__(self) -> None:
        self.file_content = self.document.content
        self.original_file_content = self.document.content
        self.output = self._get_contents()

    @property
    def path(self) -> str:
        return self.source

    @property
    def trace_path(self) -> str:
        value = self.document.metadata.get("trace_path")
        if isinstance(value, str) and value.strip():
            return value
        return self.document.label

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
            return Path(self.document.label).name
        if self.label == "ext":
            return Path(self.document.label).suffix
        return self.label

    def _get_contents(self) -> str:
        text = self.file_content
        if self.inject and text:
            from ..render.inject import inject_content_in_text

            text = inject_content_in_text(
                text,
                self.depth,
                self.trace_collector,
                self.source,
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
