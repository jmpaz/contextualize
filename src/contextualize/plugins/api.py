from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, TypedDict

PLUGIN_API_VERSION = "1"
PLUGIN_DIRS_ENV = "CONTEXTUALIZE_PLUGIN_DIRS"
PLUGIN_ENTRYPOINT_GROUP = "contextualize.plugins"


class PluginDocument(TypedDict, total=False):
    source: str
    label: str
    content: str
    metadata: dict[str, Any]


class PluginContext(TypedDict, total=False):
    format: str
    label: str
    label_suffix: str | None
    include_token_count: bool
    token_target: str
    inject: bool
    depth: int
    use_cache: bool
    cache_ttl: timedelta | None
    refresh_cache: bool
    overrides: dict[str, Any]


CanResolveFn = Callable[[str, PluginContext], bool]
ResolveFn = Callable[[str, PluginContext], list[PluginDocument]]


@dataclass(frozen=True)
class LoadedPlugin:
    name: str
    priority: int
    origin: str
    can_resolve: CanResolveFn
    resolve: ResolveFn


@dataclass(frozen=True)
class ExternalPluginSpec:
    name: str
    module: str
    api_version: str
    priority: int
    enabled: bool
    repo_dir: Path
    manifest_path: Path
