from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, TypedDict

PLUGIN_API_VERSION = "1"
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


class PluginTargetDescriptor(TypedDict, total=False):
    provider: str
    kind: str
    is_external: bool
    group_key: str


CanResolveFn = Callable[[str, PluginContext], bool]
ResolveFn = Callable[[str, PluginContext], list[PluginDocument]]
RegisterAuthCommandFn = Callable[[Any], None]
ClassifyTargetFn = Callable[[str, PluginContext], PluginTargetDescriptor | None]
NormalizeManifestConfigFn = Callable[[dict[str, Any] | None], dict[str, Any] | None]


@dataclass(frozen=True)
class LoadedPlugin:
    name: str
    priority: int
    origin: str
    can_resolve: CanResolveFn
    resolve: ResolveFn
    register_auth_command: RegisterAuthCommandFn | None = None
    classify_target: ClassifyTargetFn | None = None
    normalize_manifest_config: NormalizeManifestConfigFn | None = None
