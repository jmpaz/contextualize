from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, TypedDict

PLUGIN_API_VERSION = "1"
PLUGIN_ENTRYPOINT_GROUP = "contextualize.plugins"


class PluginDocument(TypedDict, total=False):
    """A resolved document.

    `prose` is the authored text without scaffolding (frontmatter, media
    descriptions, comments, structural markup). Omit it to defer to generic
    extraction; set to "" to declare the unit carries no authored prose.
    `prose_authors` lists the distinct authors of `prose`.
    """

    source: str
    label: str
    content: str
    metadata: dict[str, Any]
    prose: str
    prose_authors: list[str]


class PluginListItem(TypedDict, total=False):
    target: str
    label: str
    kind: str
    traverse: bool
    metadata: dict[str, Any]


class PluginListEnvelopeMetadata(TypedDict, total=False):
    summary: dict[str, Any]
    pagination: dict[str, Any] | None
    metadata: dict[str, Any]
    capabilities: dict[str, Any]


class PluginListEnvelope(PluginListEnvelopeMetadata):
    targets: list[PluginListItem]


class PluginMaterializedFile(TypedDict, total=False):
    source: str
    label: str
    filename: str
    content: bytes
    content_type: str | None
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
    cache_only: bool
    overrides: dict[str, Any]
    include_containing: bool
    list_limit: int
    list_offset: int


class PluginTargetDescriptor(TypedDict, total=False):
    provider: str
    kind: str
    is_external: bool
    group_key: str
    metadata: dict[str, Any]
    relations: list[PluginListItem]
    capabilities: dict[str, Any]
    error: str | None


CanResolveFn = Callable[[str, PluginContext], bool]
ResolveFn = Callable[[str, PluginContext], list[PluginDocument]]
ListTargetsFn = Callable[[str, PluginContext], PluginListEnvelope]
MaterializeFn = Callable[[str, PluginContext], list[PluginMaterializedFile]]
RegisterAuthCommandFn = Callable[[Any], None]
RegisterCommandFn = Callable[[Any], None]
ClassifyTargetFn = Callable[[str, PluginContext], PluginTargetDescriptor | None]
NormalizeManifestConfigFn = Callable[[dict[str, Any] | None], dict[str, Any] | None]
RegisterCliOptionsFn = Callable[[str, Any], None]
CollectCliOverridesFn = Callable[[str, dict[str, Any]], dict[str, Any] | None]


class TranscriptionProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class TranscriptionProviderUnavailableError(TranscriptionProviderError):
    pass


class TranscriptionProviderUnsupportedError(TranscriptionProviderError):
    pass


class TranscriptionProviderAuthError(TranscriptionProviderError):
    pass


@dataclass(frozen=True)
class TranscriptionRequest:
    data: bytes
    filename: str
    content_type: str | None
    timeout: float | None
    prompt: str
    bias_terms: tuple[str, ...]
    diarize: bool
    speaker_count: int | None
    language: str | None = None
    model: str | None = None
    timestamp_granularities: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    model: str
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TranscriptionGateDecision:
    needs_diarization: bool
    speaker_count: int | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TranscriptionGate:
    name: str
    analyze: Callable[
        [bytes, str, str | None, float | None, dict[str, Any]],
        TranscriptionGateDecision,
    ]
    cache_identity: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class TranscriptionProvider:
    name: str
    priority: int
    transcribe: Callable[[TranscriptionRequest], TranscriptionResult]
    cache_identity: Callable[[TranscriptionRequest], dict[str, Any]]
    is_available: Callable[[], bool] | None = None
    supports_diarization: bool = False
    default_model: str | None = None


@dataclass(frozen=True)
class LoadedPlugin:
    name: str
    priority: int
    origin: str
    can_resolve: CanResolveFn
    resolve: ResolveFn
    list_targets: ListTargetsFn | None = None
    materialize: MaterializeFn | None = None
    register_auth_command: RegisterAuthCommandFn | None = None
    classify_target: ClassifyTargetFn | None = None
    normalize_manifest_config: NormalizeManifestConfigFn | None = None
    register_cli_options: RegisterCliOptionsFn | None = None
    collect_cli_overrides: CollectCliOverridesFn | None = None
    register_command: RegisterCommandFn | None = None
    transcription_providers: tuple[TranscriptionProvider, ...] = field(
        default_factory=tuple
    )
    transcription_gates: tuple[TranscriptionGate, ...] = field(default_factory=tuple)
    plugin_kind: str = "source"
