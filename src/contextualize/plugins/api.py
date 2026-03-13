from __future__ import annotations

from dataclasses import dataclass, field
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
    timeout: float
    prompt: str
    bias_terms: tuple[str, ...]
    diarize: bool
    speaker_count: int | None


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
        [bytes, str, str | None, float, dict[str, Any]],
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
    register_cli_options: RegisterCliOptionsFn | None = None
    collect_cli_overrides: CollectCliOverridesFn | None = None
    transcription_providers: tuple[TranscriptionProvider, ...] = field(
        default_factory=tuple
    )
    transcription_gates: tuple[TranscriptionGate, ...] = field(default_factory=tuple)
    plugin_kind: str = "source"
