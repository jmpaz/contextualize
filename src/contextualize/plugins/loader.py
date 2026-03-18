from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Any

from .api import (
    PLUGIN_API_VERSION,
    PLUGIN_ENTRYPOINT_GROUP,
    LoadedPlugin,
    TranscriptionGate,
    TranscriptionProvider,
)


@dataclass(frozen=True)
class _PluginCandidate:
    plugin: LoadedPlugin


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr, flush=True)


def _as_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _validate_plugin_callables(
    *,
    name: str,
    origin: str,
    can_resolve: Any,
    resolve: Any,
    register_auth_command: Any,
    classify_target: Any,
    normalize_manifest_config: Any,
    register_cli_options: Any,
    collect_cli_overrides: Any,
    transcription_providers: Any,
    transcription_gates: Any,
    plugin_kind: Any,
    priority: int,
) -> LoadedPlugin | None:
    if not callable(can_resolve):
        _warn(f"plugin '{name}' has non-callable can_resolve ({origin})")
        return None
    if not callable(resolve):
        _warn(f"plugin '{name}' has non-callable resolve ({origin})")
        return None
    auth_hook = None
    if register_auth_command is not None:
        if not callable(register_auth_command):
            _warn(f"plugin '{name}' has non-callable register_auth_command ({origin})")
            return None
        auth_hook = register_auth_command
    classify_hook = None
    if classify_target is not None:
        if not callable(classify_target):
            _warn(f"plugin '{name}' has non-callable classify_target ({origin})")
            return None
        classify_hook = classify_target
    normalize_hook = None
    if normalize_manifest_config is not None:
        if not callable(normalize_manifest_config):
            _warn(
                f"plugin '{name}' has non-callable normalize_manifest_config ({origin})"
            )
            return None
        normalize_hook = normalize_manifest_config
    cli_register_hook = None
    if register_cli_options is not None:
        if not callable(register_cli_options):
            _warn(f"plugin '{name}' has non-callable register_cli_options ({origin})")
            return None
        cli_register_hook = register_cli_options
    cli_collect_hook = None
    if collect_cli_overrides is not None:
        if not callable(collect_cli_overrides):
            _warn(f"plugin '{name}' has non-callable collect_cli_overrides ({origin})")
            return None
        cli_collect_hook = collect_cli_overrides
    provider_items: tuple[TranscriptionProvider, ...] = ()
    if transcription_providers is not None:
        if not isinstance(transcription_providers, (list, tuple)):
            _warn(
                f"plugin '{name}' returned invalid transcription providers ({origin})"
            )
            return None
        normalized_items: list[TranscriptionProvider] = []
        for item in transcription_providers:
            if not isinstance(item, TranscriptionProvider):
                _warn(
                    f"plugin '{name}' returned non-provider transcription hook ({origin})"
                )
                return None
            normalized_items.append(item)
        provider_items = tuple(normalized_items)
    gate_items: tuple[TranscriptionGate, ...] = ()
    if transcription_gates is not None:
        if not isinstance(transcription_gates, (list, tuple)):
            _warn(f"plugin '{name}' returned invalid transcription gates ({origin})")
            return None
        normalized_gates: list[TranscriptionGate] = []
        for item in transcription_gates:
            if not isinstance(item, TranscriptionGate):
                _warn(
                    f"plugin '{name}' returned non-gate transcription hook ({origin})"
                )
                return None
            normalized_gates.append(item)
        gate_items = tuple(normalized_gates)
    normalized_kind = "source"
    if plugin_kind is not None:
        if not isinstance(plugin_kind, str):
            _warn(f"plugin '{name}' has invalid plugin_kind ({origin})")
            return None
        candidate_kind = plugin_kind.strip().lower()
        if candidate_kind not in {"source", "processor"}:
            _warn(
                f"plugin '{name}' has unsupported plugin_kind={plugin_kind!r} ({origin})"
            )
            return None
        normalized_kind = candidate_kind
    return LoadedPlugin(
        name=name,
        priority=priority,
        origin=origin,
        can_resolve=can_resolve,
        resolve=resolve,
        register_auth_command=auth_hook,
        classify_target=classify_hook,
        normalize_manifest_config=normalize_hook,
        register_cli_options=cli_register_hook,
        collect_cli_overrides=cli_collect_hook,
        transcription_providers=provider_items,
        transcription_gates=gate_items,
        plugin_kind=normalized_kind,
    )


def _iter_plugin_entrypoints() -> list[Any]:
    try:
        from importlib import metadata as importlib_metadata

        loaded = importlib_metadata.entry_points()
    except Exception:
        return []

    try:
        selected = loaded.select(group=PLUGIN_ENTRYPOINT_GROUP)
        return list(selected)
    except Exception:
        by_group = getattr(loaded, "get", None)
        if callable(by_group):
            try:
                return list(by_group(PLUGIN_ENTRYPOINT_GROUP, []))
            except Exception:
                return []
    return []


def _load_entrypoint_plugins() -> list[_PluginCandidate]:
    plugins: list[_PluginCandidate] = []
    for entrypoint in _iter_plugin_entrypoints():
        ep_name = _as_name(getattr(entrypoint, "name", None)) or "unknown"
        ep_value = _as_name(getattr(entrypoint, "value", None)) or ep_name
        origin = f"entrypoint:{ep_value}"
        try:
            exported = entrypoint.load()
        except Exception as exc:
            _warn(f"entrypoint plugin '{ep_name}' failed to load: {exc}")
            continue

        module = exported if isinstance(exported, ModuleType) else None
        plugin_obj = module if module is not None else exported

        name = _as_name(getattr(plugin_obj, "PLUGIN_NAME", None)) or ep_name
        api_version = _as_name(getattr(plugin_obj, "PLUGIN_API_VERSION", None))
        if api_version and api_version != PLUGIN_API_VERSION:
            _warn(
                f"entrypoint plugin '{name}' api_version={api_version!r} is unsupported"
            )
            continue

        priority = 0
        try:
            priority = int(getattr(plugin_obj, "PLUGIN_PRIORITY", 0))
        except Exception:
            priority = 0

        transcription_providers = None
        get_providers = getattr(plugin_obj, "get_transcription_providers", None)
        if callable(get_providers):
            try:
                transcription_providers = get_providers()
            except Exception as exc:
                _warn(
                    f"entrypoint plugin '{name}' transcription provider discovery failed: {exc}"
                )
                continue
        transcription_gates = None
        get_gates = getattr(plugin_obj, "get_transcription_gates", None)
        if callable(get_gates):
            try:
                transcription_gates = get_gates()
            except Exception as exc:
                _warn(
                    f"entrypoint plugin '{name}' transcription gate discovery failed: {exc}"
                )
                continue

        loaded = _validate_plugin_callables(
            name=name,
            origin=origin,
            can_resolve=getattr(plugin_obj, "can_resolve", None),
            resolve=getattr(plugin_obj, "resolve", None),
            register_auth_command=getattr(plugin_obj, "register_auth_command", None),
            classify_target=getattr(plugin_obj, "classify_target", None),
            normalize_manifest_config=getattr(
                plugin_obj, "normalize_manifest_config", None
            ),
            register_cli_options=getattr(plugin_obj, "register_cli_options", None),
            collect_cli_overrides=getattr(plugin_obj, "collect_cli_overrides", None),
            transcription_providers=transcription_providers,
            transcription_gates=transcription_gates,
            plugin_kind=getattr(plugin_obj, "PLUGIN_KIND", None),
            priority=priority,
        )
        if loaded is not None:
            plugins.append(_PluginCandidate(plugin=loaded))
    return plugins


def _choose_candidate(
    current: _PluginCandidate,
    candidate: _PluginCandidate,
) -> _PluginCandidate:
    if candidate.plugin.priority > current.plugin.priority:
        return candidate
    if candidate.plugin.priority < current.plugin.priority:
        return current
    if candidate.plugin.origin < current.plugin.origin:
        return candidate
    return current


def _dedupe_candidates(candidates: list[_PluginCandidate]) -> list[_PluginCandidate]:
    selected: dict[str, _PluginCandidate] = {}
    for candidate in candidates:
        name = candidate.plugin.name
        existing = selected.get(name)
        if existing is None:
            selected[name] = candidate
            continue
        chosen = _choose_candidate(existing, candidate)
        if chosen is existing:
            _warn(
                f"plugin '{name}' from {candidate.plugin.origin} ignored; using {existing.plugin.origin}"
            )
        else:
            _warn(
                f"plugin '{name}' from {existing.plugin.origin} replaced by {candidate.plugin.origin}"
            )
            selected[name] = chosen
    return list(selected.values())


@lru_cache(maxsize=1)
def get_loaded_plugins() -> tuple[LoadedPlugin, ...]:
    candidates = _load_entrypoint_plugins()
    deduped = _dedupe_candidates(candidates)
    loaded = [candidate.plugin for candidate in deduped]
    loaded.sort(key=lambda plugin: (-plugin.priority, plugin.name, plugin.origin))
    return tuple(loaded)


def clear_loaded_plugins_cache() -> None:
    get_loaded_plugins.cache_clear()
