from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Any

from .api import PLUGIN_API_VERSION, PLUGIN_ENTRYPOINT_GROUP, LoadedPlugin


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
    return LoadedPlugin(
        name=name,
        priority=priority,
        origin=origin,
        can_resolve=can_resolve,
        resolve=resolve,
        register_auth_command=auth_hook,
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

        loaded = _validate_plugin_callables(
            name=name,
            origin=origin,
            can_resolve=getattr(plugin_obj, "can_resolve", None),
            resolve=getattr(plugin_obj, "resolve", None),
            register_auth_command=getattr(plugin_obj, "register_auth_command", None),
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
