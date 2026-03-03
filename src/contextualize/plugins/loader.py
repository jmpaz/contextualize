from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from .api import (
    PLUGIN_API_VERSION,
    PLUGIN_DIRS_ENV,
    PLUGIN_ENTRYPOINT_GROUP,
    ExternalPluginSpec,
    LoadedPlugin,
)

_SOURCE_RANK = {
    "path": 2,
    "entrypoint": 1,
}


@dataclass(frozen=True)
class _PluginCandidate:
    plugin: LoadedPlugin
    source_kind: str


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr, flush=True)


def _as_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _load_dotenv_optional() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except Exception:
        return


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in paths:
        key = str(root.expanduser().resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root.expanduser())
    return deduped


def _env_plugin_roots() -> list[Path]:
    _load_dotenv_optional()
    roots: list[Path] = []
    env_raw = (os.environ.get(PLUGIN_DIRS_ENV) or "").strip()
    if env_raw:
        for part in env_raw.split(os.pathsep):
            cleaned = part.strip()
            if cleaned:
                roots.append(Path(cleaned))
    return _dedupe_paths(roots)


def _config_plugin_roots() -> list[Path]:
    return [Path("~/.config/contextualize/plugins").expanduser()]


def _plugin_roots() -> list[Path]:
    return _dedupe_paths([*_config_plugin_roots(), *_env_plugin_roots()])


def _discover_external_plugin_specs(roots: list[Path]) -> list[ExternalPluginSpec]:
    specs: list[ExternalPluginSpec] = []
    for root in _dedupe_paths(roots):
        if not root.is_dir():
            continue
        candidates: list[Path] = [root]
        children = sorted(path for path in root.iterdir() if path.is_dir())
        candidates.extend(children)
        for repo_dir in candidates:
            manifest_path = repo_dir / "plugin.yaml"
            if not manifest_path.is_file():
                continue
            spec = _parse_external_manifest(manifest_path)
            if spec is None:
                continue
            specs.append(spec)
    return specs


def _parse_external_manifest(manifest_path: Path) -> ExternalPluginSpec | None:
    raw: Any
    try:
        import yaml

        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        try:
            raw = _parse_simple_yaml_mapping(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _warn(f"invalid plugin manifest {manifest_path}: {exc}")
            return None
    if not isinstance(raw, dict):
        _warn(f"plugin manifest must be a mapping: {manifest_path}")
        return None

    name = _as_name(raw.get("name"))
    module = _as_name(raw.get("module"))
    api_version_raw = _as_name(raw.get("api_version"))
    api_version = api_version_raw or PLUGIN_API_VERSION
    enabled_raw = raw.get("enabled", True)
    enabled = (
        bool(enabled_raw)
        if not isinstance(enabled_raw, str)
        else (enabled_raw.strip().lower() not in {"0", "false", "no", "off"})
    )
    priority_raw = raw.get("priority", 0)
    priority = 0
    try:
        priority = int(priority_raw)
    except Exception:
        _warn(f"invalid plugin priority in {manifest_path}; using 0")
    if not name or not module:
        _warn(f"plugin manifest missing required fields in {manifest_path}")
        return None

    return ExternalPluginSpec(
        name=name,
        module=module,
        api_version=api_version,
        priority=priority,
        enabled=enabled,
        repo_dir=manifest_path.parent,
        manifest_path=manifest_path,
    )


def _parse_simple_yaml_mapping(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"unsupported manifest line: {raw_line}")
        key, value = line.split(":", 1)
        field = key.strip()
        parsed = _parse_simple_yaml_scalar(value.strip())
        data[field] = parsed
    return data


def _parse_simple_yaml_scalar(value: str) -> Any:
    if not value:
        return ""
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(value)
    except ValueError:
        return value


def _load_external_module(spec: ExternalPluginSpec) -> Any | None:
    source_path = _resolve_module_source(spec.repo_dir, spec.module)
    if source_path is None:
        _warn(
            f"plugin '{spec.name}' module not found: {spec.module} ({spec.manifest_path})"
        )
        return None
    module_name = _external_module_name(spec, source_path)
    module_spec = importlib.util.spec_from_file_location(module_name, source_path)
    if module_spec is None or module_spec.loader is None:
        _warn(f"unable to load plugin '{spec.name}' at {source_path}")
        return None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    parent_dir = str(spec.repo_dir)
    path_inserted = False
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        path_inserted = True
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        _warn(f"plugin '{spec.name}' failed to import: {exc}")
        return None
    finally:
        if path_inserted:
            try:
                sys.path.remove(parent_dir)
            except ValueError:
                pass
    return module


def _is_explicit_module_path(module: str) -> bool:
    return module.endswith(".py") or "/" in module or "\\" in module


def _external_module_name(spec: ExternalPluginSpec, source_path: Path) -> str:
    module_ref = spec.module.strip()
    if module_ref and not _is_explicit_module_path(module_ref):
        parts = [part for part in module_ref.split(".") if part]
        if parts and ("." in module_ref or source_path.name == "__init__.py"):
            return ".".join(parts)
    plugin_hash = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:12]
    return f"contextualize_ext_plugin_{spec.name}_{plugin_hash}"


def _resolve_module_source(repo_dir: Path, module: str) -> Path | None:
    raw = module.strip()
    if not raw:
        return None
    if _is_explicit_module_path(raw):
        candidate = (repo_dir / raw).resolve(strict=False)
        if candidate.is_file():
            return candidate
        return None

    parts = [part for part in raw.split(".") if part]
    if not parts:
        return None
    module_path = repo_dir.joinpath(*parts).with_suffix(".py")
    if module_path.is_file():
        return module_path
    package_init = repo_dir.joinpath(*parts, "__init__.py")
    if package_init.is_file():
        return package_init
    return None


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


def _load_plugins_from_specs(specs: list[ExternalPluginSpec]) -> list[_PluginCandidate]:
    plugins: list[_PluginCandidate] = []
    for spec in specs:
        if not spec.enabled:
            continue
        if spec.api_version != PLUGIN_API_VERSION:
            _warn(
                f"plugin '{spec.name}' api_version={spec.api_version!r} is unsupported "
                f"({spec.manifest_path})"
            )
            continue
        module = _load_external_module(spec)
        if module is None:
            continue
        name = _as_name(getattr(module, "PLUGIN_NAME", None)) or spec.name
        module_api = _as_name(getattr(module, "PLUGIN_API_VERSION", None))
        if module_api and module_api != PLUGIN_API_VERSION:
            _warn(
                f"plugin '{name}' api_version={module_api!r} is unsupported "
                f"({spec.manifest_path})"
            )
            continue
        module_priority = spec.priority
        try:
            module_priority = int(getattr(module, "PLUGIN_PRIORITY", spec.priority))
        except Exception:
            module_priority = spec.priority
        loaded = _validate_plugin_callables(
            name=name,
            origin=str(spec.manifest_path),
            can_resolve=getattr(module, "can_resolve", None),
            resolve=getattr(module, "resolve", None),
            register_auth_command=getattr(module, "register_auth_command", None),
            priority=module_priority,
        )
        if loaded is not None:
            plugins.append(_PluginCandidate(plugin=loaded, source_kind="path"))
    return plugins


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
            plugins.append(_PluginCandidate(plugin=loaded, source_kind="entrypoint"))
    return plugins


def _choose_candidate(
    current: _PluginCandidate,
    candidate: _PluginCandidate,
) -> _PluginCandidate:
    current_rank = _SOURCE_RANK.get(current.source_kind, 0)
    candidate_rank = _SOURCE_RANK.get(candidate.source_kind, 0)
    if candidate_rank > current_rank:
        return candidate
    if candidate_rank < current_rank:
        return current
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
    specs = _discover_external_plugin_specs(_plugin_roots())
    candidates = [
        *_load_plugins_from_specs(specs),
        *_load_entrypoint_plugins(),
    ]
    deduped = _dedupe_candidates(candidates)
    loaded = [candidate.plugin for candidate in deduped]
    loaded.sort(key=lambda plugin: (-plugin.priority, plugin.name, plugin.origin))
    return tuple(loaded)


def clear_loaded_plugins_cache() -> None:
    get_loaded_plugins.cache_clear()
