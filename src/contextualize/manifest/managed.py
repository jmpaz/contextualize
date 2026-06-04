"""Managed manifest hydration registry support."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hydrate import (
    HydrateOverrides,
    HydrateResult,
    apply_hydration_plan,
    build_hydration_plan,
    build_hydration_plan_data,
    clear_context_dir,
    find_untracked_files,
    plan_matches_existing,
)
from .source import load_manifest_text

_REPLACE_POLICIES = {"guarded", "always", "never"}


@dataclass(frozen=True)
class ManagedContext:
    name: str
    target_dir: Path
    manifest: dict[str, Any]
    replace: str


@dataclass(frozen=True)
class ManagedHydrationStatus:
    name: str
    target_dir: str
    manifest_source: str
    context_dir: str | None
    result: str
    reason: str | None
    timestamp: str
    component_count: int | None = None
    file_count: int | None = None
    replace: str = "guarded"


def default_managed_registry_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "contextualize" / "managed-contexts.json"
    return Path.home() / ".config" / "contextualize" / "managed-contexts.json"


def default_managed_status_path() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home) / "contextualize" / "managed-contexts" / "status.json"
    return Path.home() / ".local" / "state" / "contextualize" / "managed-contexts" / "status.json"


def load_managed_contexts(
    registry_path: str | os.PathLike[str] | None = None,
) -> dict[str, ManagedContext]:
    path = Path(registry_path).expanduser() if registry_path else default_managed_registry_path()
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    contexts = raw.get("contexts") if isinstance(raw, dict) else None
    if not isinstance(contexts, dict):
        raise ValueError("Managed context registry must contain a 'contexts' mapping")
    return {
        name: _parse_managed_context(name, value)
        for name, value in contexts.items()
    }


def hydrate_managed_contexts(
    names: list[str] | tuple[str, ...] | None = None,
    *,
    registry_path: str | os.PathLike[str] | None = None,
    status_path: str | os.PathLike[str] | None = None,
    overrides: HydrateOverrides | None = None,
) -> list[ManagedHydrationStatus]:
    contexts = load_managed_contexts(registry_path)
    selected_names = list(names or contexts.keys())
    effective_overrides = overrides or HydrateOverrides()
    statuses: list[ManagedHydrationStatus] = []

    for name in selected_names:
        context = contexts.get(name)
        if context is None:
            statuses.append(
                _status(
                    name=name,
                    target_dir="",
                    manifest_source="",
                    context_dir=None,
                    result="failed",
                    reason=f"managed context is not registered: {name}",
                    replace="guarded",
                )
            )
            continue
        statuses.append(_hydrate_one(context, effective_overrides))

    write_managed_status(statuses, status_path=status_path)
    return statuses


def write_managed_status(
    statuses: list[ManagedHydrationStatus],
    *,
    status_path: str | os.PathLike[str] | None = None,
) -> None:
    path = Path(status_path).expanduser() if status_path else default_managed_status_path()
    previous: dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                previous = loaded
        except (OSError, json.JSONDecodeError):
            previous = {}
    contexts = previous.get("contexts") if isinstance(previous.get("contexts"), dict) else {}
    contexts = dict(contexts)
    for status in statuses:
        contexts[status.name] = asdict(status)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": _now(),
        "contexts": contexts,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_managed_context(name: str, raw: Any) -> ManagedContext:
    if not isinstance(raw, dict):
        raise ValueError(f"Managed context '{name}' must be a mapping")
    target_dir = raw.get("targetDir") or raw.get("target_dir")
    if not isinstance(target_dir, str) or not target_dir:
        raise ValueError(f"Managed context '{name}' targetDir must be a non-empty string")
    manifest = raw.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"Managed context '{name}' manifest must be a mapping")
    sources = [key for key in ("source", "text", "data") if key in manifest]
    if len(sources) != 1:
        raise ValueError(
            f"Managed context '{name}' manifest must set exactly one of source, text, or data"
        )
    replace = raw.get("replace", "guarded")
    if replace not in _REPLACE_POLICIES:
        raise ValueError(
            f"Managed context '{name}' replace must be one of: always, guarded, never"
        )
    return ManagedContext(
        name=name,
        target_dir=Path(os.path.expanduser(target_dir)),
        manifest=manifest,
        replace=replace,
    )


def _hydrate_one(
    context: ManagedContext,
    overrides: HydrateOverrides,
) -> ManagedHydrationStatus:
    target_dir = context.target_dir
    manifest_source = _manifest_source_label(context)
    if not target_dir.is_dir():
        return _status(
            name=context.name,
            target_dir=str(target_dir),
            manifest_source=manifest_source,
            context_dir=None,
            result="failed",
            reason=f"targetDir does not exist or is not a directory: {target_dir}",
            replace=context.replace,
        )

    try:
        plan = _build_plan(context, overrides)
        existing_status = _prepare_existing_context(plan, context.replace)
        if existing_status is not None:
            return _status_from_result(
                context,
                HydrateResult(
                    context_dir=str(plan.context_dir),
                    component_count=plan.component_count,
                    file_count=_planned_file_count(plan),
                    manifest_written=plan.include_meta,
                ),
            outcome=existing_status[0],
            reason=existing_status[1],
            manifest_source=manifest_source,
        )
        result = apply_hydration_plan(plan)
        return _status_from_result(
            context,
            result,
            outcome="hydrated",
            reason=None,
            manifest_source=manifest_source,
        )
    except (OSError, ValueError) as exc:
        return _status(
            name=context.name,
            target_dir=str(target_dir),
            manifest_source=manifest_source,
            context_dir=None,
            result="failed",
            reason=str(exc),
            replace=context.replace,
        )


def _build_plan(context: ManagedContext, overrides: HydrateOverrides):
    target_dir = context.target_dir.resolve()
    manifest = context.manifest
    cwd = str(target_dir)
    if "source" in manifest:
        source = manifest["source"]
        if not isinstance(source, str) or not source:
            raise ValueError(f"Managed context '{context.name}' manifest.source must be a string")
        source_path = Path(os.path.expanduser(source))
        if not source_path.is_absolute():
            source_path = target_dir / source_path
        return build_hydration_plan(str(source_path), overrides=overrides, cwd=cwd)
    if "text" in manifest:
        text = manifest["text"]
        if not isinstance(text, str):
            raise ValueError(f"Managed context '{context.name}' manifest.text must be a string")
        data = load_manifest_text(text, source_label=f"managed context '{context.name}'")
        return build_hydration_plan_data(
            data,
            manifest_cwd=cwd,
            manifest_path=None,
            overrides=overrides,
            cwd=cwd,
        )
    data = manifest["data"]
    if not isinstance(data, dict):
        raise ValueError(f"Managed context '{context.name}' manifest.data must be a mapping")
    return build_hydration_plan_data(
        data,
        manifest_cwd=cwd,
        manifest_path=None,
        overrides=overrides,
        cwd=cwd,
    )


def _prepare_existing_context(plan, replace: str) -> tuple[str, str | None] | None:
    if not plan.context_dir.exists():
        return None
    if plan_matches_existing(plan):
        return ("up-to-date", None)
    if replace == "never":
        return ("skipped", "context exists and replacement policy is never")
    if replace == "guarded":
        untracked = find_untracked_files(plan.context_dir)
        if untracked:
            sample = ", ".join(untracked[:5])
            suffix = "" if len(untracked) <= 5 else f", and {len(untracked) - 5} more"
            return ("skipped", f"context contains untracked files: {sample}{suffix}")
    clear_context_dir(plan.context_dir)
    return None


def _planned_file_count(plan) -> int:
    written_paths = {path.as_posix() for path, _ in plan.files_to_write}
    symlinked_paths = {path.as_posix() for path, _ in plan.files_to_symlink}
    return len(written_paths | symlinked_paths)


def _status_from_result(
    context: ManagedContext,
    hydrate_result: HydrateResult,
    *,
    outcome: str,
    reason: str | None,
    manifest_source: str,
) -> ManagedHydrationStatus:
    return _status(
        name=context.name,
        target_dir=str(context.target_dir),
        manifest_source=manifest_source,
        context_dir=hydrate_result.context_dir,
        result=outcome,
        reason=reason,
        component_count=hydrate_result.component_count,
        file_count=hydrate_result.file_count,
        replace=context.replace,
    )


def _status(
    *,
    name: str,
    target_dir: str,
    manifest_source: str,
    context_dir: str | None,
    result: str,
    reason: str | None,
    replace: str,
    component_count: int | None = None,
    file_count: int | None = None,
) -> ManagedHydrationStatus:
    return ManagedHydrationStatus(
        name=name,
        target_dir=target_dir,
        manifest_source=manifest_source,
        context_dir=context_dir,
        result=result,
        reason=reason,
        timestamp=_now(),
        component_count=component_count,
        file_count=file_count,
        replace=replace,
    )


def _manifest_source_label(context: ManagedContext) -> str:
    manifest = context.manifest
    if "source" in manifest:
        return str(manifest["source"])
    if "text" in manifest:
        return "inline text"
    return "inline data"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
