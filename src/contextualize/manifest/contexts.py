"""Context registry hydration support."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from contextlib import redirect_stderr
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..progress import (
    log_progress,
    reset_progress_context,
    set_progress_context,
    write_progress_log,
)
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
from .source import load_manifest_source, load_manifest_text

_REPLACE_POLICIES = {"guarded", "always", "never"}


@dataclass(frozen=True)
class ContextEntry:
    name: str
    target_dir: Path
    manifest: dict[str, Any]
    replace: str


@dataclass(frozen=True)
class ContextHydrationStatus:
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


def default_context_registry_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "contextualize" / "contexts.json"
    return Path.home() / ".config" / "contextualize" / "contexts.json"


def default_context_status_path() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home) / "contextualize" / "contexts" / "status.json"
    return Path.home() / ".local" / "state" / "contextualize" / "contexts" / "status.json"


def manifest_link_identity(resolved_manifest_path: Path) -> str:
    return hashlib.sha256(str(resolved_manifest_path).encode("utf-8")).hexdigest()


def default_manifest_link_cache_root() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return root / "contextualize" / "cache" / "manifest-links" / "v1"


def canonical_manifest_link_dir(
    resolved_manifest_path: Path, *, cache_root: Path | None = None
) -> Path:
    digest = manifest_link_identity(resolved_manifest_path)
    root = cache_root or default_manifest_link_cache_root()
    return root / digest[:2] / digest / "context"


def default_manifest_link_status_path() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME")
    root = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return root / "contextualize" / "manifest-links" / "status.json"


def load_context_registry(
    registry_path: str | os.PathLike[str] | None = None,
) -> dict[str, ContextEntry]:
    path = Path(registry_path).expanduser() if registry_path else default_context_registry_path()
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    contexts = raw.get("contexts") if isinstance(raw, dict) else None
    if not isinstance(contexts, dict):
        raise ValueError("Context registry must contain a 'contexts' mapping")
    return {
        name: _parse_context_entry(name, value)
        for name, value in contexts.items()
    }


def hydrate_contexts(
    names: list[str] | tuple[str, ...] | None = None,
    *,
    registry_path: str | os.PathLike[str] | None = None,
    status_path: str | os.PathLike[str] | None = None,
    overrides: HydrateOverrides | None = None,
) -> list[ContextHydrationStatus]:
    contexts = load_context_registry(registry_path)
    selected_names = list(names or contexts.keys())
    effective_overrides = overrides or HydrateOverrides()
    statuses: list[ContextHydrationStatus] = []

    log_progress("hydrate", "context", "total", count=len(selected_names))
    for name in selected_names:
        log_progress("hydrate", "context", "start", target=name)
        context = contexts.get(name)
        if context is None:
            statuses.append(
                _status(
                    name=name,
                    target_dir="",
                    manifest_source="",
                    context_dir=None,
                    result="failed",
                    reason=f"context is not registered: {name}",
                    replace="guarded",
                )
            )
            log_progress(
                "hydrate",
                "context",
                "failed",
                target=name,
                detail="not registered",
            )
            continue
        with redirect_stderr(_ContextStderrPrefixer(context.name, sys.stderr)):
            token = set_progress_context(context.name)
            try:
                status = _hydrate_one(context, effective_overrides)
            finally:
                reset_progress_context(token)
        statuses.append(status)
        log_progress(
            "hydrate",
            "context",
            "failed" if status.result == "failed" else "done",
            target=name,
            detail=status.result,
        )

    write_context_status(statuses, status_path=status_path)
    return statuses


def write_context_status(
    statuses: list[ContextHydrationStatus],
    *,
    status_path: str | os.PathLike[str] | None = None,
) -> None:
    path = Path(status_path).expanduser() if status_path else default_context_status_path()
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


def _parse_context_entry(name: str, raw: Any) -> ContextEntry:
    if not isinstance(raw, dict):
        raise ValueError(f"Context '{name}' must be a mapping")
    target_dir = raw.get("targetDir") or raw.get("target_dir")
    if not isinstance(target_dir, str) or not target_dir:
        raise ValueError(f"Context '{name}' targetDir must be a non-empty string")
    manifest = raw.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"Context '{name}' manifest must be a mapping")
    sources = [key for key in ("source", "text", "data") if key in manifest]
    if len(sources) != 1:
        raise ValueError(
            f"Context '{name}' manifest must set exactly one of source, text, or data"
        )
    replace = raw.get("replace", "guarded")
    if replace not in _REPLACE_POLICIES:
        raise ValueError(
            f"Context '{name}' replace must be one of: always, guarded, never"
        )
    return ContextEntry(
        name=name,
        target_dir=Path(os.path.expanduser(target_dir)),
        manifest=manifest,
        replace=replace,
    )


def _hydrate_one(
    context: ContextEntry,
    overrides: HydrateOverrides,
    *,
    resolving: tuple[Path, ...] = (),
) -> ContextHydrationStatus:
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
        plan = _build_plan(context, overrides, resolving=resolving)
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


def _build_plan(
    context: ContextEntry,
    overrides: HydrateOverrides,
    *,
    resolving: tuple[Path, ...] = (),
):
    target_dir = context.target_dir.resolve()
    manifest = context.manifest
    cwd = str(target_dir)
    if "source" in manifest:
        source = manifest["source"]
        if not isinstance(source, str) or not source:
            raise ValueError(f"Context '{context.name}' manifest.source must be a string")
        source_path = Path(os.path.expanduser(source))
        if not source_path.is_absolute():
            source_path = target_dir / source_path
        return build_hydration_plan(
            str(source_path), overrides=overrides, cwd=cwd, _resolving=resolving
        )
    if "text" in manifest:
        text = manifest["text"]
        if not isinstance(text, str):
            raise ValueError(f"Context '{context.name}' manifest.text must be a string")
        data = load_manifest_text(text, source_label=f"context '{context.name}'")
        return build_hydration_plan_data(
            data,
            manifest_cwd=cwd,
            manifest_path=None,
            overrides=overrides,
            cwd=cwd,
            _resolving=resolving,
        )
    data = manifest["data"]
    if not isinstance(data, dict):
        raise ValueError(f"Context '{context.name}' manifest.data must be a mapping")
    return build_hydration_plan_data(
        data,
        manifest_cwd=cwd,
        manifest_path=None,
        overrides=overrides,
        cwd=cwd,
        _resolving=resolving,
    )


def _prepare_existing_context(plan, replace: str) -> tuple[str, str | None] | None:
    if not plan.context_dir.exists():
        return None
    if plan_matches_existing(plan):
        return ("up-to-date", None)
    if replace == "never":
        return ("skipped", "context exists and replacement policy is never")
    if replace == "guarded":
        untracked = find_untracked_files(
            plan.context_dir,
            planned_paths=_planned_paths(plan),
        )
        if untracked:
            sample = ", ".join(untracked[:5])
            suffix = "" if len(untracked) <= 5 else f", and {len(untracked) - 5} more"
            return ("skipped", f"context contains untracked files: {sample}{suffix}")
    clear_context_dir(plan.context_dir)
    return None


def _planned_file_count(plan) -> int:
    return len(_planned_paths(plan))


def _peek_manifest_name(resolved: Path) -> str | None:
    try:
        source = load_manifest_source(str(resolved))
    except (OSError, ValueError):
        return None
    cfg = source.data.get("config") if isinstance(source.data, dict) else None
    if not isinstance(cfg, dict):
        return None
    name = cfg.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def ensure_manifest_link_hydrated(
    manifest_path: str | os.PathLike[str],
    *,
    parent_overrides: HydrateOverrides,
    resolving: tuple[Path, ...] = (),
) -> tuple[Path, str | None]:
    resolved = Path(manifest_path).expanduser().resolve()
    if resolved in resolving:
        chain = " -> ".join(str(p) for p in (*resolving, resolved))
        raise ValueError(f"Cycle detected while resolving linked manifests: {chain}")
    if not resolved.is_file():
        raise ValueError(f"Linked manifest not found: {resolved}")

    manifest_name = _peek_manifest_name(resolved)
    canonical_dir = canonical_manifest_link_dir(resolved)
    entry = ContextEntry(
        name=f"manifest-link:{manifest_link_identity(resolved)[:16]}",
        target_dir=resolved.parent,
        manifest={"source": str(resolved)},
        replace="guarded",
    )
    child_overrides = HydrateOverrides(
        context_dir=str(canonical_dir),
        use_cache=parent_overrides.use_cache,
        cache_ttl=parent_overrides.cache_ttl,
        refresh_cache=parent_overrides.refresh_cache,
        plugin_overrides=parent_overrides.plugin_overrides,
        embedded_resolution=parent_overrides.embedded_resolution,
    )

    token = set_progress_context(entry.name)
    try:
        status = _hydrate_one(entry, child_overrides, resolving=resolving + (resolved,))
    finally:
        reset_progress_context(token)

    write_context_status(
        [status], status_path=default_manifest_link_status_path()
    )

    hydrated_dir = status.context_dir
    if status.result in ("hydrated", "up-to-date") and hydrated_dir:
        return Path(hydrated_dir), manifest_name
    raise ValueError(
        f"Failed to hydrate linked manifest {resolved}: {status.reason or status.result}"
    )


def _planned_paths(plan) -> set[str]:
    written_paths = {
        path.relative_to(plan.context_dir).as_posix()
        for path, _ in plan.files_to_write
    }
    copied_paths = {
        path.relative_to(plan.context_dir).as_posix()
        for path, _ in plan.files_to_copy
    }
    symlinked_paths = {
        path.relative_to(plan.context_dir).as_posix()
        for path, _ in plan.files_to_symlink
    }
    return written_paths | copied_paths | symlinked_paths


def _status_from_result(
    context: ContextEntry,
    hydrate_result: HydrateResult,
    *,
    outcome: str,
    reason: str | None,
    manifest_source: str,
) -> ContextHydrationStatus:
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
) -> ContextHydrationStatus:
    return ContextHydrationStatus(
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


def _manifest_source_label(context: ContextEntry) -> str:
    manifest = context.manifest
    if "source" in manifest:
        return str(manifest["source"])
    if "text" in manifest:
        return "inline text"
    return "inline data"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _ContextStderrPrefixer:
    def __init__(self, name: str, target) -> None:
        self.name = name
        self.target = target
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._write_line(line, newline=True)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._write_line(self._buffer, newline=False)
            self._buffer = ""
        self.target.flush()

    def _write_line(self, line: str, *, newline: bool) -> None:
        handled = False
        if line:
            prefixed = f"context {self.name}: {line}"
            handled = write_progress_log(prefixed)
            if not handled:
                self.target.write(prefixed)
        if newline and not handled:
            self.target.write("\n")
