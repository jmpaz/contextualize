"""Context registry hydration support."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from contextlib import redirect_stderr
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..progress import (
    log_progress,
    progress_counters,
    reset_progress_context,
    set_progress_context,
    write_progress_log,
)
from ..runtime import get_refresh_cache, get_refresh_media
from ..utils import read_config
from .hydrate import (
    HydrationRunCache,
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
_CONTEXT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


@dataclass(frozen=True)
class ContextEntry:
    name: str
    target_dir: Path
    manifest: dict[str, Any]
    replace: str
    designations: tuple[str, ...] = ()
    context_dir: str | None = None
    origin: str = "registry"
    ensure_target_dir: bool = False


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
    receipt: dict[str, Any] | None = None


def default_context_registry_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "contextualize" / "contexts.json"
    return Path.home() / ".config" / "contextualize" / "contexts.json"


def default_context_status_path() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home) / "contextualize" / "contexts" / "status.json"
    return (
        Path.home() / ".local" / "state" / "contextualize" / "contexts" / "status.json"
    )


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
    path = (
        Path(registry_path).expanduser()
        if registry_path
        else default_context_registry_path()
    )
    if not path.exists():
        if registry_path:
            raise FileNotFoundError(str(path))
        raw: dict[str, Any] = {"contexts": {}}
    else:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    contexts = raw.get("contexts") if isinstance(raw, dict) else None
    if not isinstance(contexts, dict):
        raise ValueError("Context registry must contain a 'contexts' mapping")
    static_contexts = {
        name: _parse_context_entry(name, value) for name, value in contexts.items()
    }
    return _merge_subscribed_contexts(static_contexts)


def _merge_subscribed_contexts(
    static_contexts: dict[str, ContextEntry],
) -> dict[str, ContextEntry]:
    config = read_config()
    if not isinstance(config, dict):
        config = {}
    subscriptions = _context_subscriptions(config)
    if not subscriptions:
        return static_contexts

    contexts = dict(static_contexts)
    static_sources = {
        resolved
        for context in static_contexts.values()
        if (resolved := _resolved_manifest_source(context)) is not None
    }
    discovered: dict[str, str] = {}

    for subscription in subscriptions:
        for context in _discover_subscription_contexts(subscription):
            source_path = _resolved_manifest_source(context)
            if source_path is not None and source_path in static_sources:
                continue
            if context.name in static_contexts:
                static_context = static_contexts[context.name]
                if source_path != _resolved_manifest_source(static_context):
                    _warn_context_subscription(
                        f"skipping subscribed context {context.name}: "
                        "name collides with static registry entry"
                    )
                continue
            previous_source = discovered.get(context.name)
            current_source = manifest_source_label(context)
            if previous_source is not None and previous_source != current_source:
                raise ValueError(
                    f"Subscribed context name '{context.name}' is used by both "
                    f"{previous_source} and {current_source}"
                )
            discovered[context.name] = current_source
            contexts[context.name] = context
    return contexts


def _context_subscriptions(config: dict[str, Any]) -> list[dict[str, Any]]:
    contexts = config.get("contexts")
    if not isinstance(contexts, dict):
        return []
    subscriptions = contexts.get("subscriptions", [])
    if subscriptions in (None, []):
        return []
    if not isinstance(subscriptions, list):
        raise ValueError("contexts.subscriptions must be a list")
    return [_parse_context_subscription(value) for value in subscriptions]


def _parse_context_subscription(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("contexts.subscriptions entries must be mappings")
    source = raw.get("source")
    if source != "zk":
        raise ValueError("contexts.subscriptions entries currently require source: zk")
    root = _required_subscription_string(raw, "root")
    tag = _required_subscription_string(raw, "tag")
    target_root = _required_subscription_string(raw, "targetRoot")
    context_dir = _optional_context_dir(raw, "contexts.subscriptions")
    replace = raw.get("replace", "guarded")
    if replace not in _REPLACE_POLICIES:
        raise ValueError(
            "contexts.subscriptions replace must be one of: always, guarded, never"
        )
    return {
        "source": source,
        "root": root,
        "tag": tag,
        "targetRoot": target_root,
        "replace": replace,
        "contextDir": context_dir,
    }


def _optional_context_dir(raw: dict[str, Any], label: str) -> str | None:
    value = raw.get("contextDir", raw.get("context_dir"))
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} contextDir must be a non-empty string")
    return value.strip()


def _required_subscription_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"contexts.subscriptions entries require {key}")
    return value.strip()


def _discover_subscription_contexts(
    subscription: dict[str, Any],
) -> list[ContextEntry]:
    root = Path(subscription["root"]).expanduser()
    target_root = Path(subscription["targetRoot"]).expanduser()
    notes = _zk_subscription_notes(root=root, tag=subscription["tag"])
    contexts: list[ContextEntry] = []
    for note in notes:
        path = _note_path(root, note)
        if path is None:
            _warn_context_subscription("skipping subscribed note without a path")
            continue
        if not path.is_file():
            _warn_context_subscription(f"skipping subscribed note missing file: {path}")
            continue
        try:
            source = load_manifest_source(str(path))
        except (OSError, ValueError) as exc:
            _warn_context_subscription(
                f"skipping subscribed note without a valid manifest: {path}: {exc}"
            )
            continue
        name = _subscribed_context_name(path, source.data)
        if name is None:
            _warn_context_subscription(
                f"skipping subscribed note without cx.context or config.name: {path}"
            )
            continue
        contexts.append(
            ContextEntry(
                name=name,
                target_dir=target_root / name,
                manifest={"source": str(path.resolve())},
                replace=subscription["replace"],
                context_dir=subscription["contextDir"],
                origin=f"tag:{subscription['tag']}",
                ensure_target_dir=True,
            )
        )
    return contexts


def links_discovery_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """`links.discovery` from config.yaml: the tag scope links aggregation
    scans for manifest-bearing notes. Tags default per marks-spec ruling 10;
    a missing root means the scan is skipped, legibly."""
    if config is None:
        config = read_config()
    if not isinstance(config, dict):
        config = {}
    links_cfg = config.get("links")
    discovery = links_cfg.get("discovery") if isinstance(links_cfg, dict) else None
    tags = ["ctx/manifest"]
    root: Path | None = None
    if isinstance(discovery, dict):
        raw_tags = discovery.get("tags")
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        if isinstance(raw_tags, list):
            cleaned = [str(item).strip() for item in raw_tags if str(item).strip()]
            if cleaned:
                tags = cleaned
        raw_root = discovery.get("root")
        if isinstance(raw_root, str) and raw_root.strip():
            root = Path(raw_root.strip()).expanduser()
    return {"tags": tags, "root": root}


def discover_tagged_notes(*, root: Path, tags: list[str]) -> list[Path]:
    """Candidate notes for links aggregation: one zk query per tag, glob
    over-fetching (`<tag>*` matches subtags but also e.g. `ctx/manifesto`);
    `frontmatter_has_manifest_tag` is the precise filter."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for tag in tags:
        for note in _zk_subscription_notes(root=root, tag=f"{tag}*"):
            path = _note_path(root, note)
            if path is None or path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def _zk_subscription_notes(*, root: Path, tag: str) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["zk", "list", "--tag", tag, "--format", "json", "--quiet"],
            cwd=str(root),
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "zk is required for contexts.subscriptions source: zk"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise ValueError(f"Failed to list zk notes for tag '{tag}'{suffix}") from exc

    stdout = result.stdout.strip()
    if not stdout:
        return []
    try:
        notes = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"zk list for tag '{tag}' did not return JSON") from exc
    if not isinstance(notes, list):
        raise ValueError(f"zk list for tag '{tag}' must return a list")
    return [note for note in notes if isinstance(note, dict)]


def _note_path(root: Path, note: dict[str, Any]) -> Path | None:
    raw = note.get("absPath") or note.get("path") or note.get("filename")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def _subscribed_context_name(path: Path, manifest: dict[str, Any]) -> str | None:
    override = _frontmatter_context_name(path)
    if override is not None:
        return _validate_context_name(override, source=f"{path} frontmatter cx.context")
    cfg = manifest.get("config") if isinstance(manifest, dict) else None
    manifest_name = cfg.get("name") if isinstance(cfg, dict) else None
    if isinstance(manifest_name, str):
        return _slug_context_name(manifest_name)
    return None


def _frontmatter_context_name(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end_idx = next(
            idx for idx, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration:
        return None
    import yaml

    data = yaml.safe_load("\n".join(lines[1:end_idx])) or {}
    if not isinstance(data, dict):
        return None
    cx = data.get("cx")
    if not isinstance(cx, dict):
        return None
    value = cx.get("context")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _validate_context_name(name: str, *, source: str) -> str:
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"{source} must be a context name, not a path")
    if not _CONTEXT_NAME_RE.fullmatch(name):
        raise ValueError(f"{source} is not a valid context name: {name}")
    return name


def _slug_context_name(name: str) -> str | None:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not slug:
        return None
    return _validate_context_name(slug, source=f"manifest config.name '{name}'")


def _resolved_manifest_source(context: ContextEntry) -> Path | None:
    source = context.manifest.get("source")
    if not isinstance(source, str) or not source:
        return None
    path = Path(os.path.expanduser(source))
    if not path.is_absolute():
        path = context.target_dir / path
    return path.resolve(strict=False)


def _warn_context_subscription(message: str) -> None:
    print(f"context subscription: warning: {message}", file=sys.stderr)


def hydrate_contexts(
    names: list[str] | tuple[str, ...] | None = None,
    *,
    registry_path: str | os.PathLike[str] | None = None,
    status_path: str | os.PathLike[str] | None = None,
    overrides: HydrateOverrides | None = None,
    all_contexts: bool = False,
) -> list[ContextHydrationStatus]:
    contexts = load_context_registry(registry_path)
    requested_names = list(names or ())
    if all_contexts and requested_names:
        raise ValueError("context names cannot be combined with --all")
    if not all_contexts and not requested_names:
        raise ValueError("provide one or more context names, or pass --all")
    selected_names = list(contexts) if all_contexts else requested_names
    effective_overrides = overrides or HydrateOverrides()
    run_cache = HydrationRunCache()
    statuses: list[ContextHydrationStatus] = []
    previous_payload = _load_status_payload(status_path)
    previous_contexts = previous_payload.get("contexts")
    if not isinstance(previous_contexts, dict):
        previous_contexts = {}
    run = {
        "id": str(uuid.uuid4()),
        "state": "running",
        "pid": os.getpid(),
        "started_at": _now(),
        "selected": selected_names,
        "selected_count": len(selected_names),
        "completed_count": 0,
        "current_context": None,
        "work": {},
    }
    write_context_status([], status_path=status_path, run=run)

    log_progress("hydrate", "context", "total", count=len(selected_names))
    try:
        for name in selected_names:
            run["current_context"] = name
            run["updated_at"] = _now()
            write_context_status(statuses, status_path=status_path, run=run)
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
            else:
                with redirect_stderr(_ContextStderrPrefixer(context.name, sys.stderr)):
                    token = set_progress_context(context.name)
                    try:
                        status = _hydrate_one(
                            context,
                            effective_overrides,
                            run_cache=run_cache,
                            previous_status=previous_contexts.get(name),
                        )
                    finally:
                        reset_progress_context(token)
                statuses.append(status)
                progress_outcome = {
                    "failed": "failed",
                    "partial": "partial",
                }.get(status.result, "done")
                log_progress(
                    "hydrate",
                    "context",
                    progress_outcome,
                    target=name,
                    detail=status.reason or status.result,
                )
            run["completed_count"] = len(statuses)
            run["work"] = {
                selected: progress_counters(selected) for selected in selected_names
            }
            write_context_status(statuses, status_path=status_path, run=run)
    except BaseException:
        run["state"] = "interrupted"
        run["current_context"] = run.get("current_context")
        run["updated_at"] = _now()
        run["work"] = {
            selected: progress_counters(selected) for selected in selected_names
        }
        write_context_status(statuses, status_path=status_path, run=run)
        raise

    run["state"] = "complete"
    run["current_context"] = None
    run["completed_at"] = _now()
    run["updated_at"] = run["completed_at"]
    write_context_status(statuses, status_path=status_path, run=run)
    return statuses


def write_context_status(
    statuses: list[ContextHydrationStatus],
    *,
    status_path: str | os.PathLike[str] | None = None,
    run: dict[str, Any] | None = None,
) -> None:
    path = (
        Path(status_path).expanduser() if status_path else default_context_status_path()
    )
    previous: dict[str, Any] = {}
    previous = _load_status_payload(path)
    contexts = (
        previous.get("contexts") if isinstance(previous.get("contexts"), dict) else {}
    )
    contexts = dict(contexts)
    for status in statuses:
        contexts[status.name] = asdict(status)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "updated_at": _now(),
        "contexts": contexts,
    }
    if run is not None:
        payload["run"] = dict(run)
    _atomic_write_json(path, payload)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_status_payload(
    status_path: str | os.PathLike[str] | Path | None,
) -> dict[str, Any]:
    path = (
        Path(status_path).expanduser() if status_path else default_context_status_path()
    )
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


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
    origin = raw.get("origin", "registry")
    if not isinstance(origin, str) or not origin.strip():
        raise ValueError(f"Context '{name}' origin must be a non-empty string")
    designations = _parse_context_designations(name, raw.get("designations"))
    return ContextEntry(
        name=name,
        target_dir=Path(os.path.expanduser(target_dir)),
        manifest=manifest,
        replace=replace,
        designations=designations,
        context_dir=_optional_context_dir(raw, f"Context '{name}'"),
        origin=origin.strip(),
    )


def _parse_context_designations(name: str, raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"Context '{name}' designations must be a list")
    designations: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Context '{name}' designations[{index}] must be a non-empty string"
            )
        designation = value.strip()
        if not _CONTEXT_NAME_RE.fullmatch(designation):
            raise ValueError(
                f"Context '{name}' designation '{designation}' must contain only "
                "letters, numbers, '.', '_', or '-'"
            )
        if designation in designations:
            raise ValueError(
                f"Context '{name}' designation '{designation}' is duplicated"
            )
        designations.append(designation)
    return tuple(designations)


def _hydrate_one(
    context: ContextEntry,
    overrides: HydrateOverrides,
    *,
    resolving: tuple[Path, ...] = (),
    run_cache: HydrationRunCache | None = None,
    previous_status: Any = None,
) -> ContextHydrationStatus:
    target_dir = context.target_dir
    manifest_source = manifest_source_label(context)
    if (
        not target_dir.is_dir()
        and context.ensure_target_dir
        and not target_dir.exists()
    ):
        target_dir.mkdir(parents=True)
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
        fast_receipt = _current_hydration_receipt(
            context,
            overrides,
            previous_status,
        )
        if fast_receipt is not None:
            context_dir = previous_status.get("context_dir")
            log_progress(
                "hydrate",
                "receipt",
                "cache_hit",
                target=context.name,
            )
            return _status(
                name=context.name,
                target_dir=str(target_dir),
                manifest_source=manifest_source,
                context_dir=context_dir,
                result="up-to-date",
                reason=None,
                replace=context.replace,
                component_count=previous_status.get("component_count"),
                file_count=previous_status.get("file_count"),
                receipt=fast_receipt,
            )
        plan = _build_plan(
            context,
            overrides,
            resolving=resolving,
            run_cache=run_cache,
        )
        _reject_manifest_inside_context_root(context, plan.context_dir)
        linked_failure_reason = _linked_manifest_failure_reason(
            plan.linked_manifest_failures
        )
        receipt = _hydration_receipt(context, overrides, plan)
        existing_status = _prepare_existing_context(plan, context.replace)
        if existing_status is not None:
            outcome, reason = existing_status
            if outcome == "up-to-date" and linked_failure_reason:
                outcome = "partial"
                reason = linked_failure_reason
            return _status_from_result(
                context,
                HydrateResult(
                    context_dir=str(plan.context_dir),
                    component_count=plan.component_count,
                    file_count=_planned_file_count(plan),
                    manifest_written=plan.include_meta,
                ),
                outcome=outcome,
                reason=reason,
                manifest_source=manifest_source,
                receipt=receipt,
            )
        result = apply_hydration_plan(plan)
        return _status_from_result(
            context,
            result,
            outcome="partial" if linked_failure_reason else "hydrated",
            reason=linked_failure_reason,
            manifest_source=manifest_source,
            receipt=receipt,
        )
    except (OSError, ValueError) as exc:
        reason = str(exc)
        failed_component = _component_failure_name(reason)
        if failed_component:
            log_progress(
                "hydrate",
                "component",
                "failed",
                target=failed_component,
                detail=reason,
            )
        return _status(
            name=context.name,
            target_dir=str(target_dir),
            manifest_source=manifest_source,
            context_dir=None,
            result="failed",
            reason=reason,
            replace=context.replace,
        )


def _build_plan(
    context: ContextEntry,
    overrides: HydrateOverrides,
    *,
    resolving: tuple[Path, ...] = (),
    run_cache: HydrationRunCache | None = None,
):
    target_dir = context.target_dir.resolve()
    manifest = context.manifest
    cwd = str(target_dir)
    effective_overrides = _effective_context_overrides(context, overrides)
    if "source" in manifest:
        source = manifest["source"]
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"Context '{context.name}' manifest.source must be a string"
            )
        source_path = Path(os.path.expanduser(source))
        if not source_path.is_absolute():
            source_path = target_dir / source_path
        return build_hydration_plan(
            str(source_path),
            overrides=effective_overrides,
            cwd=cwd,
            _resolving=resolving,
            _run_cache=run_cache,
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
            overrides=effective_overrides,
            cwd=cwd,
            _resolving=resolving,
            _run_cache=run_cache,
        )
    data = manifest["data"]
    if not isinstance(data, dict):
        raise ValueError(f"Context '{context.name}' manifest.data must be a mapping")
    return build_hydration_plan_data(
        data,
        manifest_cwd=cwd,
        manifest_path=None,
        overrides=effective_overrides,
        cwd=cwd,
        _resolving=resolving,
        _run_cache=run_cache,
    )


def resolved_context_dir(context: ContextEntry) -> Path | None:
    if context.context_dir is None:
        return None
    path = Path(os.path.expanduser(context.context_dir))
    if not path.is_absolute():
        path = context.target_dir / path
    return path.resolve()


def _reject_manifest_inside_context_root(
    context: ContextEntry, context_dir: Path
) -> None:
    source = context.manifest.get("source")
    if not isinstance(source, str):
        return
    source_path = Path(os.path.expanduser(source))
    if not source_path.is_absolute():
        source_path = context.target_dir / source_path
    try:
        source_path.resolve().relative_to(context_dir.resolve())
    except ValueError:
        return
    raise ValueError(
        f"Context '{context.name}' manifest source is inside its context root: "
        f"{source_path.resolve()}"
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


def _component_failure_name(reason: str) -> str | None:
    prefix = "Component '"
    if not reason.startswith(prefix):
        return None
    remainder = reason[len(prefix) :]
    name, separator, _ = remainder.partition("'")
    if separator and name:
        return name
    return None


def _linked_manifest_failure_reason(failures) -> str | None:
    if not failures:
        return None
    first = failures[0]
    count = len(failures)
    noun = "linked manifest" if count == 1 else "linked manifests"
    suffix = "" if count == 1 else f", and {count - 1} more"
    return f"{count} {noun} failed: {first.manifest_path}: {first.reason}{suffix}"


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
    run_cache: HydrationRunCache | None = None,
) -> tuple[Path, str | None]:
    resolved = Path(manifest_path).expanduser().resolve()
    if resolved in resolving:
        chain = " -> ".join(str(p) for p in (*resolving, resolved))
        raise ValueError(f"Cycle detected while resolving linked manifests: {chain}")
    if not resolved.is_file():
        raise ValueError(f"Linked manifest not found: {resolved}")

    cache_key = (
        str(resolved),
        parent_overrides.use_cache,
        parent_overrides.cache_ttl.total_seconds()
        if parent_overrides.cache_ttl is not None
        else None,
        json.dumps(parent_overrides.plugin_overrides, sort_keys=True, default=str),
        parent_overrides.embedded_resolution,
    )
    can_reuse = bool(
        run_cache is not None
        and parent_overrides.use_cache is not False
        and not parent_overrides.refresh_cache
    )
    if can_reuse and cache_key in run_cache.linked_manifests:
        log_progress(
            "hydrate",
            "linked manifest",
            "coalesced",
            target=str(resolved),
        )
        return run_cache.linked_manifests[cache_key]

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
        status = _hydrate_one(
            entry,
            child_overrides,
            resolving=resolving + (resolved,),
            run_cache=run_cache,
        )
    finally:
        reset_progress_context(token)

    write_context_status([status], status_path=default_manifest_link_status_path())

    hydrated_dir = status.context_dir
    if status.result in ("hydrated", "up-to-date") and hydrated_dir:
        result = (Path(hydrated_dir), manifest_name)
        if can_reuse:
            run_cache.linked_manifests[cache_key] = result
        return result
    raise ValueError(
        f"Failed to hydrate linked manifest {resolved}: {status.reason or status.result}"
    )


def _planned_paths(plan) -> set[str]:
    written_paths = {
        path.relative_to(plan.context_dir).as_posix() for path, _ in plan.files_to_write
    }
    copied_paths = {
        path.relative_to(plan.context_dir).as_posix() for path, _ in plan.files_to_copy
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
    receipt: dict[str, Any] | None = None,
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
        receipt=receipt,
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
    receipt: dict[str, Any] | None = None,
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
        receipt=receipt,
    )


def _effective_context_overrides(
    context: ContextEntry,
    overrides: HydrateOverrides,
) -> HydrateOverrides:
    if overrides.context_dir is None and context.context_dir is not None:
        return replace(overrides, context_dir=context.context_dir)
    return overrides


def _context_input_fingerprint(context: ContextEntry) -> str:
    manifest = context.manifest
    if "source" in manifest:
        source = _resolved_manifest_source(context)
        if source is None or not source.is_file():
            payload = {"source": str(source) if source else None, "missing": True}
        else:
            payload = {
                "source": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
    else:
        payload = manifest
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _context_options_fingerprint(
    context: ContextEntry,
    overrides: HydrateOverrides,
) -> str:
    effective = _effective_context_overrides(context, overrides)
    payload = {
        "target_dir": str(context.target_dir.resolve(strict=False)),
        "replace": context.replace,
        "overrides": asdict(effective),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _hydration_receipt(
    context: ContextEntry, overrides: HydrateOverrides, plan
) -> dict[str, Any]:
    return {
        "version": 1,
        "input_fingerprint": _context_input_fingerprint(context),
        "options_fingerprint": _context_options_fingerprint(context, overrides),
        "local_inputs": plan.local_input_fingerprints,
        "freshness": plan.freshness_receipts,
        "provider_revisions": plan.provider_revisions,
        "enrichment": plan.enrichment_completeness,
        "outputs": _plan_output_receipts(plan),
        "planned_paths": sorted(_planned_paths(plan)),
        "created_at": _now(),
    }


def _current_hydration_receipt(
    context: ContextEntry,
    overrides: HydrateOverrides,
    previous_status: Any,
) -> dict[str, Any] | None:
    if not isinstance(previous_status, dict):
        return None
    if previous_status.get("result") not in {"hydrated", "up-to-date"}:
        return None
    receipt = previous_status.get("receipt")
    if not isinstance(receipt, dict) or receipt.get("version") != 1:
        return None
    if (
        overrides.use_cache is False
        or overrides.refresh_cache
        or get_refresh_cache()
        or get_refresh_media()
    ):
        return None
    context_dir = previous_status.get("context_dir")
    if not isinstance(context_dir, str) or not Path(context_dir).is_dir():
        return None
    if receipt.get("input_fingerprint") != _context_input_fingerprint(context):
        return None
    if receipt.get("options_fingerprint") != _context_options_fingerprint(
        context, overrides
    ):
        return None
    if not _local_inputs_are_current(receipt.get("local_inputs")):
        return None
    if not _outputs_are_current(Path(context_dir), receipt):
        return None
    if not _provider_revisions_are_current(receipt.get("provider_revisions")):
        return None
    enrichment = receipt.get("enrichment")
    if isinstance(enrichment, dict) and not all(
        value is True for value in enrichment.values()
    ):
        return None
    freshness = receipt.get("freshness")
    provider_revisions = receipt.get("provider_revisions")
    if provider_revisions and not freshness:
        return None
    if not _freshness_receipts_are_current(freshness, overrides):
        return None
    return receipt


def _plan_output_receipts(plan) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for path, content in plan.files_to_write:
        outputs.append(
            {
                "path": path.relative_to(plan.context_dir).as_posix(),
                "kind": "file",
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    for path, source in plan.files_to_copy:
        outputs.append(
            {
                "path": path.relative_to(plan.context_dir).as_posix(),
                "kind": "file",
                "sha256": _hash_path(source),
            }
        )
    for path, source in plan.files_to_symlink:
        outputs.append(
            {
                "path": path.relative_to(plan.context_dir).as_posix(),
                "kind": "symlink",
                "target": str(source.resolve(strict=False)),
            }
        )
    for path, source in plan.dirs_to_symlink:
        outputs.append(
            {
                "path": path.relative_to(plan.context_dir).as_posix(),
                "kind": "dir-symlink",
                "target": str(source.resolve(strict=False)),
            }
        )
    return sorted(outputs, key=lambda item: (item["path"], item["kind"]))


def _outputs_are_current(context_dir: Path, receipt: dict[str, Any]) -> bool:
    outputs = receipt.get("outputs")
    planned_paths = receipt.get("planned_paths")
    if not isinstance(outputs, list) or not isinstance(planned_paths, list):
        return False
    for output in outputs:
        if not isinstance(output, dict) or not isinstance(output.get("path"), str):
            return False
        path = context_dir / output["path"]
        kind = output.get("kind")
        if kind == "file":
            if not path.is_file() or path.is_symlink():
                return False
            if _hash_path(path) != output.get("sha256"):
                return False
        elif kind in {"symlink", "dir-symlink"}:
            if not path.is_symlink():
                return False
            if str(path.resolve(strict=False)) != output.get("target"):
                return False
        else:
            return False
    if not all(isinstance(path, str) for path in planned_paths):
        return False
    return not find_untracked_files(context_dir, planned_paths=set(planned_paths))


def _local_inputs_are_current(raw_inputs: Any) -> bool:
    if raw_inputs is None:
        return False
    if not isinstance(raw_inputs, list):
        return False
    for item in raw_inputs:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            return False
        path = Path(item["path"])
        if not path.is_file():
            return False
        stat_result = path.stat()
        if stat_result.st_size == item.get(
            "size"
        ) and stat_result.st_mtime_ns == item.get("mtime_ns"):
            continue
        if _hash_path(path) != item.get("sha256"):
            return False
    return True


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provider_revisions_are_current(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    from ..plugins.loader import get_loaded_plugins

    current = {
        plugin.name: plugin.revision or plugin.origin for plugin in get_loaded_plugins()
    }
    return all(current.get(name) == revision for name, revision in raw.items())


def _freshness_receipts_are_current(
    raw: Any,
    overrides: HydrateOverrides,
) -> bool:
    if raw is None:
        return False
    if not isinstance(raw, list):
        return False
    from ..plugins.resolve import probe_plugin_freshness

    for receipt in raw:
        if not isinstance(receipt, dict):
            return False
        provider = receipt.get("provider")
        if not isinstance(provider, str):
            return False
        result = probe_plugin_freshness(
            provider,
            receipt,
            overrides=overrides.plugin_overrides,
        )
        if result.get("state") != "fresh":
            log_progress(
                provider,
                "freshness probe",
                result.get("state", "unknown"),
                target=str(receipt.get("target") or receipt.get("source") or ""),
                detail=result.get("reason"),
            )
            return False
    return True


def manifest_source_label(context: ContextEntry) -> str:
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
