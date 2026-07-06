"""Context registry hydration support."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
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
from ..utils import read_config
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
_CONTEXT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


@dataclass(frozen=True)
class ContextEntry:
    name: str
    target_dir: Path
    manifest: dict[str, Any]
    replace: str
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
        name: _parse_context_entry(name, value)
        for name, value in contexts.items()
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
    }


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
                ensure_target_dir=True,
            )
        )
    return contexts


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
        raise ValueError("zk is required for contexts.subscriptions source: zk") from exc
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
    manifest_source = manifest_source_label(context)
    if not target_dir.is_dir() and context.ensure_target_dir and not target_dir.exists():
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
        plan = _build_plan(context, overrides, resolving=resolving)
        linked_failure_reason = _linked_manifest_failure_reason(
            plan.linked_manifest_failures
        )
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
            )
        result = apply_hydration_plan(plan)
        return _status_from_result(
            context,
            result,
            outcome="partial" if linked_failure_reason else "hydrated",
            reason=linked_failure_reason,
            manifest_source=manifest_source,
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


def _component_failure_name(reason: str) -> str | None:
    prefix = "Component '"
    if not reason.startswith(prefix):
        return None
    remainder = reason[len(prefix):]
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
