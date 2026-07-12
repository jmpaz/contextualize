from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote

from .contexts import ContextEntry, load_context_registry, manifest_source_label
from .manifest import coerce_mark_spec, mark_spec_items, normalize_components
from .source import ManifestSource, load_manifest_source, load_manifest_text


AUTHORED_EDITION_SCHEMA_VERSION = 1
_MEMBER_ROLES = {"files": "material", "repos": "repository", "manifests": "context"}
_EXTERNAL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_VOICE_RE = re.compile(
    r"^(?:store:)?(?P<key>voice/.+?\.m4a)(?:@(?P<span>[^#]+))?$"
)


@dataclass(frozen=True)
class AuthoredDiagnostic:
    code: str
    message: str
    severity: str = "error"
    source_path: str | None = None
    line: int | None = None
    position_key: str | None = None
    portal_key: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "code": self.code,
                "message": self.message,
                "severity": self.severity,
                "sourcePath": self.source_path,
                "line": self.line,
                "positionKey": self.position_key,
                "portalKey": self.portal_key,
                "details": self.details or None,
            }
        )


@dataclass(frozen=True)
class AuthoredPosition:
    id: str
    stable_id: str
    locators: tuple[str, ...]
    key: str
    parent_id: str | None
    child_ids: tuple[str, ...]
    portal_ids: tuple[str, ...]
    kind: str
    name: str | None
    order: int
    disabled: bool
    source_path: str | None
    line_start: int | None
    line_end: int | None
    framing: tuple[dict[str, Any], ...]
    options: dict[str, Any]
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "id": self.id,
                "stableId": self.stable_id,
                "locators": list(self.locators),
                "key": self.key,
                "parentId": self.parent_id,
                "childIds": list(self.child_ids),
                "portalIds": list(self.portal_ids),
                "kind": self.kind,
                "name": self.name,
                "order": self.order,
                "disabled": self.disabled,
                "sourcePath": self.source_path,
                "lineStart": self.line_start,
                "lineEnd": self.line_end,
                "framing": list(self.framing),
                "options": self.options,
                "raw": self.raw,
            }
        )


@dataclass(frozen=True)
class AuthoredPortal:
    id: str
    stable_id: str
    key: str
    position_id: str
    position_stable_id: str
    position_locators: tuple[str, ...]
    role: str
    order: int
    disabled: bool
    authored_target: str | None
    target_aliases: tuple[str, ...]
    target_position_id: str | None
    source_path: str | None
    line_start: int | None
    line_end: int | None
    comment: dict[str, Any] | None
    inline_comment: str | None
    options: dict[str, Any]
    ranges: tuple[dict[str, Any], ...]
    dynamic: dict[str, Any] | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "id": self.id,
                "stableId": self.stable_id,
                "key": self.key,
                "positionId": self.position_id,
                "positionStableId": self.position_stable_id,
                "role": self.role,
                "order": self.order,
                "disabled": self.disabled,
                "authoredTarget": self.authored_target,
                "targetAliases": list(self.target_aliases),
                "targetPositionId": self.target_position_id,
                "reverse": {
                    "positionId": self.position_id,
                    "positionStableId": self.position_stable_id,
                    "positionLocators": list(self.position_locators),
                    "sourcePath": self.source_path,
                    "lineStart": self.line_start,
                    "lineEnd": self.line_end,
                    "role": self.role,
                    "order": self.order,
                },
                "sourcePath": self.source_path,
                "lineStart": self.line_start,
                "lineEnd": self.line_end,
                "comment": self.comment,
                "inlineComment": self.inline_comment,
                "options": self.options,
                "ranges": list(self.ranges),
                "dynamic": self.dynamic,
                "status": self.status,
            }
        )


@dataclass(frozen=True)
class AuthoredEdition:
    context: dict[str, Any]
    positions: tuple[AuthoredPosition, ...]
    portals: tuple[AuthoredPortal, ...]
    diagnostics: tuple[AuthoredDiagnostic, ...]
    schema_version: int = AUTHORED_EDITION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "context": self.context,
            "positions": [position.to_dict() for position in self.positions],
            "portals": [portal.to_dict() for portal in self.portals],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


DynamicResolver = Callable[[str, Mapping[str, Any]], Mapping[str, Any] | None]


def compile_authored_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    context_name: str | None = None,
    compiled_at: str | None = None,
    dynamic_resolver: DynamicResolver | None = None,
) -> AuthoredEdition:
    source = load_manifest_source(manifest_path)
    name = context_name or _manifest_name(source) or Path(manifest_path).stem
    return _EditionCompiler(
        name=name,
        root_source=source,
        source_label=str(Path(manifest_path).expanduser().resolve()),
        authority="file-backed-authored",
        compiled_at=compiled_at or _now(),
        dynamic_resolver=dynamic_resolver,
    ).compile()


def compile_authored_context(
    context: ContextEntry,
    *,
    compiled_at: str | None = None,
    dynamic_resolver: DynamicResolver | None = None,
) -> AuthoredEdition:
    source, authority = _source_for_context(context)
    return _EditionCompiler(
        name=context.name,
        root_source=source,
        source_label=manifest_source_label(context),
        authority=authority,
        compiled_at=compiled_at or _now(),
        dynamic_resolver=dynamic_resolver,
    ).compile()


def compile_authored_registry(
    registry_path: str | os.PathLike[str] | None = None,
    *,
    names: Iterable[str] | None = None,
    compiled_at: str | None = None,
    dynamic_resolver: DynamicResolver | None = None,
) -> list[AuthoredEdition]:
    registry = load_context_registry(registry_path)
    selected = list(names) if names is not None else sorted(registry)
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise ValueError(f"Unknown context(s): {', '.join(unknown)}")
    edition_time = compiled_at or _now()
    return [
        compile_authored_context(
            registry[name],
            compiled_at=edition_time,
            dynamic_resolver=dynamic_resolver,
        )
        for name in selected
    ]


class _EditionCompiler:
    def __init__(
        self,
        *,
        name: str,
        root_source: ManifestSource,
        source_label: str,
        authority: str,
        compiled_at: str,
        dynamic_resolver: DynamicResolver | None,
    ) -> None:
        self.name = name
        self.root_source = root_source
        self.source_label = source_label
        self.authority = authority
        self.compiled_at = compiled_at
        self.dynamic_resolver = dynamic_resolver
        self.position_records: list[dict[str, Any]] = []
        self.portal_records: list[dict[str, Any]] = []
        self.diagnostics: list[AuthoredDiagnostic] = []
        self.sources: list[str] = []

    def compile(self) -> AuthoredEdition:
        root_key = "root"
        self._compile_source(
            self.root_source,
            parent_key=None,
            root_key=root_key,
            order=0,
            stack=(),
            locator_base=self.name,
        )
        self._diagnose_locator_conflicts()
        fingerprint = _fingerprint_value({
            "schemaVersion": AUTHORED_EDITION_SCHEMA_VERSION,
            "context": self.name,
            "authority": self.authority,
            "positions": self.position_records,
            "portals": self.portal_records,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        })
        edition = hashlib.sha256(_canonical_json(fingerprint).encode("utf-8")).hexdigest()[:24]
        position_ids = {
            record["key"]: _position_id(self.name, edition, record["key"])
            for record in self.position_records
        }
        position_stable_ids = {
            record["key"]: _position_stable_id(self.name, record["key"])
            for record in self.position_records
        }
        portal_ids = {
            record["key"]: _portal_id(self.name, edition, record["key"])
            for record in self.portal_records
        }
        portal_stable_ids = {
            record["key"]: _portal_stable_id(self.name, record["key"])
            for record in self.portal_records
        }
        positions = tuple(
            AuthoredPosition(
                id=position_ids[record["key"]],
                stable_id=position_stable_ids[record["key"]],
                locators=tuple(record["locators"]),
                key=record["key"],
                parent_id=position_ids.get(record["parent_key"]),
                child_ids=tuple(position_ids[key] for key in record["child_keys"]),
                portal_ids=tuple(portal_ids[key] for key in record["portal_keys"]),
                kind=record["kind"],
                name=record["name"],
                order=record["order"],
                disabled=record["disabled"],
                source_path=record["source_path"],
                line_start=record["line_start"],
                line_end=record["line_end"],
                framing=tuple(record["framing"]),
                options=record["options"],
                raw=record["raw"],
            )
            for record in self.position_records
        )
        portals = tuple(
            AuthoredPortal(
                id=portal_ids[record["key"]],
                stable_id=portal_stable_ids[record["key"]],
                key=record["key"],
                position_id=position_ids[record["position_key"]],
                position_stable_id=position_stable_ids[record["position_key"]],
                position_locators=tuple(
                    next(
                        item["locators"]
                        for item in self.position_records
                        if item["key"] == record["position_key"]
                    )
                ),
                role=record["role"],
                order=record["order"],
                disabled=record["disabled"],
                authored_target=record["authored_target"],
                target_aliases=tuple(record["target_aliases"]),
                target_position_id=position_ids.get(record["target_position_key"]),
                source_path=record["source_path"],
                line_start=record["line_start"],
                line_end=record["line_end"],
                comment=record["comment"],
                inline_comment=record["inline_comment"],
                options=record["options"],
                ranges=tuple(record["ranges"]),
                dynamic=record["dynamic"],
                status=record["status"],
            )
            for record in self.portal_records
        )
        context = {
            "name": self.name,
            "source": self.source_label,
            "edition": edition,
            "compiledAt": self.compiled_at,
            "authority": self.authority,
            "hydration": "optional-projection",
            "rootPositionId": position_ids[root_key],
            "sources": self.sources,
        }
        return AuthoredEdition(
            context=context,
            positions=positions,
            portals=portals,
            diagnostics=tuple(self.diagnostics),
        )

    def _diagnose_locator_conflicts(self) -> None:
        positions_by_locator: dict[str, list[str]] = {}
        for record in self.position_records:
            for locator in record["locators"]:
                positions_by_locator.setdefault(locator, []).append(record["key"])
        for locator, keys in positions_by_locator.items():
            if len(keys) < 2:
                continue
            self.diagnostics.append(
                AuthoredDiagnostic(
                    code="authored-locator-ambiguous",
                    message=f"Authored locator resolves to {len(keys)} positions: {locator}",
                    severity="warning",
                    details={"locator": locator, "positionKeys": keys},
                )
            )

    def _compile_source(
        self,
        source: ManifestSource,
        *,
        parent_key: str | None,
        root_key: str,
        order: int,
        stack: tuple[Path, ...],
        locator_base: str,
    ) -> str:
        source_path = source.manifest_path
        resolved_path = Path(source_path).resolve() if source_path else None
        active_stack = stack + ((resolved_path,) if resolved_path else ())
        source_identity = source_path or f"inline:{root_key}"
        if source_identity not in self.sources:
            self.sources.append(source_identity)
        cfg = source.data.get("config") or {}
        root_record = {
            "key": root_key,
            "parent_key": parent_key,
            "child_keys": [],
            "portal_keys": [],
            "kind": "manifest",
            "name": _manifest_name(source),
            "order": order,
            "disabled": False,
            "source_path": source_path,
            "line_start": source.source_format.line if source.source_format else None,
            "line_end": None,
            "framing": _framing(cfg, None),
            "options": _without(cfg, {"text", "prefix", "suffix", "comment"}),
            "raw": None,
            "locators": [self.name] if parent_key is None else [f"{locator_base}/~manifest"],
            "child_locator_base": locator_base,
        }
        self.position_records.append(root_record)
        components = source.data.get("components")
        if not isinstance(components, list):
            self.diagnostics.append(
                AuthoredDiagnostic(
                    code="manifest-components-invalid",
                    message="Manifest components must be a list",
                    source_path=source_path,
                    position_key=root_key,
                )
            )
            return root_key
        try:
            effective_leaves = iter(normalize_components(components))
        except ValueError as exc:
            effective_leaves = iter(())
            self.diagnostics.append(
                AuthoredDiagnostic(
                    code="manifest-grammar-invalid",
                    message=str(exc),
                    source_path=source_path,
                    position_key=root_key,
                )
            )
        outline = source.source_format.outline if source.source_format else []
        root_record["child_keys"] = self._compile_component_list(
            raw_components=components,
            outline_nodes=outline,
            source=source,
            parent_key=root_key,
            effective_leaves=effective_leaves,
            stack=active_stack,
        )
        return root_key

    def _compile_component_list(
        self,
        *,
        raw_components: list[Any],
        outline_nodes: list[dict[str, Any]],
        source: ManifestSource,
        parent_key: str,
        effective_leaves: Iterable[dict[str, Any]],
        stack: tuple[Path, ...],
    ) -> list[str]:
        nodes = outline_nodes or [
            _synthetic_node(entry, index)
            for index, entry in enumerate(raw_components)
        ]
        active_raw = iter(raw_components)
        child_keys: list[str] = []
        for ordinal, node in enumerate(nodes):
            raw = None if node.get("disabled") else next(active_raw, None)
            key = f"{parent_key}/p{ordinal:04d}"
            kind = node.get("kind") or _raw_kind(raw)
            effective = None
            if not node.get("disabled") and kind != "group":
                effective = next(effective_leaves, raw if isinstance(raw, dict) else {})
            options_source = effective if effective is not None else raw
            options = _position_options(options_source, kind)
            record = {
                "key": key,
                "parent_key": parent_key,
                "child_keys": [],
                "portal_keys": [],
                "kind": kind,
                "name": node.get("name") or _raw_name(raw) or (effective or {}).get("name"),
                "order": int(node.get("order", ordinal)),
                "disabled": bool(node.get("disabled")),
                "source_path": source.manifest_path,
                "line_start": node.get("line_start"),
                "line_end": node.get("line_end"),
                "framing": _framing(options_source, node),
                "options": options,
                "raw": node.get("raw"),
                "locators": [
                    self._child_locator(
                        parent_key, node.get("name") or _raw_name(raw), ordinal
                    )
                ],
                "child_locator_base": None,
            }
            self.position_records.append(record)
            child_keys.append(key)
            if record["disabled"]:
                continue
            if kind == "group":
                nested = raw.get("components") if isinstance(raw, dict) else []
                if not isinstance(nested, list):
                    nested = []
                record["child_keys"] = self._compile_component_list(
                    raw_components=nested,
                    outline_nodes=node.get("children") or [],
                    source=source,
                    parent_key=key,
                    effective_leaves=effective_leaves,
                    stack=stack,
                )
                continue
            semantic = (
                effective
                if isinstance(effective, dict)
                else (raw if isinstance(raw, dict) else {})
            )
            record["portal_keys"] = self._compile_members(
                semantic=semantic,
                outline_members=node.get("members") or {},
                source=source,
                position_key=key,
                stack=stack,
            )
        return child_keys

    def _compile_members(
        self,
        *,
        semantic: dict[str, Any],
        outline_members: dict[str, list[dict[str, Any]]],
        source: ManifestSource,
        position_key: str,
        stack: tuple[Path, ...],
    ) -> list[str]:
        portal_keys: list[str] = []
        portal_ordinal = 0
        for member_key, role in _MEMBER_ROLES.items():
            specs = semantic.get(member_key)
            if not isinstance(specs, list):
                specs = []
            outlines = outline_members.get(member_key) or [
                _synthetic_member(index) for index in range(len(specs))
            ]
            active_specs = iter(specs)
            for member_outline in outlines:
                disabled = bool(member_outline.get("disabled"))
                spec = None if disabled else next(active_specs, None)
                expanded_specs = _expand_member_spec(member_key, spec)
                for expanded_spec in expanded_specs:
                    portal_key = f"{position_key}/m{portal_ordinal:04d}"
                    portal_ordinal += 1
                    target, options = _member_target_and_options(expanded_spec)
                    target_aliases: list[str] = []
                    status = "disabled" if disabled else "unresolved"
                    target_position_key = None
                    ranges = _member_ranges(expanded_spec, member_outline, target)
                    for authored_range in ranges:
                        problem = authored_range.get("problem")
                        if not problem:
                            continue
                        self.diagnostics.append(
                            AuthoredDiagnostic(
                                code=problem,
                                message=f"Invalid authored range: {problem}",
                                source_path=source.manifest_path,
                                line=authored_range.get("lineStart"),
                                position_key=position_key,
                                portal_key=portal_key,
                            )
                        )
                    dynamic = None
                    if not disabled and target is not None:
                        target_aliases, status = self._resolve_aliases(
                            target, source, role, portal_key, member_outline
                        )
                        target_aliases.extend(
                            alias
                            for alias in _voice_span_aliases(target, ranges)
                            if alias not in target_aliases
                        )
                        if _is_dynamic(target, options):
                            dynamic = self._dynamic_record(target, options)
                            status = "dynamic"
                        if role == "context":
                            target_position_key, status = self._compile_manifest_portal(
                                target=target,
                                source=source,
                                portal_key=portal_key,
                                position_key=position_key,
                                order=int(member_outline.get("order", portal_ordinal - 1)),
                                stack=stack,
                                status=status,
                            )
                    record = {
                        "key": portal_key,
                        "position_key": position_key,
                        "role": role,
                        "order": int(member_outline.get("order", portal_ordinal - 1)),
                        "disabled": disabled,
                        "authored_target": target,
                        "target_aliases": target_aliases,
                        "target_position_key": target_position_key,
                        "source_path": source.manifest_path,
                        "line_start": member_outline.get("line_start"),
                        "line_end": member_outline.get("line_end"),
                        "comment": member_outline.get("comment"),
                        "inline_comment": member_outline.get("inline_comment"),
                        "options": (
                            options
                            if expanded_spec is not None
                            else {"raw": member_outline.get("raw")}
                        ),
                        "ranges": ranges,
                        "dynamic": dynamic,
                        "status": status,
                    }
                    self.portal_records.append(record)
                    portal_keys.append(portal_key)
        return portal_keys

    def _locator_for(self, position_key: str) -> str:
        record = next(item for item in self.position_records if item["key"] == position_key)
        return record["locators"][0]

    def _child_locator(self, parent_key: str, name: str | None, ordinal: int) -> str:
        parent_record = next(item for item in self.position_records if item["key"] == parent_key)
        parent = parent_record.get("child_locator_base") or self._locator_for(parent_key)
        segment = name or f"component-{ordinal + 1:03d}"
        if parent_record["kind"] == "manifest" and parent.rsplit("/", 1)[-1] == segment:
            return parent
        separator = ":" if parent_key == "root" else "/"
        return f"{parent}{separator}{segment}"

    def _resolve_aliases(
        self,
        target: str,
        source: ManifestSource,
        role: str,
        portal_key: str,
        outline: dict[str, Any],
    ) -> tuple[list[str], str]:
        from ..git.target import parse_git_target
        from ..utils import brace_expand

        aliases = _target_aliases(target)
        if (
            _EXTERNAL_RE.match(target)
            or target.startswith("//")
            or parse_git_target(target)
        ):
            return aliases, "external"
        base = _manifest_base_dir(source)
        authored_paths = (
            brace_expand(target) if "{" in target and "}" in target else [target]
        )
        resolved_paths = []
        for authored_path in authored_paths:
            candidate = Path(os.path.expanduser(authored_path))
            resolved_paths.append(
                (candidate if candidate.is_absolute() else base / candidate).resolve()
            )
        missing = [path for path in resolved_paths if not path.exists()]
        if not missing:
            for resolved in resolved_paths:
                absolute = str(resolved)
                if absolute not in aliases:
                    aliases.append(absolute)
            return aliases, "resolved"
        self.diagnostics.append(
            AuthoredDiagnostic(
                code="reference-unresolved",
                message=f"Authored {role} reference does not exist: {missing[0]}",
                source_path=source.manifest_path,
                line=outline.get("line_start"),
                portal_key=portal_key,
                details={
                    "authoredTarget": target,
                    "missing": [str(path) for path in missing],
                },
            )
        )
        return aliases, "unresolved"

    def _compile_manifest_portal(
        self,
        *,
        target: str,
        source: ManifestSource,
        portal_key: str,
        position_key: str,
        order: int,
        stack: tuple[Path, ...],
        status: str,
    ) -> tuple[str | None, str]:
        if _EXTERNAL_RE.match(target):
            self.diagnostics.append(
                AuthoredDiagnostic(
                    code="included-manifest-unsupported",
                    message=f"Included manifest must be a local file: {target}",
                    source_path=source.manifest_path,
                    position_key=position_key,
                    portal_key=portal_key,
                )
            )
            return None, "unresolved"
        candidate = Path(os.path.expanduser(target))
        resolved = candidate if candidate.is_absolute() else Path(source.manifest_cwd) / candidate
        resolved = resolved.resolve()
        if not resolved.is_file():
            return None, "unresolved"
        if resolved in stack:
            chain = [str(path) for path in (*stack, resolved)]
            self.diagnostics.append(
                AuthoredDiagnostic(
                    code="included-manifest-cycle",
                    message=f"Included manifest cycle: {' -> '.join(chain)}",
                    source_path=source.manifest_path,
                    position_key=position_key,
                    portal_key=portal_key,
                    details={"chain": chain},
                )
            )
            return None, "cycle"
        child_key = f"{position_key}/include-{order:04d}"
        try:
            included = load_manifest_source(resolved)
        except (OSError, ValueError) as exc:
            self.diagnostics.append(
                AuthoredDiagnostic(
                    code="included-manifest-invalid",
                    message=str(exc),
                    source_path=source.manifest_path,
                    position_key=position_key,
                    portal_key=portal_key,
                    details={"path": str(resolved)},
                )
            )
            return None, "unresolved"
        self._compile_source(
            included,
            parent_key=position_key,
            root_key=child_key,
            order=order,
            stack=stack,
            locator_base=f"{self._locator_for(position_key)}/{resolved.stem}",
        )
        position = next(record for record in self.position_records if record["key"] == position_key)
        if child_key not in position["child_keys"]:
            position["child_keys"].append(child_key)
        return child_key, "resolved"

    def _dynamic_record(self, target: str, options: dict[str, Any]) -> dict[str, Any]:
        request = {"target": target, "options": options}
        resolved = self.dynamic_resolver(target, options) if self.dynamic_resolver else None
        if resolved is None:
            coverage = {
                "state": "unacquired",
                "exact": False,
                "missing": ["provider acquisition is outside authored compilation"],
            }
            members: list[Any] = []
        else:
            coverage = dict(resolved.get("coverage") or {})
            coverage.setdefault("state", "partial")
            coverage.setdefault("exact", False)
            members = list(resolved.get("members") or [])
        return {
            "query": request,
            "editionTime": self.compiled_at,
            "coverage": coverage,
            "members": members,
        }


def _source_for_context(context: ContextEntry) -> tuple[ManifestSource, str]:
    manifest = context.manifest
    if "source" in manifest:
        raw = manifest["source"]
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"Context '{context.name}' manifest.source must be a string")
        path = Path(os.path.expanduser(raw))
        if not path.is_absolute():
            path = context.target_dir / path
        return load_manifest_source(path), "file-backed-authored"
    if "text" in manifest:
        text = manifest["text"]
        if not isinstance(text, str):
            raise ValueError(f"Context '{context.name}' manifest.text must be a string")
        data = load_manifest_text(text, source_label=f"context '{context.name}'")
        return (
            ManifestSource(
                data=data,
                manifest_cwd=str(context.target_dir),
                manifest_path=None,
            ),
            "registry-inline-authored",
        )
    data = manifest.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"Context '{context.name}' manifest.data must be a mapping")
    return (
        ManifestSource(
            data=data,
            manifest_cwd=str(context.target_dir),
            manifest_path=None,
        ),
        "registry-inline-authored",
    )


def _manifest_name(source: ManifestSource) -> str | None:
    cfg = source.data.get("config")
    if not isinstance(cfg, dict):
        return None
    name = cfg.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else None


def _manifest_base_dir(source: ManifestSource) -> Path:
    cfg = source.data.get("config")
    if isinstance(cfg, dict) and "root" in cfg:
        return Path(os.path.expanduser(cfg.get("root") or "~")).resolve()
    return Path(source.manifest_cwd).resolve()


def _raw_kind(entry: Any) -> str:
    if isinstance(entry, dict) and "group" in entry:
        return "group"
    if isinstance(entry, dict) and "set" in entry:
        return "set"
    return "component"


def _raw_name(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    for key in ("group", "set", "name"):
        value = entry.get(key)
        if isinstance(value, str):
            return value
    return None


def _synthetic_node(entry: Any, order: int) -> dict[str, Any]:
    raw = entry if isinstance(entry, dict) else {}
    return {
        "kind": _raw_kind(raw),
        "name": _raw_name(raw),
        "order": order,
        "disabled": False,
        "children": [],
        "members": {},
    }


def _synthetic_member(order: int) -> dict[str, Any]:
    return {"order": order, "disabled": False}


def _position_options(value: Any, kind: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    omitted = {"components", "files", "repos", "manifests", "text", "prefix", "suffix", "comment"}
    if kind == "group":
        omitted.add("group")
    elif kind == "set":
        omitted.add("set")
    else:
        omitted.add("name")
    return _without(value, omitted)


def _framing(value: Any, outline: dict[str, Any] | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if outline:
        comment = outline.get("comment")
        if isinstance(comment, dict) and comment.get("text"):
            result.append({"kind": "source-comment", **comment})
        inline = outline.get("inline_comment")
        if inline:
            result.append({"kind": "inline-comment", "text": inline})
    if isinstance(value, dict):
        for key in ("comment", "prefix", "text", "suffix"):
            text = value.get(key)
            if isinstance(text, str) and text:
                result.append({"kind": key, "text": text})
    return result


def _member_target_and_options(spec: Any) -> tuple[str | None, dict[str, Any]]:
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict):
        for key in ("path", "target", "url"):
            target = spec.get(key)
            if isinstance(target, str):
                return target, _without(spec, {key})
        return None, dict(spec)
    if spec is None:
        return None, {}
    return str(spec), {}


def _expand_member_spec(member_key: str, spec: Any) -> list[Any]:
    if member_key != "repos" or not isinstance(spec, dict):
        return [spec]
    items = spec.get("items")
    if not isinstance(items, list):
        return [spec]
    collection = _without(spec, {"items"})
    expanded = []
    for order, item in enumerate(items):
        if isinstance(item, dict):
            expanded.append({**item, "collection": {**collection, "order": order}})
        else:
            expanded.append(
                {
                    "target": item,
                    "collection": {**collection, "order": order},
                }
            )
    return expanded


def _member_ranges(
    spec: Any,
    outline: dict[str, Any],
    target: str | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    voice = _VOICE_RE.match(target or "")
    if voice and voice.group("span"):
        result.append(_coerced_range(coerce_mark_spec({"at": voice.group("span")}), "path"))
    if not isinstance(spec, dict):
        return result
    if "range" in spec:
        result.append({"kind": "range", "value": spec["range"]})
    outline_marks = outline.get("marks") or []
    raw_marks = mark_spec_items(spec.get("marks"))
    raw_iter = iter(raw_marks)
    mark_outlines = outline_marks or [
        _synthetic_member(index) for index in range(len(raw_marks))
    ]
    for index, mark_outline in enumerate(mark_outlines):
        disabled = bool(mark_outline.get("disabled"))
        raw = None if disabled else next(raw_iter, None)
        coerced = coerce_mark_spec(raw) if raw is not None else None
        result.append(_coerced_range(
            coerced,
            "mark",
            order=int(mark_outline.get("order", index)),
            disabled=disabled,
            outline=mark_outline,
        ))
    return result


def _coerced_range(
    coerced: dict[str, Any] | None,
    origin: str,
    *,
    order: int = 0,
    disabled: bool = False,
    outline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outline = outline or {}
    return _omit_none(
        {
            "kind": "voice-span",
            "origin": origin,
            "order": order,
            "disabled": disabled,
            "authored": coerced.get("authored") if coerced else None,
            "startSeconds": coerced.get("start_seconds") if coerced else None,
            "endSeconds": coerced.get("end_seconds") if coerced else None,
            "quote": coerced.get("quote") if coerced else None,
            "refs": coerced.get("refs") if coerced else None,
            "problem": coerced.get("problem") if coerced else None,
            "comment": outline.get("comment"),
            "inlineComment": outline.get("inline_comment"),
            "lineStart": outline.get("line_start"),
            "lineEnd": outline.get("line_end"),
            "raw": outline.get("raw") if disabled else None,
        }
    )


def _target_aliases(target: str) -> list[str]:
    aliases = [target]
    voice = _VOICE_RE.match(target)
    if voice:
        key = voice.group("key")
        for alias in (key, f"store:{key}"):
            if alias not in aliases:
                aliases.append(alias)
        span = voice.group("span")
        if span:
            ranged = coerce_mark_spec({"at": span})
            start = ranged.get("start_seconds")
            end = ranged.get("end_seconds")
            if isinstance(start, (int, float)):
                fragment = f"t={_seconds_text(start)}"
                if isinstance(end, (int, float)):
                    fragment += f",{_seconds_text(end)}"
                aliases.append(f"store:{key}#{fragment}")
    return aliases


def _voice_span_aliases(target: str, ranges: list[dict[str, Any]]) -> list[str]:
    voice = _VOICE_RE.match(target)
    if not voice:
        return []
    key = voice.group("key")
    aliases: list[str] = []
    for item in ranges:
        if item.get("kind") != "voice-span" or item.get("disabled"):
            continue
        start = item.get("startSeconds")
        if not isinstance(start, (int, float)):
            continue
        end = item.get("endSeconds")
        fragment = f"t={_seconds_text(start)}"
        if isinstance(end, (int, float)):
            fragment += f",{_seconds_text(end)}"
        authored = item.get("authored")
        aliases.append(f"store:{key}#{fragment}")
        if isinstance(authored, str) and authored:
            aliases.append(f"{key}@{authored}")
    return aliases


def _seconds_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _is_dynamic(target: str, options: dict[str, Any]) -> bool:
    return bool(
        options.get("dynamic")
        or options.get("query")
        or target.startswith("store:source:")
        or ("?" in target and _EXTERNAL_RE.match(target))
    )


def _without(value: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in keys and not key.startswith("__")
    }


def _position_id(context: str, edition: str, key: str) -> str:
    return f"ctx://context/{quote(context, safe='')}@{edition}/{quote(key, safe='/')}"


def _position_stable_id(context: str, key: str) -> str:
    return f"ctx://context/{quote(context, safe='')}/{quote(key, safe='/')}"


def _portal_id(context: str, edition: str, key: str) -> str:
    return f"ctx://authored-portal/{quote(context, safe='')}@{edition}/{quote(key, safe='/')}"


def _portal_stable_id(context: str, key: str) -> str:
    return f"ctx://authored-portal/{quote(context, safe='')}/{quote(key, safe='/')}"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _fingerprint_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _fingerprint_value(item)
            for key, item in value.items()
            if key not in {"compiledAt", "editionTime"}
        }
    if isinstance(value, list):
        return [_fingerprint_value(item) for item in value]
    return value


def _omit_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
