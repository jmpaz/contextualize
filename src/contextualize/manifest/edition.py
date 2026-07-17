from __future__ import annotations

import hashlib
import json
import os
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
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
_PORTAL_OPTION_EXCLUSIONS = {
    "comment",
    "description",
    "inlineComment",
    "label",
    "marks",
    "prefix",
    "quote",
    "range",
    "raw",
    "suffix",
    "text",
    "title",
}
_PROSE_KEYS = (
    "comment",
    "description",
    "label",
    "prefix",
    "quote",
    "suffix",
    "text",
    "title",
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
QuoteResolver = Callable[[str], Mapping[str, Any] | None]


def compile_authored_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    context_name: str | None = None,
    compiled_at: str | None = None,
    dynamic_resolver: DynamicResolver | None = None,
    quote_resolver: QuoteResolver | None = None,
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
        quote_resolver=quote_resolver,
    ).compile()


def compile_authored_context(
    context: ContextEntry,
    *,
    compiled_at: str | None = None,
    dynamic_resolver: DynamicResolver | None = None,
    quote_resolver: QuoteResolver | None = None,
) -> AuthoredEdition:
    source, authority = _source_for_context(context)
    return _EditionCompiler(
        name=context.name,
        root_source=source,
        source_label=manifest_source_label(context),
        authority=authority,
        compiled_at=compiled_at or _now(),
        dynamic_resolver=dynamic_resolver,
        quote_resolver=quote_resolver,
    ).compile()


def compile_authored_registry(
    registry_path: str | os.PathLike[str] | None = None,
    *,
    names: Iterable[str] | None = None,
    compiled_at: str | None = None,
    dynamic_resolver: DynamicResolver | None = None,
    quote_resolver: QuoteResolver | None = None,
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
            quote_resolver=quote_resolver,
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
        quote_resolver: QuoteResolver | None,
    ) -> None:
        self.name = name
        self.root_source = root_source
        self.source_label = source_label
        self.authority = authority
        self.compiled_at = compiled_at
        self.dynamic_resolver = dynamic_resolver
        self.quote_resolver = quote_resolver
        self.quote_representations: dict[str, Mapping[str, Any] | None] = {}
        self.position_records: list[dict[str, Any]] = []
        self.portal_records: list[dict[str, Any]] = []
        self.diagnostics: list[AuthoredDiagnostic] = []
        self.sources: list[str] = []
        self.position_stable_identity_counts: dict[str, int] = {}
        self.portal_stable_identity_counts: dict[str, int] = {}
        self.position_stable_identities: set[str] = set()
        self.portal_stable_identities: set[str] = set()

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
        self._assign_portal_stable_identities()
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
            record["key"]: _position_stable_id(
                self.name, record["stable_identity"]
            )
            for record in self.position_records
        }
        portal_ids = {
            record["key"]: _portal_id(self.name, edition, record["key"])
            for record in self.portal_records
        }
        portal_stable_ids = {
            record["key"]: _portal_stable_id(
                self.name, record["stable_identity"]
            )
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
            "framing": _framing(cfg, None, lead_text=_manifest_lead_text(source)),
            "options": _without(cfg, {"text", "prefix", "suffix", "comment"}),
            "raw": None,
            "locators": [self.name] if parent_key is None else [f"{locator_base}/~manifest"],
            "child_locator_base": locator_base,
        }
        root_record["stable_identity"] = self._allocate_stable_identity(
            root_record["locators"][0],
            self.position_stable_identity_counts,
            self.position_stable_identities,
        )
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
        effective_leaves: Iterator[dict[str, Any]],
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
            record["stable_identity"] = self._allocate_stable_identity(
                record["locators"][0],
                self.position_stable_identity_counts,
                self.position_stable_identities,
            )
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
            for member_index, member_outline in enumerate(outlines):
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
                    ranges = self._resolve_quote_ranges(target, ranges)
                    for authored_range in ranges:
                        problem = authored_range.get("problem")
                        if not problem:
                            continue
                        resolution = authored_range.get("quoteResolution")
                        diagnostic_code = (
                            f"mark-quote-{resolution['state']}"
                            if isinstance(resolution, Mapping)
                            and resolution.get("state") in {"unresolved", "ambiguous"}
                            else problem
                        )
                        message = (
                            "Legacy quote selector did not match the referenced source"
                            if diagnostic_code == "mark-quote-unresolved"
                            else "Legacy quote selector matched multiple source ranges"
                            if diagnostic_code == "mark-quote-ambiguous"
                            else f"Invalid authored range: {problem}"
                        )
                        details = self._authored_range_location(
                            position_key=position_key,
                            role=role,
                            member_index=int(
                                member_outline.get("order", member_index)
                            ),
                            member_outline=member_outline,
                            target=target,
                            authored_range=authored_range,
                        )
                        if isinstance(resolution, Mapping):
                            details["quoteResolution"] = dict(resolution)
                        self.diagnostics.append(
                            AuthoredDiagnostic(
                                code=diagnostic_code,
                                message=message,
                                source_path=source.manifest_path,
                                line=authored_range.get("lineStart"),
                                position_key=position_key,
                                portal_key=portal_key,
                                details=details,
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
                    comment, inline_comment = _portal_annotation(
                        member_outline.get("comment"),
                        member_outline.get("inline_comment"),
                        ranges,
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
                        "comment": comment,
                        "inline_comment": inline_comment,
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

    def _resolve_quote_ranges(
        self,
        target: str | None,
        ranges: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.quote_resolver is None or target is None:
            return ranges
        resolved: list[dict[str, Any]] = []
        for authored_range in ranges:
            if (
                authored_range.get("origin") != "mark"
                or authored_range.get("problem") != "mark-quote-requires-range"
                or not authored_range.get("quote")
            ):
                resolved.append(authored_range)
                continue
            if target not in self.quote_representations:
                self.quote_representations[target] = self.quote_resolver(target)
            representation = self.quote_representations[target]
            if representation is None:
                resolved.append(authored_range)
                continue
            matches = _quote_matches(
                authored_range["quote"],
                representation,
            )
            updated = dict(authored_range)
            if not matches:
                updated["quoteResolution"] = _quote_resolution(
                    state="unresolved",
                    target=target,
                    quote=authored_range["quote"],
                    representation=representation,
                )
                updated["problem"] = "mark-quote-unresolved"
            elif len(matches) > 1:
                updated["quoteResolution"] = _quote_resolution(
                    state="ambiguous",
                    target=target,
                    quote=authored_range["quote"],
                    representation=representation,
                    matches=matches,
                )
                updated["problem"] = "mark-quote-ambiguous"
            else:
                match = matches[0]
                updated.pop("problem", None)
                updated["quoteResolution"] = _quote_resolution(
                    state="resolved",
                    target=target,
                    quote=authored_range["quote"],
                    representation=representation,
                    match=match,
                )
            resolved.append(updated)
        return resolved

    def _assign_portal_stable_identities(self) -> None:
        bases = {
            record["key"]: self._portal_stable_key(
                position_key=record["position_key"],
                role=record["role"],
                target=record["authored_target"],
                options=record["options"],
                ranges=record["ranges"],
            )
            for record in self.portal_records
        }
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in self.portal_records:
            grouped.setdefault(bases[record["key"]], []).append(record)

        for base, records in grouped.items():
            disambiguators = {
                _canonical_json(_portal_representation_selector(record))
                for record in records
            }
            use_representation = len(disambiguators) > 1
            for record in records:
                stable_key = base
                if use_representation:
                    stable_key = self._portal_stable_key(
                        position_key=record["position_key"],
                        role=record["role"],
                        target=record["authored_target"],
                        options=record["options"],
                        ranges=record["ranges"],
                        disambiguator=_portal_representation_selector(record),
                    )
                record["stable_identity"] = self._allocate_stable_identity(
                    stable_key,
                    self.portal_stable_identity_counts,
                    self.portal_stable_identities,
                )

    @staticmethod
    def _allocate_stable_identity(
        base: str,
        counts: dict[str, int],
        used: set[str],
    ) -> str:
        occurrence = counts.get(base, 0) + 1
        candidate = base if occurrence == 1 else f"{base}~{occurrence}"
        while candidate in used:
            occurrence += 1
            candidate = f"{base}~{occurrence}"
        counts[base] = occurrence
        used.add(candidate)
        return candidate

    def _portal_stable_key(
        self,
        *,
        position_key: str,
        role: str,
        target: str | None,
        options: Mapping[str, Any],
        ranges: list[dict[str, Any]],
        disambiguator: Mapping[str, Any] | None = None,
    ) -> str:
        position = next(
            item for item in self.position_records if item["key"] == position_key
        )
        semantic = {
            "position": position["stable_identity"],
            "role": role,
            "target": target,
            "options": _stable_portal_options(options),
            "ranges": [_stable_range(item) for item in ranges],
        }
        if disambiguator:
            semantic["disambiguator"] = disambiguator
        digest = hashlib.sha256(
            _canonical_json(semantic).encode("utf-8")
        ).hexdigest()[:24]
        return f"{position['stable_identity']}/{role}/{digest}"

    def _authored_range_location(
        self,
        *,
        position_key: str,
        role: str,
        member_index: int,
        member_outline: Mapping[str, Any],
        target: str | None,
        authored_range: Mapping[str, Any],
    ) -> dict[str, Any]:
        range_location = _omit_none(
            {
                "origin": authored_range.get("origin"),
                "index": authored_range.get("order"),
                "authored": authored_range.get("authored"),
                "lineStart": authored_range.get("lineStart"),
                "lineEnd": authored_range.get("lineEnd"),
            }
        )
        location = {
            "context": self.name,
            "component": {
                "key": position_key,
                "locator": self._locator_for(position_key),
            },
            "reference": _omit_none(
                {
                    "role": role,
                    "index": member_index,
                    "target": target,
                    "lineStart": member_outline.get("line_start"),
                    "lineEnd": member_outline.get("line_end"),
                }
            ),
            "range": range_location,
        }
        if authored_range.get("origin") == "mark":
            location["mark"] = {
                key: value
                for key, value in range_location.items()
                if key != "origin"
            }
        return {"authoredLocation": location}

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
        resolved = candidate if candidate.is_absolute() else _manifest_base_dir(source) / candidate
        resolved = resolved.resolve()
        if not resolved.is_file():
            if status != "unresolved":
                self.diagnostics.append(
                    AuthoredDiagnostic(
                        code="included-manifest-unresolved",
                        message=f"Included manifest does not exist: {resolved}",
                        source_path=source.manifest_path,
                        position_key=position_key,
                        portal_key=portal_key,
                        details={
                            "authoredTarget": target,
                            "resolvedPath": str(resolved),
                        },
                    )
                )
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


def _framing(
    value: Any,
    outline: dict[str, Any] | None,
    *,
    lead_text: str | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if lead_text:
        result.append({"kind": "text", "text": lead_text})
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


def _manifest_lead_text(source: ManifestSource) -> str | None:
    """The survey's own editorial prose, authored ahead of its fenced YAML.

    Distinct from `config.text`/`config.comment` (authored *inside* the YAML
    data), this is document prose the manifest format itself carries outside
    the fence -- the paragraph(s) that introduce a voice-survey brief before
    its pointer inventory begins. It lands once on the manifest's own
    position, not on any of its components or portals.
    """
    source_format = source.source_format
    return source_format.lead_text if source_format else None


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


def _portal_annotation(
    comment: dict[str, Any] | None,
    inline_comment: str | None,
    ranges: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """A portal's own editorial annotation, if the manifest structurally gives it one.

    A member's `comment`/`inline_comment` are authored directly on its own
    line (a `#` line before `- path: ...`, or trailing it) and always win.
    Absent that, a pointer wrapped by exactly one active mark inherits that
    mark's annotation -- the single wrapper is the unambiguous prose for the
    pointer as a whole. Two or more marks each annotate their own sub-range
    instead, and offer nothing unambiguous to promote, so the portal stays
    unannotated at its own level (its ranges still carry their own prose).
    """
    if comment is not None or inline_comment is not None:
        return comment, inline_comment
    active_marks = [
        item
        for item in ranges
        if item.get("kind") == "voice-span"
        and item.get("origin") == "mark"
        and not item.get("disabled")
    ]
    if len(active_marks) != 1:
        return comment, inline_comment
    only = active_marks[0]
    mark_comment = only.get("comment")
    mark_inline_comment = only.get("inlineComment")
    if mark_comment is None and mark_inline_comment is None:
        return comment, inline_comment
    return mark_comment, mark_inline_comment


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


def _quote_matches(
    quote: str,
    representation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_segments = representation.get("segments")
    if not isinstance(raw_segments, list):
        return []
    segments = []
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, Mapping) or not isinstance(segment.get("text"), str):
            continue
        normalized = dict(segment)
        normalized.setdefault("segmentIndex", index)
        segments.append(normalized)
    if not segments:
        return []
    authored_quote = quote.strip()
    if not authored_quote:
        return []
    normalized_quote = _normalize_quote_text(authored_quote)
    candidates: list[dict[str, Any]] = []
    for start in range(len(segments)):
        source_text = ""
        normalized_text = ""
        for end in range(start, len(segments)):
            segment_text = segments[end]["text"]
            source_text = f"{source_text}\n{segment_text}" if source_text else segment_text
            normalized_text = _normalize_quote_text(source_text)
            match_mode = None
            if authored_quote in source_text:
                match_mode = "exact"
            elif normalized_quote in normalized_text:
                match_mode = "normalized"
            if match_mode is None:
                continue
            candidate = _quote_candidate(
                segments[start : end + 1],
                source_text=source_text,
                match_mode=match_mode,
            )
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        return []
    exact = [candidate for candidate in candidates if candidate["matchMode"] == "exact"]
    candidates = exact or candidates
    smallest = min(len(candidate["segmentIndexes"]) for candidate in candidates)
    candidates = [
        candidate
        for candidate in candidates
        if len(candidate["segmentIndexes"]) == smallest
    ]
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            candidate["startSeconds"],
            candidate["endSeconds"],
            tuple(candidate["segmentIndexes"]),
            candidate["text"],
        )
        unique[key] = candidate
    return list(unique.values())


def _quote_candidate(
    segments: list[Mapping[str, Any]],
    *,
    source_text: str,
    match_mode: str,
) -> dict[str, Any] | None:
    start = _segment_number(segments[0], "startSeconds", "start_time")
    end = _segment_number(segments[-1], "endSeconds", "end_time")
    if start is None or end is None or end <= start:
        return None
    indexes = [
        segment.get("segmentIndex", index)
        for index, segment in enumerate(segments)
    ]
    return {
        "startSeconds": start,
        "endSeconds": end,
        "segmentIndexes": indexes,
        "text": source_text,
        "matchMode": match_mode,
    }


def _segment_number(
    segment: Mapping[str, Any],
    canonical: str,
    alternate: str,
) -> float | None:
    value = segment.get(canonical, segment.get(alternate))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or not math.isfinite(value):
        return None
    return float(value)


def _normalize_quote_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _quote_resolution(
    *,
    state: str,
    target: str,
    quote: str,
    representation: Mapping[str, Any],
    match: Mapping[str, Any] | None = None,
    matches: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    source = representation.get("source")
    result: dict[str, Any] = {
        "state": state,
        "target": target,
        "source": source if isinstance(source, str) and source else target,
        "quote": quote,
    }
    if match is not None:
        result.update(
            {
                "matchMode": match["matchMode"],
                "range": {
                    "startSeconds": match["startSeconds"],
                    "endSeconds": match["endSeconds"],
                },
                "evidence": {
                    "text": match["text"],
                    "segmentIndexes": list(match["segmentIndexes"]),
                },
            }
        )
    if matches is not None:
        result["candidates"] = [
            {
                "range": {
                    "startSeconds": candidate["startSeconds"],
                    "endSeconds": candidate["endSeconds"],
                },
                "segmentIndexes": list(candidate["segmentIndexes"]),
                "text": candidate["text"],
                "matchMode": candidate["matchMode"],
            }
            for candidate in matches
        ]
    return result


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


def _stable_portal_options(options: Mapping[str, Any]) -> dict[str, Any]:
    stable = _without(options, _PORTAL_OPTION_EXCLUSIONS)
    collection = stable.get("collection")
    if isinstance(collection, dict):
        stable["collection"] = _without(collection, {"order"})
    return stable


def _stable_range(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value[key]
        for key in ("kind", "origin", "disabled", "startSeconds", "endSeconds", "value")
        if key in value
    }
    if "startSeconds" not in result and "endSeconds" not in result:
        for key in ("authored", "problem"):
            if key in value:
                result[key] = value[key]
    return result


def _portal_representation_selector(record: Mapping[str, Any]) -> dict[str, Any]:
    options = record.get("options")
    option_prose = (
        _prose_fields(options) if isinstance(options, Mapping) else {}
    )
    range_prose = []
    for item in record.get("ranges") or []:
        if not isinstance(item, Mapping):
            continue
        prose = _prose_fields(item)
        if prose:
            range_prose.append(prose)
    selector = {
        "options": option_prose,
        "memberComment": _prose_value(record.get("comment")),
        "memberInlineComment": record.get("inline_comment"),
        "ranges": range_prose,
    }
    return {
        key: value
        for key, value in selector.items()
        if value not in ({}, [], None, "")
    }


def _prose_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {}
    for key in _PROSE_KEYS:
        if key not in value:
            continue
        prose = _prose_value(value[key])
        if prose not in (None, ""):
            fields[key] = prose
    return fields


def _prose_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("text")
    return value


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
