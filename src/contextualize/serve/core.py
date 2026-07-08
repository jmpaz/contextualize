"""Query core: one internal API over (registry + parsed manifests + index +
cache state), rendered by the CLI and MCP as thin, structurally identical
JSON producers. Every designed state below names the condition and at
least one path out; none falls through as a raw error."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..manifest.source import MEMBER_KEYS
from ..manifest.contexts import (
    ContextEntry,
    default_context_registry_path,
    default_context_status_path,
    load_context_registry,
    manifest_source_label,
)
from .resolve import (
    ManifestHandle,
    Target,
    component_for_node,
    component_members,
    load_manifest_handle,
    parse_selector,
    resolve_live_spec,
    resolve_target,
    spec_alias,
    spec_text,
)

NEXT_STEPS = {
    "not-found": [
        "Check `contextualize contexts list` for registered names.",
        "Or pass a path to an existing manifest file.",
    ],
    "not-found-node": [
        "Run `contextualize show <origin>` to see available components.",
    ],
    "unhydrated": [
        "Run `contextualize contexts hydrate <name>` to hydrate this context.",
        "`cat`/`links`/`status` still resolve directly from the manifest source in the meantime.",
    ],
    "disabled": [
        "This entry is commented out in the manifest source; remove the leading '#' to enable it.",
    ],
    "disabled-only": [
        "Enable at least one member by removing its leading '#' in the manifest source.",
    ],
    "empty-group": [
        "Add components to this group in the manifest source.",
    ],
    "unresolvable-source": [
        "Confirm the manifest file still exists at its recorded path.",
    ],
    "stale-cache": [
        "Run `contextualize contexts hydrate <name>` to refresh.",
    ],
}


def _age_seconds(path: Path) -> float | None:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def format_age(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    return f"{hours // 24}d"


def _origin_block(handle: ManifestHandle) -> dict[str, Any]:
    return {
        "kind": "registered" if handle.registry_name else "path",
        "name": handle.registry_name,
        "manifest_path": str(handle.manifest_path) if handle.manifest_path else None,
        "context_dir": str(handle.context_dir) if handle.context_dir else None,
        "hydrated": handle.hydrated,
    }


def _base_result(handle: ManifestHandle, selector) -> dict[str, Any]:
    return {"selector": selector.raw, "origin": _origin_block(handle)}


def _not_found(raw: str, origin: str) -> dict[str, Any]:
    return {
        "selector": raw,
        "origin": {
            "kind": "unknown",
            "name": origin,
            "manifest_path": None,
            "context_dir": None,
            "hydrated": False,
        },
        "node": None,
        "state": "not-found",
        "detail": f"'{origin}' is not a registered context and no manifest exists at that path.",
        "next_steps": NEXT_STEPS["not-found"],
    }


def _unresolvable_source(handle: ManifestHandle) -> dict[str, Any]:
    return {
        "selector": handle.origin,
        "origin": _origin_block(handle),
        "node": None,
        "state": "unresolvable-source",
        "detail": handle.source_error or "Manifest source could not be read.",
        "next_steps": NEXT_STEPS["unresolvable-source"],
    }


def _next_steps(state: str, handle: ManifestHandle) -> list[str]:
    name = handle.registry_name or handle.origin
    return [
        step.replace("<name>", name).replace("<origin>", handle.origin)
        for step in NEXT_STEPS.get(state, NEXT_STEPS["not-found-node"])
    ]


def _node_path_segment(node: dict[str, Any]) -> str:
    name = node.get("name")
    if isinstance(name, str) and name:
        return name
    return f"#{node.get('order', 0)}"


def _group_state(children: list[dict[str, Any]]) -> tuple[str, str | None]:
    if not children:
        return "empty-group", "Group has no members."
    if all(child.get("disabled") for child in children):
        return "disabled-only", "All members of this group are disabled."
    return "ok", None


def _leaf_state(members: dict[str, list[dict[str, Any]]]) -> tuple[str, str | None]:
    items = [item for group in members.values() for item in group]
    if items and all(item["disabled"] for item in items):
        return "disabled-only", "All members of this component are disabled."
    return "ok", None


def _render_members(
    node: dict[str, Any], comp: dict[str, Any] | None
) -> dict[str, list[dict[str, Any]]]:
    rendered: dict[str, list[dict[str, Any]]] = {}
    node_members = node.get("members", {})
    comp_lists = {key: (comp.get(key) if comp else None) or [] for key in MEMBER_KEYS}
    for key in MEMBER_KEYS:
        items = node_members.get(key, [])
        if not items:
            continue
        entries = []
        enabled_index = 0
        for item in items:
            entry = {
                "order": item["order"],
                "disabled": item["disabled"],
                "comment": item.get("comment"),
                "inline_comment": item.get("inline_comment"),
            }
            if item["disabled"]:
                entry["raw"] = item.get("raw")
            else:
                raw_list = comp_lists[key]
                if enabled_index < len(raw_list):
                    raw_entry = raw_list[enabled_index]
                    entry["spec"] = spec_text(raw_entry)
                    entry["alias"] = spec_alias(raw_entry)
                enabled_index += 1
            entries.append(entry)
        rendered[key] = entries
    return rendered


def _render_node(
    node: dict[str, Any],
    dotted_path: tuple[str, ...],
    handle: ManifestHandle,
    depth_remaining: int | None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": node["kind"],
        "name": node.get("name"),
        "path": ".".join(dotted_path),
        "order": node.get("order"),
        "disabled": node.get("disabled", False),
        "comment": node.get("comment"),
        "inline_comment": node.get("inline_comment"),
    }
    if node.get("disabled"):
        base["raw"] = node.get("raw")
        base["state"] = "disabled"
        return base

    if node["kind"] == "group":
        children = node.get("children", [])
        state, detail = _group_state(children)
        base["state"] = state
        base["detail"] = detail
        if depth_remaining is not None and depth_remaining <= 0:
            base["collapsed"] = True
            base["child_count"] = len(children)
        else:
            next_depth = None if depth_remaining is None else depth_remaining - 1
            base["children"] = [
                _render_node(child, dotted_path + (_node_path_segment(child),), handle, next_depth)
                for child in children
            ]
        return base

    comp = component_for_node(handle, dotted_path)
    members = _render_members(node, comp)
    base["members"] = members
    if node["kind"] == "set" and isinstance(node.get("set"), dict):
        base["set"] = node["set"]
    state, detail = _leaf_state(members)
    base["state"] = state
    base["detail"] = detail
    return base


def _manifest_state(handle: ManifestHandle) -> tuple[str, str | None]:
    if not handle.hydrated:
        return "unhydrated", "This manifest has not been hydrated yet."
    if not handle.outline:
        return "empty-group", "Manifest has no components."
    return "ok", None


def _member_outline_item(node: dict[str, Any], key: str, position: int) -> dict[str, Any] | None:
    items = [item for item in node.get("members", {}).get(key, []) if not item["disabled"]]
    return items[position] if position < len(items) else None


def _render_disabled_member(key: str, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "member",
        "key": key,
        "disabled": True,
        "raw": item.get("raw"),
        "order": item.get("order"),
        "comment": item.get("comment"),
        "inline_comment": item.get("inline_comment"),
        "state": "disabled",
        "detail": None,
    }


def _render_member(node: dict[str, Any], key: str, position: int, entry: Any) -> dict[str, Any]:
    item = _member_outline_item(node, key, position)
    return {
        "kind": "member",
        "key": key,
        "spec": spec_text(entry),
        "alias": spec_alias(entry),
        "order": item["order"] if item else position,
        "disabled": False,
        "comment": item.get("comment") if item else None,
        "inline_comment": item.get("inline_comment") if item else None,
        "state": "ok",
        "detail": None,
    }


def show(
    selector_text: str,
    *,
    depth: int | None = None,
    registry_path: str | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    selector = parse_selector(selector_text, cwd=cwd)
    handle = load_manifest_handle(selector.origin, registry_path=registry_path, cwd=cwd)
    if handle is None:
        return _not_found(selector_text, selector.origin)
    if handle.source is None:
        return _unresolvable_source(handle)

    target = resolve_target(handle, selector.tokens)
    result = _base_result(handle, selector)

    if target.kind == "not-found":
        result.update(node=None, state="not-found", detail=target.detail, next_steps=_next_steps("not-found-node", handle))
        return result

    if target.kind == "root":
        result["node"] = {
            "kind": "manifest",
            "name": None,
            "path": "",
            "children": [
                _render_node(node, (_node_path_segment(node),), handle, depth) for node in handle.outline
            ],
        }
        state, detail = _manifest_state(handle)
        result.update(state=state, detail=detail, next_steps=_next_steps(state, handle) if state in NEXT_STEPS else [])
        return result

    if target.kind == "disabled":
        if target.disabled_member is not None:
            key, item = target.disabled_member
            result["node"] = _render_disabled_member(key, item)
        elif target.node is not None:
            result["node"] = _render_node(target.node, target.dotted_path, handle, depth)
        else:
            result["node"] = None
        result.update(state="disabled", detail=target.detail, next_steps=_next_steps("disabled", handle))
        return result

    if target.kind == "member":
        key, position, entry = target.require_member()
        result["node"] = _render_member(target.require_node(), key, position, entry)
        result.update(state="ok", detail=None, next_steps=[])
        return result

    result["node"] = _render_node(target.require_node(), target.dotted_path, handle, depth)
    state, detail = result["node"]["state"], result["node"].get("detail")
    result.update(state=state, detail=detail, next_steps=_next_steps(state, handle) if state in NEXT_STEPS else [])
    return result


def _index_entries_for(handle: ManifestHandle, dotted_path: tuple[str, ...]) -> list[dict[str, Any]] | None:
    if not handle.index:
        return None
    components = handle.index.get("components")
    if not isinstance(components, dict):
        return None
    entries = components.get(".".join(dotted_path))
    return entries if isinstance(entries, list) else None


def _payload_kind_for_hydrated_path(path: Path) -> str:
    return "pointer" if path.is_symlink() else "copy"


def _one_member(
    node: dict[str, Any], key: str, position: int, entry: Any, handle: ManifestHandle
) -> tuple[list[dict[str, Any]], list[str], str, tuple[int, int] | None]:
    spec = resolve_live_spec(handle, spec_text(entry))
    member = {"key": key, "spec": spec, "alias": spec_alias(entry), "payload": "pointer"}
    item = _member_outline_item(node, key, position)
    span = (item["line_start"], item["line_end"]) if item else None
    return [member], [spec], "pointer", span


def _gather_leaf(
    node: dict[str, Any], dotted_path: tuple[str, ...], comp: dict[str, Any], handle: ManifestHandle
) -> tuple[list[dict[str, Any]], list[str], str, tuple[int, int] | None]:
    span = (node["line_start"], node["line_end"])
    if node["kind"] == "set":
        set_info = node.get("set")
        if isinstance(set_info, dict) and handle.context_dir is not None:
            fused_path = handle.context_dir / set_info["context_path"]
            if fused_path.is_file():
                member = {
                    "key": "set",
                    "spec": str(fused_path),
                    "alias": None,
                    "payload": "copy",
                }
                return [member], [str(fused_path)], "copy", span

    entries = _index_entries_for(handle, dotted_path)
    if entries:
        members = []
        specs = []
        payload_kinds: set[str] = set()
        for entry in entries:
            context_path = entry.get("context_path")
            if not context_path or handle.context_dir is None:
                continue
            absolute = handle.context_dir / context_path
            kind = _payload_kind_for_hydrated_path(absolute)
            payload_kinds.add(kind)
            members.append({"key": "resolved", "spec": str(absolute), "alias": None, "payload": kind})
            specs.append(str(absolute))
        if specs:
            payload = payload_kinds.pop() if len(payload_kinds) == 1 else "mixed"
            return members, specs, payload, span

    members = []
    specs = []
    for key, _position, entry in component_members(comp):
        resolved_spec = resolve_live_spec(handle, spec_text(entry))
        members.append(
            {"key": key, "spec": resolved_spec, "alias": spec_alias(entry), "payload": "pointer"}
        )
        specs.append(resolved_spec)
    return members, specs, "pointer", span


def _iter_leaves_with_path(nodes, prefix: tuple[str, ...]):
    for node in nodes:
        if node.get("disabled"):
            continue
        path = prefix + (_node_path_segment(node),)
        if node["kind"] == "group":
            yield from _iter_leaves_with_path(node.get("children", []), path)
        else:
            yield node, path


def _gather_group(
    node: dict[str, Any], dotted_path: tuple[str, ...], handle: ManifestHandle
) -> tuple[list[dict[str, Any]], list[str], str, tuple[int, int] | None]:
    members: list[dict[str, Any]] = []
    specs: list[str] = []
    payload_kinds: set[str] = set()
    for leaf, leaf_path in _iter_leaves_with_path(node.get("children", []), dotted_path):
        comp = component_for_node(handle, leaf_path)
        if comp is None:
            continue
        leaf_members, leaf_specs, payload_kind, _span = _gather_leaf(leaf, leaf_path, comp, handle)
        members.extend(leaf_members)
        specs.extend(leaf_specs)
        payload_kinds.add(payload_kind)
    payload = payload_kinds.pop() if len(payload_kinds) == 1 else ("mixed" if payload_kinds else "pointer")
    return members, specs, payload, (node["line_start"], node["line_end"])


def _adjacency_for_span(
    handle: ManifestHandle, line_start: int, line_end: int, around: int
) -> dict[str, Any] | None:
    if handle.manifest_path is None:
        return None
    try:
        text = handle.manifest_path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = text.splitlines()
    start = max(1, line_start - around)
    end = min(len(lines), line_end + around)
    return {"line_start": start, "line_end": end, "text": "\n".join(lines[start - 1 : end])}


def _no_specs_state(target: Target) -> tuple[str, str]:
    if target.kind == "group":
        state, detail = _group_state(target.require_node().get("children", []))
        if state != "ok":
            return state, detail or ""
        return (
            "not-found",
            "This group has enabled components, but none resolve to a file/repo/manifest member.",
        )
    node_members = target.node.get("members", {}) if target.node else {}
    all_items = [item for items in node_members.values() for item in items]
    if all_items and all(item["disabled"] for item in all_items):
        return "disabled-only", "All members of this component are disabled."
    return (
        "not-found",
        "This component has no files/repos/manifests members; its content may be authored "
        "inline (text/prefix/suffix), which cat does not draw directly.",
    )


def cat_selector(
    selector_text: str,
    *,
    around: int | None = None,
    registry_path: str | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    selector = parse_selector(selector_text, cwd=cwd)
    handle = load_manifest_handle(selector.origin, registry_path=registry_path, cwd=cwd)
    if handle is None:
        return {**_not_found(selector_text, selector.origin), "members": [], "specs": [], "adjacency": None, "payload": None}
    if handle.source is None:
        return {**_unresolvable_source(handle), "members": [], "specs": [], "adjacency": None, "payload": None}

    result = _base_result(handle, selector)
    result.update(members=[], specs=[], adjacency=None, payload=None)

    target = resolve_target(handle, selector.tokens)

    if target.kind == "root":
        result.update(
            node=None,
            state="not-found",
            detail="cat needs a component selector: name-or-path:component[.member].",
            next_steps=_next_steps("not-found-node", handle),
        )
        return result

    if target.kind == "not-found":
        result.update(node=None, state="not-found", detail=target.detail, next_steps=_next_steps("not-found-node", handle))
        return result

    node = target.require_node()
    result["node"] = {"path": ".".join(target.dotted_path), "kind": node["kind"], "name": node.get("name")}

    if target.kind == "disabled":
        result.update(state="disabled", detail=target.detail, next_steps=_next_steps("disabled", handle))
        return result

    if target.kind == "group":
        members, specs, payload, span = _gather_group(node, target.dotted_path, handle)
    elif target.kind == "member":
        key, position, entry = target.require_member()
        members, specs, payload, span = _one_member(node, key, position, entry, handle)
    else:
        assert target.comp is not None, f"kind={target.kind!r} carries a comp"
        members, specs, payload, span = _gather_leaf(node, target.dotted_path, target.comp, handle)

    result["members"] = members
    result["specs"] = specs
    result["payload"] = payload

    if not specs:
        state, detail = _no_specs_state(target)
        result.update(state=state, detail=detail, next_steps=_next_steps(state, handle))
        return result

    if around is not None and span is not None:
        adjacency = _adjacency_for_span(handle, span[0], span[1], around)
        result["adjacency"] = adjacency
        if adjacency is None:
            result["adjacency_note"] = handle.source_error or "Authored source unavailable for adjacency."

    result.update(state="ok", detail=None, next_steps=[])
    return result


def draw_substance(result: dict[str, Any]) -> dict[str, Any]:
    """Non-ok results pass through untouched: a designed state is already the
    complete answer, and inventing content for it would be fabrication."""
    if result.get("state") != "ok":
        return result
    from ..references.factory import create_file_references

    drawn = create_file_references(list(result["specs"]))
    return {**result, "content": drawn["concatenated"]}


def _resolved(path: Path | str) -> str:
    return str(Path(path).resolve(strict=False))


def _entry_source_identity(handle: ManifestHandle, entry: dict[str, Any]) -> str | None:
    if entry.get("kind") == "set":
        return None
    if entry.get("source_type") == "local":
        local = _local_entry_source_path(handle, entry)
        return _resolved(local) if local is not None else None
    source_type = entry.get("source_type")
    source_ref = entry.get("source_ref")
    if not source_type or not source_ref:
        return None
    source_path = entry.get("source_path")
    return f"{source_type}:{source_ref}" + (f":{source_path}" if source_path else "")


def _member_source_identities(handle: ManifestHandle) -> dict[str, list[str]]:
    identities: dict[str, list[str]] = {}
    components = handle.index.get("components") if handle.index else None
    if not isinstance(components, dict):
        return identities
    for dotted, entries in components.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            identity = _entry_source_identity(handle, entry)
            if identity is not None and dotted not in identities.setdefault(identity, []):
                identities[identity].append(dotted)
    return identities


def links(
    selector_text: str,
    *,
    direction: str = "both",
    registry_path: str | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    if direction not in ("in", "out", "both"):
        raise ValueError("direction must be 'in', 'out', or 'both'")

    try:
        registry = load_context_registry(registry_path)
    except (OSError, ValueError):
        registry = {}

    selector = parse_selector(selector_text, cwd=cwd)
    handle = load_manifest_handle(selector.origin, registry=registry, registry_path=registry_path, cwd=cwd)
    if handle is None:
        return {**_not_found(selector_text, selector.origin), "out": None, "in": None, "shared": None, "coverage": None}
    if handle.source is None:
        return {**_unresolvable_source(handle), "out": None, "in": None, "shared": None, "coverage": None}

    result = _base_result(handle, selector)
    context_label = handle.registry_name or handle.display_name

    out_edges: list[dict[str, Any]] | None = None
    if direction in ("out", "both"):
        out_edges = (
            [dict(edge, source_context=context_label) for edge in handle.references_out]
            if handle.hydrated
            else []
        )

    in_edges: list[dict[str, Any]] | None = None
    shared: list[dict[str, Any]] | None = None
    coverage = {"registry_total": len(registry), "scanned": 0, "skipped_unhydrated": []}
    if direction in ("in", "both") and handle.manifest_path is not None:
        in_edges = []
        own_identities = _member_source_identities(handle) if handle.hydrated else {}
        shared = [] if handle.hydrated else None
        target_resolved = _resolved(handle.manifest_path)
        for name, entry in registry.items():
            if name == handle.registry_name:
                continue
            other = load_manifest_handle(name, registry=registry, registry_path=registry_path, cwd=cwd)
            if other is None:
                continue
            if not other.hydrated:
                coverage["skipped_unhydrated"].append(name)
                continue
            coverage["scanned"] += 1
            for edge in other.references_out:
                edge_target = edge.get("target_path")
                if edge_target and _resolved(edge_target) == target_resolved:
                    in_edges.append(dict(edge, source_context=name, target_context=context_label))
            if shared is None or not own_identities:
                continue
            for identity, their_components in _member_source_identities(other).items():
                own_components = own_identities.get(identity)
                if own_components:
                    shared.append(
                        {
                            "kind": "shared-member",
                            "source": identity,
                            "context": name,
                            "components": own_components,
                            "their_components": their_components,
                        }
                    )

    result["out"] = out_edges
    result["in"] = in_edges
    result["shared"] = shared
    result["coverage"] = coverage

    if direction in ("out", "both") and not handle.hydrated:
        result.update(
            state="unhydrated",
            detail="Outbound references are recorded at hydration time.",
            next_steps=_next_steps("unhydrated", handle),
        )
        return result

    has_any = bool(out_edges) or bool(in_edges) or bool(shared)
    if not has_any:
        result.update(state="ok", detail="No references recorded in either direction.", next_steps=[])
    else:
        result.update(state="ok", detail=None, next_steps=[])
    return result


def _latest_hydration_status(name: str | None, status_path: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    import json

    path = Path(status_path).expanduser() if status_path else default_context_status_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    contexts = payload.get("contexts") if isinstance(payload, dict) else None
    if not isinstance(contexts, dict):
        return None
    return contexts.get(name)


def _empty_drift() -> dict[str, Any]:
    return {
        "any": False,
        "sources_changed": [],
        "members_vanished": [],
        "references_gone": [],
        "hydration_stale": False,
    }


def _local_entry_source_path(handle: ManifestHandle, entry: dict[str, Any]) -> Path | None:
    context_path = entry.get("context_path")
    if context_path and handle.context_dir is not None:
        hydrated = handle.context_dir / context_path
        if hydrated.is_symlink():
            try:
                return hydrated.resolve(strict=False)
            except OSError:
                pass
    source_ref = entry.get("source_ref")
    source_path = entry.get("source_path")
    if source_ref and source_path:
        return Path(source_ref) / source_path
    return None


def _compute_drift(handle: ManifestHandle, index_mtime: float | None) -> dict[str, Any]:
    sources_changed = []
    members_vanished = []
    references_gone = []
    hydration_stale = False

    if index_mtime is not None and handle.manifest_path is not None:
        try:
            hydration_stale = handle.manifest_path.stat().st_mtime > index_mtime
        except OSError:
            pass

    components = handle.index.get("components", {}) if handle.index else {}
    if isinstance(components, dict):
        for comp_name, entries in components.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("source_type") != "local":
                    continue
                absolute = _local_entry_source_path(handle, entry)
                if absolute is None:
                    continue
                try:
                    mtime = absolute.stat().st_mtime
                except OSError:
                    members_vanished.append({"component": comp_name, "path": str(absolute)})
                    continue
                if index_mtime is not None and mtime > index_mtime:
                    sources_changed.append({"component": comp_name, "path": str(absolute)})

    for edge in handle.references_out:
        target = edge.get("target_path")
        if target and not Path(target).exists():
            references_gone.append(edge)

    return {
        "any": bool(sources_changed or members_vanished or references_gone or hydration_stale),
        "sources_changed": sources_changed,
        "members_vanished": members_vanished,
        "references_gone": references_gone,
        "hydration_stale": hydration_stale,
    }


def _drift_detail(drift: dict[str, Any]) -> str:
    bits = []
    if drift["hydration_stale"]:
        bits.append("manifest source is newer than the hydrated index")
    if drift["sources_changed"]:
        bits.append(f"{len(drift['sources_changed'])} source(s) changed since resolution")
    if drift["members_vanished"]:
        bits.append(f"{len(drift['members_vanished'])} member target(s) vanished")
    if drift["references_gone"]:
        bits.append(f"{len(drift['references_gone'])} referenced manifest(s) gone")
    return "; ".join(bits) or "drift detected"


def _full_status(handle: ManifestHandle, *, status_path: str | None = None) -> dict[str, Any]:
    name = handle.registry_name or handle.display_name
    hydration = _latest_hydration_status(handle.registry_name, status_path)
    result: dict[str, Any] = {
        "selector": name,
        "origin": _origin_block(handle),
        "name": name,
        "manifest_path": str(handle.manifest_path) if handle.manifest_path else None,
        "context_dir": str(handle.context_dir) if handle.context_dir else None,
        "hydrated": handle.hydrated,
        "hydration": hydration,
    }
    if not handle.hydrated:
        result.update(
            cache_age_seconds=None,
            drift=_empty_drift(),
            state="unhydrated",
            detail="Not yet hydrated.",
            next_steps=_next_steps("unhydrated", handle),
        )
        return result

    assert handle.context_dir is not None, "a hydrated handle has a context dir"
    index_path = handle.context_dir / "index.json"
    index_mtime: float | None
    try:
        index_mtime = index_path.stat().st_mtime
    except OSError:
        index_mtime = None

    drift = _compute_drift(handle, index_mtime)
    result["cache_age_seconds"] = (time.time() - index_mtime) if index_mtime is not None else None
    result["drift"] = drift

    if drift["any"]:
        state = "stale-cache" if (drift["hydration_stale"] or drift["sources_changed"]) else "dangling-reference"
        result.update(state=state, detail=_drift_detail(drift), next_steps=_next_steps("stale-cache", handle))
    else:
        result.update(state="ok", detail=None, next_steps=[])
    return result


def _context_status(
    name: str,
    registry: dict[str, ContextEntry],
    *,
    registry_path: str | None,
    status_path: str | None,
    cwd: str | None,
) -> dict[str, Any]:
    handle = load_manifest_handle(name, registry=registry, registry_path=registry_path, cwd=cwd)
    if handle is None:
        not_found = _not_found(name, name)
        return {**not_found, "name": name, "detail": "Registry entry could not be resolved."}
    if handle.source is None:
        return {**_unresolvable_source(handle), "name": name}
    return _full_status(handle, status_path=status_path)


def status(
    selector_text: str | None = None,
    *,
    registry_path: str | None = None,
    status_path: str | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    try:
        registry = load_context_registry(registry_path)
    except (OSError, ValueError) as exc:
        return {
            "state": "not-found",
            "detail": str(exc),
            "next_steps": NEXT_STEPS["not-found"],
            "contexts": [],
        }

    if selector_text is None:
        contexts = [
            _context_status(name, registry, registry_path=registry_path, status_path=status_path, cwd=cwd)
            for name in sorted(registry)
        ]
        drifted = sum(1 for entry in contexts if entry.get("drift", {}).get("any"))
        return {
            "state": "ok",
            "detail": None,
            "next_steps": [],
            "registry": {
                "total": len(registry),
                "path": str(registry_path) if registry_path else str(default_context_registry_path()),
            },
            "contexts": contexts,
            "drift_summary": {"drifted": drifted, "total": len(registry)},
        }

    selector = parse_selector(selector_text, cwd=cwd)
    handle = load_manifest_handle(selector.origin, registry=registry, registry_path=registry_path, cwd=cwd)
    if handle is None:
        return {**_not_found(selector_text, selector.origin), "hydration": None, "drift": _empty_drift()}
    if handle.source is None:
        base = _unresolvable_source(handle)
        return {**base, "hydration": None, "drift": _empty_drift()}
    return _full_status(handle, status_path=status_path)


def shelf_for_cwd(cwd: str, registry: dict[str, ContextEntry]) -> list[dict[str, Any]]:
    cwd_path = Path(cwd).resolve()
    shelf = []
    for name, entry in registry.items():
        try:
            target = entry.target_dir.resolve()
        except OSError:
            continue
        if cwd_path != target and target not in cwd_path.parents:
            continue
        handle = load_manifest_handle(name, registry={name: entry}, cwd=cwd)
        one_line = "not yet hydrated"
        freshness = None
        if handle is not None and handle.hydrated:
            components = handle.index.get("components", {}) if handle.index else {}
            count = len(components) if isinstance(components, dict) else 0
            age_seconds = _age_seconds(handle.context_dir / "index.json") if handle.context_dir else None
            freshness = format_age(age_seconds)
            one_line = f"{count} component(s)" + (f" - hydrated {freshness} ago" if freshness else "")
        shelf.append(
            {
                "name": name,
                "manifest_source": manifest_source_label(entry),
                "target": str(entry.target_dir),
                "one_line": one_line,
                "freshness": freshness,
            }
        )
    shelf.sort(key=lambda item: item["name"])
    return shelf
