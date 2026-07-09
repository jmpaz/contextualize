"""Rich text rendering for the CLI surface. `--json` bypasses this module
entirely and prints the query core's dict as-is -- this is the only place
raw structure becomes prose."""

from __future__ import annotations

import io
from typing import Any

from .core import format_age

STATE_LABELS = {
    "not-found": "not found",
    "unhydrated": "unhydrated",
    "disabled": "disabled",
    "disabled-only": "disabled-only",
    "empty-group": "empty group",
    "empty-manifest": "empty manifest",
    "unresolvable-source": "unresolvable source",
    "stale-cache": "stale",
    "hydrate-failed": "hydrate failed",
    "dangling-reference": "dangling reference",
    "mark-quote-requires-range": "quote needs a range",
    "mark-invalid-time": "invalid mark time",
    "mark-at-and-span": "at and span",
    "mark-missing-time": "missing mark time",
    "mark-beyond-duration": "beyond duration",
    "marks-on-untimed-target": "untimed target",
    "marks-require-single-document": "multi-document target",
    "mark-params-unsupported": "params with mark",
    "transcript-drift": "transcript drift",
}

DISABLED_MARK = "⊘"


def _state_banner(result: dict[str, Any]) -> list[str]:
    state = result.get("state")
    if not state or state == "ok":
        return []
    label = STATE_LABELS.get(state, state)
    detail = result.get("detail")
    lines = [f"[{label}] {detail}" if detail else f"[{label}]"]
    for step in result.get("next_steps") or []:
        lines.append(f"    -> {step}")
    return lines


def _origin_header(result: dict[str, Any]) -> list[str]:
    origin = result.get("origin") or {}
    name = origin.get("name") or result.get("selector")
    header = str(name)
    if origin.get("manifest_path"):
        header += f"  ({origin['manifest_path']})"
    hydrated = "hydrated" if origin.get("hydrated") else "not hydrated"
    return [header, f"  {hydrated}"]


def _comment_lines(comment: dict[str, Any] | None, pad: str) -> list[str]:
    if not comment:
        return []
    return [f"{pad}# {line}" for line in comment["text"].splitlines()]


def _member_line(item: dict[str, Any], pad: str) -> str:
    if item.get("disabled"):
        raw = (item.get("raw") or "").splitlines()[0]
        return f"{pad}{DISABLED_MARK} {raw}"
    spec = item.get("spec", "")
    label = f"{spec}  (alias: {item['alias']})" if item.get("alias") else spec
    inline = item.get("inline_comment")
    if inline:
        label = f"{label}  # {inline}"
    return f"{pad}- {label}"


def _mark_line(mark: dict[str, Any], pad: str) -> str:
    if mark.get("disabled"):
        raw = (mark.get("raw") or "").splitlines()[0] if mark.get("raw") else ""
        return f"{pad}{DISABLED_MARK} {raw}"
    label = f"@ {mark.get('at')}"
    if mark.get("quote"):
        label += "  (quote)"
    state = mark.get("state", "ok")
    if state != "ok":
        label += f"  [{state}]"
    inline = mark.get("inline_comment")
    if inline:
        label += f"  # {inline}"
    return f"{pad}{label}"


def _mark_block_lines(item: dict[str, Any], pad: str) -> list[str]:
    lines: list[str] = []
    for mark in item.get("marks") or []:
        lines.extend(_comment_lines(mark.get("comment"), pad))
        lines.append(_mark_line(mark, pad))
    return lines


def _render_component(node: dict[str, Any], indent: int, ordinal: int | None) -> list[str]:
    pad = "  " * indent
    lines = _comment_lines(node.get("comment"), pad)
    marker = f"{ordinal:>2} " if ordinal is not None else ""
    name = node.get("name") or "(unnamed)"

    if node.get("disabled"):
        raw_first = (node.get("raw") or "").splitlines()[0]
        lines.append(f"{pad}{marker}{DISABLED_MARK} {name}  ({raw_first})")
        return lines

    state_note = "" if node.get("state", "ok") == "ok" else f"  [{node['state']}]"

    if node["kind"] == "group":
        lines.append(f"{pad}{marker}{name}  (group){state_note}")
        if node.get("collapsed"):
            lines.append(f"{pad}    ... {node['child_count']} member(s), use --depth to expand")
        else:
            for index, child in enumerate(node.get("children", []), 1):
                lines.extend(_render_component(child, indent + 1, index))
        return lines

    counts = {key: len(items) for key, items in node.get("members", {}).items()}
    summary = ", ".join(f"{count} {key}" for key, count in counts.items())
    if not summary:
        inline_keys = node.get("inline_content") or []
        summary = f"inline {', '.join(inline_keys)}" if inline_keys else "no members"
    kind_label = " (set)" if node["kind"] == "set" else ""
    inline = f"   # {node['inline_comment']}" if node.get("inline_comment") else ""
    lines.append(f"{pad}{marker}{name}{kind_label}  {summary}{state_note}{inline}")
    for items in node.get("members", {}).values():
        for item in items:
            lines.append(_member_line(item, pad + "    "))
            lines.extend(_mark_block_lines(item, pad + "      "))
    return lines


def _render_node(node: dict[str, Any] | None) -> list[str]:
    if node is None:
        return []
    if node["kind"] == "manifest":
        if not node["children"]:
            return ["  (no components)"]
        lines: list[str] = []
        for index, child in enumerate(node["children"], 1):
            lines.extend(_render_component(child, 1, index))
        return lines
    if node["kind"] == "member":
        if node.get("disabled"):
            raw_first = (node.get("raw") or "").splitlines()[0]
            return [f"  {DISABLED_MARK} {raw_first}"]
        return [_member_line(node, "  ")] + _mark_block_lines(node, "    ")
    if node["kind"] == "mark":
        lines = [f"  on {node['member_spec']}"] if node.get("member_spec") else []
        lines.extend(_comment_lines(node.get("comment"), "  "))
        lines.append(_mark_line(node, "  "))
        if node.get("address"):
            lines.append(f"    {node['address']}")
        if node.get("quote"):
            lines.append("    quote:")
            lines.extend(
                f"      {line}" for line in str(node["quote"]).rstrip("\n").splitlines()
            )
        for ref in node.get("refs") or []:
            lines.append(f"    ref: {ref}")
        return lines
    return _render_component(node, 1, None)


def render_show(result: dict[str, Any]) -> str:
    lines = _origin_header(result)
    lines.append("")
    lines.extend(_state_banner(result))
    lines.extend(_render_node(result.get("node")))
    return "\n".join(lines)


def render_cat_adjacency(result: dict[str, Any]) -> str | None:
    adjacency = result.get("adjacency")
    if not adjacency:
        return None
    header = f"-- authored context (lines {adjacency['line_start']}-{adjacency['line_end']}) --"
    return f"{header}\n{adjacency['text']}"


def render_links(result: dict[str, Any]) -> str:
    lines = _origin_header(result)
    lines.append("  links")
    lines.extend(_state_banner(result))

    if result.get("out") is not None:
        lines.append(f"\n  out ({len(result['out'])})")
        for edge in result["out"]:
            lines.append(
                f"    {edge.get('component')} -> {edge.get('target_path')}"
                f"  ({edge.get('form')}, {edge.get('detected_via')}, {edge.get('payload')})"
            )

    if result.get("in") is not None:
        lines.append(f"\n  in ({len(result['in'])})")
        for edge in result["in"]:
            if edge.get("kind") == "mark":
                lines.append(_mark_edge_line(edge))
                continue
            lines.append(
                f"    {edge.get('source_context')}  ({edge.get('form')}, {edge.get('payload')})"
            )

    if result.get("shared") is not None:
        lines.append(f"\n  shared members ({len(result['shared'])})")
        for edge in result["shared"]:
            here = ", ".join(edge.get("components") or [])
            theirs = ", ".join(edge.get("their_components") or [])
            lines.append(
                f"    {edge.get('source')}  (here: {here}; {edge.get('context')}: {theirs})"
            )

    coverage = result.get("coverage")
    if coverage and result.get("in") is not None:
        skipped = coverage.get("skipped_unhydrated") or []
        note = f"  ({len(skipped)} unhydrated: {', '.join(skipped)})" if skipped else ""
        lines.append(
            f"\n  coverage: scanned {coverage['scanned']}/{coverage['registry_total']} registered contexts{note}"
        )
        tag_scope = coverage.get("tag_scope")
        if tag_scope:
            lines.append(_tag_scope_line(tag_scope))
    return "\n".join(lines)


def _mark_edge_line(edge: dict[str, Any]) -> str:
    who = edge.get("source_context") or edge.get("source_note")
    line = f"    {edge.get('address')}  ({who}"
    if edge.get("component"):
        line += f", {edge['component']}"
    line += ")"
    state = edge.get("state")
    if state and state != "ok":
        line += f"  [{state}]"
    inline = edge.get("inline_comment")
    if inline:
        line += f"  # {inline}"
    return line


def _tag_scope_line(tag_scope: dict[str, Any]) -> str:
    tags = ", ".join(tag_scope.get("tags") or [])
    if tag_scope.get("skipped"):
        return f"  tag scope: {tags} - skipped: {tag_scope['skipped']}"
    return (
        f"  tag scope: {tags} @ {tag_scope.get('root')}"
        f" - {tag_scope.get('notes_with_manifest', 0)}/{tag_scope.get('notes_scanned', 0)}"
        " note(s) with manifests"
    )


def _render_drift(drift: dict[str, Any]) -> list[str]:
    unchecked = drift.get("marks_unchecked")
    if not drift.get("any"):
        lines = ["  drift: none"]
        if unchecked:
            lines.append(f"    (marks unchecked: {unchecked})")
        return lines
    lines = ["  drift:"]
    if drift.get("hydration_stale"):
        lines.append("    - hydration is older than its manifest source")
    for item in drift.get("sources_changed", []):
        lines.append(f"    - source changed since resolution: {item['path']}")
    for item in drift.get("members_vanished", []):
        lines.append(f"    - member target vanished: {item['path']}")
    for edge in drift.get("references_gone", []):
        lines.append(f"    - referenced manifest gone: {edge.get('target_path')}")
    for item in drift.get("marks_drifted", []):
        lines.append(f"    - mark drifted: {item.get('address')} ({item.get('reason')})")
    if unchecked:
        lines.append(f"    (marks unchecked: {unchecked})")
    return lines


def _render_status_context(result: dict[str, Any]) -> str:
    lines = _origin_header(result)
    lines.append("  status")
    lines.extend(_state_banner(result))

    hydration = result.get("hydration")
    if hydration:
        lines.append(
            f"  last hydrate: {hydration.get('result')} at {hydration.get('timestamp')}"
        )
    age = result.get("cache_age_seconds")
    if age is not None:
        lines.append(f"  cache age: {format_age(age)}")

    drift = result.get("drift")
    if drift is not None:
        lines.extend(_render_drift(drift))
    return "\n".join(lines)


def _render_status_registry(result: dict[str, Any]) -> str:
    from rich.console import Console
    from rich.table import Table

    registry = result.get("registry", {})
    drift_summary = result.get("drift_summary", {})
    header = (
        f"registry: {registry.get('total', 0)} context(s)"
        f"  ({drift_summary.get('drifted', 0)} drifted)"
    )

    table = Table(box=None, padding=(0, 2), show_edge=False, header_style="bold")
    for column in ("name", "state", "cache age"):
        table.add_column(column, no_wrap=True)
    for entry in result.get("contexts", []):
        age = entry.get("cache_age_seconds")
        table.add_row(entry.get("name", ""), entry.get("state", ""), format_age(age) or "-")

    buffer = io.StringIO()
    Console(file=buffer, width=120).print(table)
    return f"{header}\n{buffer.getvalue().rstrip(chr(10))}"


def render_status(result: dict[str, Any]) -> str:
    if "contexts" in result:
        return _render_status_registry(result)
    return _render_status_context(result)
