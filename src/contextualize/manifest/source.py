"""Manifest source loading helpers."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml


@dataclass(frozen=True)
class ManifestSlice:
    body: str
    line: int


@dataclass(frozen=True)
class ManifestFormat:
    body: str
    line: int
    source_path: str | None
    group_slices: dict[tuple[str, ...], ManifestSlice]
    outline: list[dict[str, Any]]


@dataclass(frozen=True)
class ManifestSource:
    data: dict[str, Any]
    manifest_cwd: str
    manifest_path: str | None
    source_format: ManifestFormat | None = None


_FENCE_RE = re.compile(
    r"(?ms)^```(?P<label>[A-Za-z0-9_-]*)[^\n]*\n(?P<body>.*?)^```[ \t]*$"
)

_MEMBER_KEYS = ("files", "repos", "manifests")
_COMPONENT_KEY_NAMES = ("name", "group", "set")
_DASH_ITEM_RE = re.compile(r"^(?P<indent> *)-(?P<rest>.*)$")
_DISABLED_ITEM_RE = re.compile(r"^(?P<indent> *)#[ \t]*-[ \t]+(?P<rest>.*)$")


def load_manifest_source(path: str | os.PathLike[str]) -> ManifestSource:
    source_path = Path(path).expanduser()
    manifest_cwd = str(source_path.resolve().parent)
    if source_path.suffix.lower() == ".nix":
        return ManifestSource(
            data=_load_nix_manifest_source(source_path, manifest_cwd),
            manifest_cwd=manifest_cwd,
            manifest_path=str(source_path.resolve()),
        )

    with source_path.open("r", encoding="utf-8") as fh:
        text = fh.read()

    data, source_format = _load_manifest_text_with_format(
        text,
        source_label=str(source_path),
        source_path=str(source_path.resolve()),
        require_entire_document=source_path.suffix.lower() in {".yaml", ".yml"},
    )
    return ManifestSource(
        data=data,
        manifest_cwd=manifest_cwd,
        manifest_path=str(source_path.resolve()),
        source_format=source_format,
    )


def load_manifest_text(
    text: str,
    *,
    source_label: str = "<manifest>",
    require_entire_document: bool = False,
) -> dict[str, Any]:
    data, _source_format = _load_manifest_text_with_format(
        text,
        source_label=source_label,
        source_path=None,
        require_entire_document=require_entire_document,
    )
    return data


def _load_manifest_text_with_format(
    text: str,
    *,
    source_label: str,
    source_path: str | None,
    require_entire_document: bool,
) -> tuple[dict[str, Any], ManifestFormat | None]:
    direct_error: Exception | None = None
    try:
        data = yaml.safe_load(text)
    except Exception as exc:
        direct_error = exc
    else:
        if _is_manifest_mapping(data):
            return data, _build_manifest_format(text, source_path=source_path, line=1)
        if require_entire_document:
            raise ValueError(
                f"Manifest source {source_label} must be a mapping with 'config' and 'components'"
            )

    if require_entire_document and direct_error is not None:
        raise ValueError(
            f"Invalid YAML in {source_label}: {direct_error}"
        ) from direct_error

    labeled_errors: list[str] = []
    for match in _FENCE_RE.finditer(text):
        label = match.group("label").lower()
        if label not in {"yaml", "yml"}:
            continue
        body = match.group("body")
        try:
            data = yaml.safe_load(body)
        except Exception as exc:
            labeled_errors.append(str(exc))
            continue
        if _is_manifest_mapping(data):
            line = text[: match.start("body")].count("\n") + 1
            return data, _build_manifest_format(
                body,
                source_path=source_path,
                line=line,
            )

    if labeled_errors:
        raise ValueError(
            f"No valid manifest YAML block found in {source_label}: {labeled_errors[0]}"
        )
    raise ValueError(f"No contextualize manifest found in {source_label}")


def path_contains_manifest(path: str | os.PathLike[str]) -> bool:
    try:
        load_manifest_source(path)
    except (OSError, ValueError):
        return False
    return True


def _load_nix_manifest_source(path: Path, cwd: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nix", "eval", "--json", "--file", str(path.resolve())],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise ValueError("Nix is required to load .nix manifests") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise ValueError(f"Failed to evaluate Nix manifest {path}{suffix}") from exc

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Nix manifest {path} did not evaluate to JSON") from exc
    if not _is_manifest_mapping(data):
        raise ValueError(
            f"Nix manifest {path} must evaluate to a mapping with 'components'"
        )
    return data


def _is_manifest_mapping(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("components"), list)


def _build_manifest_format(
    body: str,
    *,
    source_path: str | None,
    line: int,
) -> ManifestFormat:
    lines = body.splitlines(keepends=True)
    return ManifestFormat(
        body=body,
        line=line,
        source_path=source_path,
        group_slices=_extract_group_slices(lines, line),
        outline=_build_outline(lines, line),
    )


# --- group manifest slicing (hydrated `.context/<name>/**/manifest.yaml` copies) ---


def _extract_group_slices(
    lines: list[str],
    start_line: int,
) -> dict[tuple[str, ...], ManifestSlice]:
    components_range = _find_components_range(lines, 0, len(lines), max_indent=0)
    if components_range is None:
        return {}
    start, end = components_range
    group_slices: dict[tuple[str, ...], ManifestSlice] = {}
    _collect_group_slices(lines, start, end, (), group_slices, start_line)
    return group_slices


def _find_components_range(
    lines: list[str],
    start: int,
    end: int,
    *,
    max_indent: int | None = None,
) -> tuple[int, int] | None:
    for index in range(start, end):
        match = re.match(r"^(?P<indent> *)components\s*:\s*(?:#.*)?$", lines[index])
        if not match:
            continue
        indent = len(match.group("indent"))
        if max_indent is not None and indent > max_indent:
            continue
        return index + 1, _find_block_end(lines, index + 1, end, indent)
    return None


def _collect_group_slices(
    lines: list[str],
    start: int,
    end: int,
    parent_path: tuple[str, ...],
    group_slices: dict[tuple[str, ...], ManifestSlice],
    start_line: int,
) -> None:
    index = start
    while index < end:
        match = re.match(
            r"^(?P<indent> *)-\s+group\s*:\s*(?P<name>[^#\n]+?)(?:\s+#.*)?$",
            lines[index],
        )
        if not match:
            index += 1
            continue

        item_indent = len(match.group("indent"))
        item_end = _find_block_end(lines, index + 1, end, item_indent)
        group_name = _parse_group_name(match.group("name"))
        if group_name is None:
            index = item_end
            continue

        group_path = parent_path + (group_name,)
        group_slices[group_path] = ManifestSlice(
            body=_build_group_manifest_slice(lines[index:item_end], item_indent),
            line=start_line + index,
        )
        child_range = _find_components_range(
            lines,
            index + 1,
            item_end,
            max_indent=None,
        )
        if child_range is not None:
            child_start, child_end = child_range
            _collect_group_slices(
                lines,
                child_start,
                child_end,
                group_path,
                group_slices,
                start_line,
            )
        index = item_end


def _parse_group_name(raw: str) -> str | None:
    return _parse_scalar(raw)


def _build_group_manifest_slice(lines: list[str], item_indent: int) -> str:
    remove = max(0, item_indent - 2)
    shifted = []
    for line in lines:
        if line.startswith(" " * remove):
            shifted.append(line[remove:])
        else:
            shifted.append(line)
    text = "components:\n" + "".join(shifted)
    return text if text.endswith("\n") else f"{text}\n"


def _find_block_end(
    lines: list[str],
    start: int,
    end: int,
    parent_indent: int,
) -> int:
    for index in range(start, end):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= parent_indent:
            return index
    return end


# --- authored outline: order, comments, disabled members (§3.1, §3.2) ---


def _build_outline(lines: list[str], start_line: int) -> list[dict[str, Any]]:
    top = _find_components_range(lines, 0, len(lines), max_indent=0)
    if top is None:
        return []
    start, end = top
    indent = _first_item_indent(lines, start, end)
    if indent is None:
        return []
    return _parse_component_list(lines, start, end, indent, start_line)


def iter_active_leaves(nodes: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Depth-first walk yielding non-disabled component/set nodes.

    Mirrors the traversal order of `manifest.normalize_components`, so the
    Nth node yielded here corresponds to the Nth entry of its flattened
    component list.
    """
    for node in nodes:
        if node.get("disabled"):
            continue
        if node["kind"] == "group":
            yield from iter_active_leaves(node.get("children", []))
        else:
            yield node


def _find_item_end(lines: list[str], start: int, end: int, parent_indent: int) -> int:
    """Find where a component/member item's own span ends.

    Unlike `_find_block_end` (used for group-slice text boundaries, which
    treats every comment as trailing filler regardless of indent), this
    stops at a same-or-shallower comment line too -- such a line is a
    sibling's block comment or a disabled sibling, not this item's content.
    """
    for index in range(start, end):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if indent <= parent_indent:
            return index
    return end


def _first_item_indent(lines: list[str], start: int, end: int) -> int | None:
    for index in range(start, end):
        raw = lines[index].rstrip("\n")
        stripped = raw.strip()
        if not stripped:
            continue
        m = _DASH_ITEM_RE.match(raw)
        if m:
            return len(m.group("indent"))
        if stripped.startswith("#"):
            continue
        return None
    return None


def _parse_component_list(
    lines: list[str],
    start: int,
    end: int,
    indent: int,
    start_line: int,
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    index = start
    order = 0
    pending: list[tuple[int, str]] = []
    while index < end:
        raw = lines[index].rstrip("\n")
        bare = raw.strip()
        if not bare:
            pending = []
            index += 1
            continue
        line_indent = len(raw) - len(raw.lstrip(" "))
        if line_indent != indent:
            index += 1
            continue

        disabled_match = _DISABLED_ITEM_RE.match(raw)
        if disabled_match:
            item_end = _consume_disabled_block(lines, index, end, indent)
            comment = _reduce_pending(pending)
            pending = []
            nodes.append(
                _build_disabled_component_node(
                    lines, index, item_end, start_line, order, comment
                )
            )
            order += 1
            index = item_end
            continue

        if bare.startswith("#"):
            pending.append((start_line + index, bare.lstrip("#").strip()))
            index += 1
            continue

        item_match = _DASH_ITEM_RE.match(raw)
        if not item_match:
            index += 1
            continue

        item_end = _find_item_end(lines, index + 1, end, indent)
        comment = _reduce_pending(pending)
        pending = []
        nodes.append(
            _build_component_node(lines, index, item_end, start_line, order, comment)
        )
        order += 1
        index = item_end

    return nodes


def _consume_disabled_block(lines: list[str], start: int, end: int, indent: int) -> int:
    index = start + 1
    while index < end:
        line = lines[index].rstrip("\n")
        bare = line.strip()
        if not bare:
            break
        line_indent = len(line) - len(line.lstrip(" \t"))
        if bare.startswith("#") and line_indent >= indent:
            index += 1
            continue
        break
    return index


def _reduce_pending(pending: list[tuple[int, str]]) -> dict[str, Any] | None:
    if not pending:
        return None
    text = "\n".join(t for _, t in pending)
    return {
        "text": text,
        "line_start": pending[0][0],
        "line_end": pending[-1][0],
    }


def _disabled_raw_text(lines: list[str], start: int, end: int) -> str:
    out = []
    for index in range(start, end):
        line = lines[index].rstrip("\n")
        match = re.match(r"^ *#[ \t]?(?P<rest>.*)$", line)
        out.append(match.group("rest") if match else line.strip())
    return "\n".join(out)


def _build_disabled_component_node(
    lines: list[str],
    start: int,
    end: int,
    start_line: int,
    order: int,
    comment: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_text = _disabled_raw_text(lines, start, end)
    kind, name = _classify_disabled_component(raw_text)
    return {
        "kind": kind,
        "name": name,
        "order": order,
        "disabled": True,
        "line_start": start_line + start,
        "line_end": start_line + end - 1,
        "comment": comment,
        "inline_comment": None,
        "members": {},
        "children": [],
        "raw": raw_text,
    }


def _classify_disabled_component(raw_text: str) -> tuple[str, str | None]:
    first = raw_text.splitlines()[0].strip() if raw_text else ""
    if first.startswith("-"):
        first = first[1:].strip()
    match = re.match(r"^(?P<key>[A-Za-z0-9_-]+)\s*:\s*(?P<value>.*)$", first)
    if match and match.group("key") in _COMPONENT_KEY_NAMES:
        key = match.group("key")
        kind = "group" if key == "group" else ("set" if key == "set" else "component")
        return kind, _parse_scalar(match.group("value"))
    return "component", None


def _build_component_node(
    lines: list[str],
    start: int,
    end: int,
    start_line: int,
    order: int,
    comment: dict[str, Any] | None,
) -> dict[str, Any]:
    first_line = lines[start].rstrip("\n")
    _, inline_comment = _split_inline_comment(first_line)
    indent = len(first_line) - len(first_line.lstrip(" "))
    key, name = _detect_item_key(lines, start, end, indent)
    line_start = start_line + start
    line_end = start_line + end - 1

    if key == "group":
        children: list[dict[str, Any]] = []
        child_range = _find_components_range(lines, start + 1, end, max_indent=None)
        if child_range is not None:
            child_start, child_end = child_range
            child_indent = _first_item_indent(lines, child_start, child_end)
            if child_indent is not None:
                children = _parse_component_list(
                    lines, child_start, child_end, child_indent, start_line
                )
        return {
            "kind": "group",
            "name": name,
            "order": order,
            "disabled": False,
            "line_start": line_start,
            "line_end": line_end,
            "comment": comment,
            "inline_comment": inline_comment,
            "members": {},
            "children": children,
        }

    kind = "set" if key == "set" else "component"
    members = _parse_all_member_lists(lines, start, end, indent, start_line)
    return {
        "kind": kind,
        "name": name,
        "order": order,
        "disabled": False,
        "line_start": line_start,
        "line_end": line_end,
        "comment": comment,
        "inline_comment": inline_comment,
        "members": members,
        "children": [],
    }


def _detect_item_key(
    lines: list[str], start: int, end: int, indent: int
) -> tuple[str, str | None]:
    first_line = lines[start].rstrip("\n")
    code, _ = _split_inline_comment(first_line)
    match = re.match(r"^ *-\s*(?P<key>[A-Za-z0-9_-]+)\s*:\s*(?P<value>.*)$", code)
    if match and match.group("key") in _COMPONENT_KEY_NAMES:
        return match.group("key"), _parse_scalar(match.group("value"))

    child_indent: int | None = None
    for index in range(start + 1, end):
        line = lines[index].rstrip("\n")
        bare = line.strip()
        if not bare or bare.startswith("#"):
            continue
        line_indent = len(line) - len(line.lstrip(" "))
        if line_indent <= indent:
            break
        if child_indent is None:
            child_indent = line_indent
        if line_indent != child_indent:
            continue
        code, _ = _split_inline_comment(line)
        match = re.match(r"^ *(?P<key>[A-Za-z0-9_-]+)\s*:\s*(?P<value>.*)$", code)
        if match and match.group("key") in _COMPONENT_KEY_NAMES:
            return match.group("key"), _parse_scalar(match.group("value"))

    return "name", None


def _parse_all_member_lists(
    lines: list[str],
    start: int,
    end: int,
    indent: int,
    start_line: int,
) -> dict[str, list[dict[str, Any]]]:
    members: dict[str, list[dict[str, Any]]] = {}
    for key in _MEMBER_KEYS:
        list_range = _find_named_list_range(lines, start + 1, end, key, indent)
        if list_range is None:
            continue
        list_start, list_end = list_range
        item_indent = _first_item_indent(lines, list_start, list_end)
        if item_indent is None:
            continue
        parsed = _parse_member_items(lines, list_start, list_end, item_indent, start_line)
        if parsed:
            members[key] = parsed
    return members


def _find_named_list_range(
    lines: list[str],
    start: int,
    end: int,
    key: str,
    parent_indent: int,
) -> tuple[int, int] | None:
    pattern = re.compile(rf"^(?P<indent> *){key}\s*:\s*(?:#.*)?$")
    for index in range(start, end):
        line = lines[index].rstrip("\n")
        match = pattern.match(line)
        if not match:
            continue
        list_indent = len(match.group("indent"))
        if list_indent <= parent_indent:
            continue
        return index + 1, _find_block_end(lines, index + 1, end, list_indent)
    return None


def _parse_member_items(
    lines: list[str],
    start: int,
    end: int,
    indent: int,
    start_line: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    index = start
    order = 0
    pending: list[tuple[int, str]] = []
    while index < end:
        raw = lines[index].rstrip("\n")
        bare = raw.strip()
        if not bare:
            pending = []
            index += 1
            continue
        line_indent = len(raw) - len(raw.lstrip(" "))
        if line_indent != indent:
            index += 1
            continue

        disabled_match = _DISABLED_ITEM_RE.match(raw)
        if disabled_match:
            item_end = _consume_disabled_block(lines, index, end, indent)
            comment = _reduce_pending(pending)
            pending = []
            items.append(
                {
                    "order": order,
                    "disabled": True,
                    "line_start": start_line + index,
                    "line_end": start_line + item_end - 1,
                    "comment": comment,
                    "inline_comment": None,
                    "raw": _disabled_raw_text(lines, index, item_end),
                }
            )
            order += 1
            index = item_end
            continue

        if bare.startswith("#"):
            pending.append((start_line + index, bare.lstrip("#").strip()))
            index += 1
            continue

        item_match = _DASH_ITEM_RE.match(raw)
        if not item_match:
            index += 1
            continue

        item_end = _find_item_end(lines, index + 1, end, indent)
        _, inline_comment = _split_inline_comment(raw)
        comment = _reduce_pending(pending)
        pending = []
        items.append(
            {
                "order": order,
                "disabled": False,
                "line_start": start_line + index,
                "line_end": start_line + item_end - 1,
                "comment": comment,
                "inline_comment": inline_comment,
            }
        )
        order += 1
        index = item_end

    return items


def _split_inline_comment(line: str) -> tuple[str, str | None]:
    in_single = False
    in_double = False
    for index, ch in enumerate(line):
        if in_single:
            if ch == "'":
                in_single = False
        elif in_double:
            if ch == '"':
                in_double = False
        elif ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        elif ch == "#" and (index == 0 or line[index - 1] in " \t"):
            code = line[:index].rstrip()
            comment = line[index + 1 :].strip()
            return code, (comment or None)
    return line, None


def _parse_scalar(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = yaml.safe_load(raw)
    except Exception:
        return raw
    return value if isinstance(value, str) else raw
