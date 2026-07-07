"""Manifest source loading helpers."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ManifestFormat:
    body: str
    group_slices: dict[tuple[str, ...], str]


@dataclass(frozen=True)
class ManifestSource:
    data: dict[str, Any]
    manifest_cwd: str
    manifest_path: str | None
    source_format: ManifestFormat | None = None


_FENCE_RE = re.compile(
    r"(?ms)^```(?P<label>[A-Za-z0-9_-]*)[^\n]*\n(?P<body>.*?)^```[ \t]*$"
)


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
        require_entire_document=require_entire_document,
    )
    return data


def _load_manifest_text_with_format(
    text: str,
    *,
    source_label: str,
    require_entire_document: bool,
) -> tuple[dict[str, Any], ManifestFormat | None]:
    direct_error: Exception | None = None
    try:
        data = yaml.safe_load(text)
    except Exception as exc:
        direct_error = exc
    else:
        if _is_manifest_mapping(data):
            return data, _build_manifest_format(text)
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
            return data, _build_manifest_format(body)

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


def _build_manifest_format(body: str) -> ManifestFormat:
    filtered = _strip_commented_list_items(body)
    return ManifestFormat(
        body=filtered,
        group_slices=_extract_group_slices(filtered),
    )


def _strip_commented_list_items(text: str) -> str:
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    skip_indent: int | None = None
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" \t"))
        if skip_indent is not None:
            if not stripped:
                skip_indent = None
            elif stripped.startswith("#") and indent >= skip_indent:
                continue
            else:
                skip_indent = None
        if re.match(r"^[ \t]*#[ \t]*-[ \t]+", line):
            skip_indent = indent
            continue
        kept.append(line)
    return "".join(kept)


def _extract_group_slices(text: str) -> dict[tuple[str, ...], str]:
    lines = text.splitlines(keepends=True)
    components_range = _find_components_range(lines, 0, len(lines), max_indent=0)
    if components_range is None:
        return {}
    start, end = components_range
    group_slices: dict[tuple[str, ...], str] = {}
    _collect_group_slices(lines, start, end, (), group_slices)
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
    group_slices: dict[tuple[str, ...], str],
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
        group_slices[group_path] = _build_group_manifest_slice(
            lines[index:item_end], item_indent
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
                lines, child_start, child_end, group_path, group_slices
            )
        index = item_end


def _parse_group_name(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = yaml.safe_load(raw)
    except Exception:
        value = raw
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


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
