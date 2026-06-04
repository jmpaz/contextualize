"""Manifest source loading helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ManifestSource:
    data: dict[str, Any]
    manifest_cwd: str
    manifest_path: str | None


_FENCE_RE = re.compile(
    r"(?ms)^```(?P<label>[A-Za-z0-9_-]*)[^\n]*\n(?P<body>.*?)^```[ \t]*$"
)


def load_manifest_source(path: str | os.PathLike[str]) -> ManifestSource:
    source_path = Path(path).expanduser()
    with source_path.open("r", encoding="utf-8") as fh:
        text = fh.read()

    manifest_cwd = str(source_path.resolve().parent)
    data = load_manifest_text(
        text,
        source_label=str(source_path),
        require_entire_document=source_path.suffix.lower() in {".yaml", ".yml"},
    )
    return ManifestSource(
        data=data,
        manifest_cwd=manifest_cwd,
        manifest_path=str(source_path.resolve()),
    )


def load_manifest_text(
    text: str,
    *,
    source_label: str = "<manifest>",
    require_entire_document: bool = False,
) -> dict[str, Any]:
    direct_error: Exception | None = None
    try:
        data = yaml.safe_load(text)
    except Exception as exc:
        direct_error = exc
    else:
        if _is_manifest_mapping(data):
            return data
        if require_entire_document:
            raise ValueError(
                f"Manifest source {source_label} must be a mapping with 'config' and 'components'"
            )

    if require_entire_document and direct_error is not None:
        raise ValueError(f"Invalid YAML in {source_label}: {direct_error}") from direct_error

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
            return data

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


def _is_manifest_mapping(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("components"), list)
