from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Union

import yaml


@dataclass
class Component:
    name: str
    files: list[str] = field(default_factory=list)
    text: str | None = None
    prefix: str | None = None
    suffix: str | None = None
    wrap: str | None = None
    comment: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    group_path: tuple[str, ...] | None = None
    group_base: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Component":
        return cls(
            name=data.get("name", ""),
            files=data.get("files", []),
            text=data.get("text"),
            prefix=data.get("prefix"),
            suffix=data.get("suffix"),
            wrap=data.get("wrap"),
            comment=data.get("comment"),
            options={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "name",
                    "files",
                    "text",
                    "prefix",
                    "suffix",
                    "wrap",
                    "comment",
                    "__group_path",
                    "__group_base",
                }
            },
            group_path=data.get("__group_path"),
            group_base=data.get("__group_base"),
        )


@dataclass
class Manifest:
    config: dict[str, Any] = field(default_factory=dict)
    components: list[Component] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        from .normalize import extract_groups, normalize_components

        config = data.get("config", {})
        raw_components = data.get("components", [])
        normalized = normalize_components(raw_components)
        components = [Component.from_dict(c) for c in normalized]
        groups = extract_groups(normalized)
        return cls(config=config, components=components, groups=groups)


def parse_manifest(source: Union[str, Path, IO]) -> dict[str, Any]:
    if isinstance(source, Path):
        with open(source, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif isinstance(source, str):
        if "\n" not in source and Path(source).exists():
            with open(source, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(source)
    else:
        data = yaml.safe_load(source)

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError("Manifest must be a YAML mapping")

    if "components" not in data:
        data["components"] = []
    if "config" not in data:
        data["config"] = {}

    return data


def load_manifest(path: Union[str, Path]) -> dict[str, Any]:
    return parse_manifest(Path(path))
