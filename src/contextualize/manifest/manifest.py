from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Dict, Tuple, Union

import yaml

GROUP_DELIMITER = "."
GROUP_PATH_KEY = "__group_path"
GROUP_BASE_KEY = "__group_base"

_DEFAULT_KEYS = {
    "wrap",
    "prefix",
    "suffix",
    "comment",
    "link-depth",
    "link-scope",
    "link-skip",
}
_GROUP_KEYS = {"group", "components", *_DEFAULT_KEYS}


def normalize_components(components: list[Any]) -> list[dict[str, Any]]:
    if not isinstance(components, list):
        raise ValueError("'components' must be a list")

    normalized: list[dict[str, Any]] = []
    used_names: set[str] = set()
    counter = 1

    def next_auto_name() -> str:
        nonlocal counter
        while True:
            candidate = f"component-{counter:03d}"
            counter += 1
            if candidate not in used_names:
                return candidate

    def validate_name(name: str, *, kind: str, allow_delimiter: bool) -> None:
        if not name or name in {".", ".."}:
            raise ValueError(f"{kind} must be a non-empty name")
        parts = Path(name).parts
        if len(parts) != 1 or "/" in name or "\\" in name:
            raise ValueError(f"{kind} must not contain path separators")
        if not allow_delimiter and GROUP_DELIMITER in name:
            raise ValueError(f"{kind} must not contain '{GROUP_DELIMITER}'")

    def join_group_name(group_path: list[str], name: str) -> str:
        if not group_path:
            return name
        return f"{GROUP_DELIMITER.join(group_path)}{GROUP_DELIMITER}{name}"

    def collect_group_defaults(entry: dict[str, Any]) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for key in _DEFAULT_KEYS:
            if key in entry and entry[key] is not None:
                defaults[key] = entry[key]
        return defaults

    def add_component(
        entry: dict[str, Any], group_path: list[str], defaults: dict[str, Any]
    ) -> None:
        if "group" in entry:
            raise ValueError("Component cannot define 'group'")
        if "components" in entry:
            raise ValueError("Component cannot define 'components' without 'group'")

        comp = dict(entry)
        name = comp.get("name")
        if name is None:
            name = next_auto_name()
        if not isinstance(name, str):
            raise ValueError("Component name must be a string")
        name = name.strip()
        if not name:
            raise ValueError("Component name must be a non-empty string")
        validate_name(name, kind="Component name", allow_delimiter=not group_path)

        full_name = join_group_name(group_path, name)
        if full_name in used_names:
            raise ValueError(f"Duplicate component name: {full_name}")
        used_names.add(full_name)

        for key, value in defaults.items():
            if key not in comp:
                comp[key] = value

        comp["name"] = full_name
        if group_path:
            comp[GROUP_PATH_KEY] = tuple(group_path)
            comp[GROUP_BASE_KEY] = name
        normalized.append(comp)

    def process(
        entries: list[Any], group_path: list[str], defaults: dict[str, Any]
    ) -> None:
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Components must be mappings")
            if "group" in entry:
                group_name = entry.get("group")
                if not isinstance(group_name, str):
                    raise ValueError("Group name must be a string")
                group_name = group_name.strip()
                if not group_name:
                    raise ValueError("Group name must be a non-empty string")
                validate_name(group_name, kind="Group name", allow_delimiter=False)

                if "components" not in entry:
                    raise ValueError(f"Group '{group_name}' must define components")
                group_components = entry.get("components")
                if not isinstance(group_components, list):
                    raise ValueError("Group components must be a list")

                extra_keys = set(entry) - _GROUP_KEYS
                if extra_keys:
                    unknown = ", ".join(sorted(extra_keys))
                    raise ValueError(
                        f"Group '{group_name}' has invalid keys: {unknown}"
                    )

                group_defaults = collect_group_defaults(entry)
                merged_defaults = dict(defaults)
                merged_defaults.update(group_defaults)
                process(group_components, group_path + [group_name], merged_defaults)
            else:
                add_component(entry, group_path, defaults)

    process(components, [], {})
    return normalized


def extract_groups(normalized_components: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for comp in normalized_components:
        group_path = comp.get(GROUP_PATH_KEY)
        if group_path:
            prefix = ""
            for part in group_path:
                prefix = part if not prefix else f"{prefix}{GROUP_DELIMITER}{part}"
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(comp["name"])
    return groups


def coerce_file_spec(spec: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(spec, dict):
        raw = spec.get("path") or spec.get("target") or spec.get("url")
        if not raw or not isinstance(raw, str):
            raise ValueError(
                f"Invalid file spec mapping; expected 'path' string: {spec}"
            )
        return raw, spec
    if isinstance(spec, str):
        return spec, {}
    raise ValueError(
        f"Invalid file spec; expected string or mapping, got: {type(spec)}"
    )


def component_selectors(comp: dict[str, Any]) -> set[str]:
    selectors: set[str] = set()
    name = comp.get("name")
    if isinstance(name, str) and name:
        selectors.add(name)
    group_path = comp.get(GROUP_PATH_KEY)
    if group_path:
        if isinstance(group_path, str):
            group_parts = [group_path]
        else:
            group_parts = list(group_path)
        prefix = ""
        for part in group_parts:
            prefix = part if not prefix else f"{prefix}{GROUP_DELIMITER}{part}"
            selectors.add(prefix)
    return selectors


def validate_manifest(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append("Manifest must be a mapping")
        return errors

    config = data.get("config", {})
    if not isinstance(config, dict):
        errors.append("'config' must be a mapping")

    components = data.get("components")
    if components is not None and not isinstance(components, list):
        errors.append("'components' must be a list")
    elif isinstance(components, list):
        for i, comp in enumerate(components):
            if not isinstance(comp, dict):
                errors.append(f"Component at index {i} must be a mapping")
                continue

            if "group" in comp:
                if not isinstance(comp.get("group"), str):
                    errors.append(f"Group at index {i} must have a string 'group' name")
                if "components" not in comp:
                    errors.append(f"Group at index {i} must define 'components'")
                elif not isinstance(comp.get("components"), list):
                    errors.append(f"Group at index {i} 'components' must be a list")
            else:
                if "files" not in comp and "text" not in comp:
                    errors.append(f"Component at index {i} must have 'files' or 'text'")
                files = comp.get("files")
                if files is not None and not isinstance(files, list):
                    errors.append(f"Component at index {i} 'files' must be a list")
    return errors


def validate_component(comp: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    name = comp.get("name")
    if not name:
        errors.append("Component must have a 'name'")
    elif not isinstance(name, str):
        errors.append("Component 'name' must be a string")

    files = comp.get("files")
    text = comp.get("text")

    if files is None and text is None:
        errors.append(f"Component '{name}' must have 'files' or 'text'")

    if files is not None:
        if not isinstance(files, list):
            errors.append(f"Component '{name}' 'files' must be a list")
        else:
            for i, f in enumerate(files):
                if not isinstance(f, (str, dict)):
                    errors.append(
                        f"Component '{name}' file at index {i} must be string or mapping"
                    )

    wrap = comp.get("wrap")
    if wrap is not None and wrap not in {"md", "xml", None}:
        errors.append(f"Component '{name}' 'wrap' must be 'md' or 'xml'")
    return errors


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
