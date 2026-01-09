from pathlib import Path
from typing import Any

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


def normalize_manifest_components(components: list[Any]) -> list[dict[str, Any]]:
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
