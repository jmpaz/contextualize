from pathlib import Path
from typing import Any

GROUP_DELIMITER = "."
GROUP_PATH_KEY = "__group_path"
GROUP_BASE_KEY = "__group_base"
SET_KEY = "__is_set"

_DEFAULT_KEYS = {
    "wrap",
    "prefix",
    "suffix",
    "comment",
    "link-depth",
    "link-scope",
    "link-skip",
    "strip-paths",
    "gitignore",
    "arena",
    "atproto",
    "discord",
    "whatsapp",
    "soundcloud",
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
        entry: dict[str, Any],
        group_path: list[str],
        defaults: dict[str, Any],
        *,
        is_set: bool = False,
    ) -> None:
        kind = "Set" if is_set else "Component"
        if "group" in entry:
            raise ValueError(f"{kind} cannot define 'group'")
        if "components" in entry:
            raise ValueError(f"{kind} cannot define 'components' without 'group'")

        comp = dict(entry)
        if is_set:
            if "name" in comp:
                raise ValueError("Set cannot also define 'name'")
            name = comp.pop("set")
        else:
            name = comp.get("name")
        if name is None:
            name = next_auto_name()
        if not isinstance(name, str):
            raise ValueError(f"{kind} name must be a string")
        name = name.strip()
        if not name:
            raise ValueError(f"{kind} name must be a non-empty string")
        validate_name(name, kind=f"{kind} name", allow_delimiter=True)

        full_name = join_group_name(group_path, name)
        if full_name in used_names:
            raise ValueError(f"Duplicate component name: {full_name}")
        used_names.add(full_name)

        if is_set:
            for unsupported in ("repos", "manifests"):
                if unsupported in comp:
                    raise ValueError(
                        f"Set '{full_name}' does not support '{unsupported}'"
                    )
            if not comp.get("files"):
                raise ValueError(f"Set '{full_name}' must define 'files'")

        for key, value in defaults.items():
            if key not in comp:
                comp[key] = value

        comp["name"] = full_name
        if is_set:
            comp[SET_KEY] = True
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
                validate_name(group_name, kind="Group name", allow_delimiter=True)

                if "components" not in entry:
                    raise ValueError(f"Group '{group_name}' must define components")
                group_components = entry.get("components")
                if group_components is None:
                    group_components = []
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
            elif "set" in entry:
                add_component(entry, group_path, defaults, is_set=True)
            else:
                add_component(entry, group_path, defaults)

    process(components, [], {})
    return normalized


def coerce_mark_spec(mark: Any) -> dict[str, Any]:
    """Coerce one `marks:` entry into a record.

    Authored mistakes land in `problem`, never a raise (designed states,
    marks spec §4.3). YAML 1.1 reads unquoted `at: 4:12` as the sexagesimal
    int 252, so int/float arrivals re-render their authored form from
    canonical seconds.
    """
    from ..references.address import format_clock_time, parse_time_range

    record: dict[str, Any] = {
        "authored": None,
        "start_seconds": None,
        "end_seconds": None,
        "quote": None,
        "refs": [],
        "problem": None,
    }
    if not isinstance(mark, dict):
        record["authored"] = str(mark) if mark is not None else None
        record["problem"] = "mark-invalid"
        return record

    record["quote"] = _coerce_mark_quote(mark.get("quote"))
    record["refs"] = _coerce_mark_refs(mark.get("refs"))

    has_at = mark.get("at") is not None
    has_span = mark.get("span") is not None
    if has_at and has_span:
        record["problem"] = "mark-at-and-span"
        return record
    if not has_at and not has_span:
        record["problem"] = "mark-missing-time"
        return record

    value = mark["at"] if has_at else mark["span"]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        record["authored"] = str(value)
        record["problem"] = "mark-invalid-time"
        return record
    if isinstance(value, (int, float)):
        if value < 0:
            record["authored"] = str(value)
            record["problem"] = "mark-invalid-time"
            return record
        record["authored"] = format_clock_time(value)
        record["start_seconds"] = float(value)
    else:
        authored = value.strip()
        record["authored"] = authored
        span = parse_time_range(authored)
        if span is None:
            record["problem"] = "mark-invalid-time"
            return record
        start_seconds, end_seconds = span
        if end_seconds is not None and end_seconds < start_seconds:
            record["problem"] = "mark-invalid-time"
            return record
        record["start_seconds"] = start_seconds
        record["end_seconds"] = end_seconds

    if record["quote"] is not None and record["end_seconds"] is None:
        record["problem"] = "mark-quote-requires-range"
    return record


def _coerce_mark_quote(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _coerce_mark_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        ref = value.strip()
        return [ref] if ref else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def coerce_file_spec(spec: Any) -> tuple[str, dict[str, Any]]:
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
