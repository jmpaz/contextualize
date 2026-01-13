from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
import hashlib
import os
import re
import shutil
import stat

from ..git.cache import ensure_repo, expand_git_paths, parse_git_target
from .manifest import (
    GROUP_BASE_KEY,
    GROUP_PATH_KEY,
    coerce_file_spec,
    normalize_components,
)
from ..references import URLReference, create_file_references
from ..references.helpers import (
    is_http_url,
    parse_git_url_target,
    parse_target_spec,
    resolve_symbol_ranges,
    split_spec_symbols,
)
from ..utils import extract_ranges


@dataclass(frozen=True)
class HydrateOverrides:
    context_dir: str | None = None
    access: str | None = None
    path_strategy: str | None = None
    agents_prompt: str | None = None
    agents_filenames: tuple[str, ...] = ()
    omit_meta: bool = False


@dataclass(frozen=True)
class HydrateResult:
    context_dir: str
    component_count: int
    file_count: int
    manifest_written: bool


@dataclass
class ResolvedItem:
    source_type: str
    source_ref: str
    source_rev: str | None
    source_path: str
    context_subpath: str
    content: str
    manifest_spec: str


@dataclass(frozen=True)
class HydratePlan:
    context_dir: Path
    files_to_write: list[tuple[Path, str]]
    used_paths: set[str]
    component_count: int
    include_meta: bool
    access: str


def apply_hydration_plan(plan: HydratePlan) -> HydrateResult:
    _write_files(plan.files_to_write)
    if plan.access == "read-only":
        _apply_read_only(plan.context_dir)

    file_count = len({path.as_posix() for path, _ in plan.files_to_write})
    return HydrateResult(
        context_dir=str(plan.context_dir),
        component_count=plan.component_count,
        file_count=file_count,
        manifest_written=plan.include_meta,
    )


def plan_matches_existing(plan: HydratePlan) -> bool:
    context_dir = plan.context_dir
    if not context_dir.exists() or not context_dir.is_dir():
        return False

    expected_hashes: dict[str, tuple[int, str]] = {}
    expected_dirs: set[str] = set()
    for path, content in plan.files_to_write:
        rel = path.relative_to(context_dir).as_posix()
        data = content.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        expected_hashes[rel] = (len(data), digest)
        _collect_parent_dirs(rel, expected_dirs)

    seen_files: set[str] = set()
    for root, dirs, files in os.walk(context_dir):
        rel_root = os.path.relpath(root, context_dir)
        if rel_root != "." and rel_root not in expected_dirs:
            return False
        for name in files:
            file_path = Path(root) / name
            rel_file = file_path.relative_to(context_dir).as_posix()
            expected = expected_hashes.get(rel_file)
            if expected is None:
                return False
            size, digest = expected
            try:
                if file_path.stat().st_size != size:
                    return False
            except OSError:
                return False
            if _hash_file(file_path) != digest:
                return False
            seen_files.add(rel_file)
        for name in dirs:
            dir_path = Path(root) / name
            rel_dir = dir_path.relative_to(context_dir).as_posix()
            if rel_dir not in expected_dirs:
                return False

    if seen_files != set(expected_hashes.keys()):
        return False

    if plan.access == "read-only":
        if not _all_read_only(context_dir, expected_dirs, set(expected_hashes.keys())):
            return False

    return True


def clear_context_dir(path: Path) -> None:
    _clear_context_dir(path)


_RANGE_RE = re.compile(r"^\s*L?(\d+)\s*(?:-|:)\s*L?(\d+)\s*$")


def hydrate_manifest(
    manifest_path: str,
    *,
    overrides: HydrateOverrides,
    cwd: str,
) -> HydrateResult:
    plan = build_hydration_plan(
        manifest_path,
        overrides=overrides,
        cwd=cwd,
    )
    return apply_hydration_plan(plan)


def hydrate_manifest_data(
    data: dict[str, Any],
    manifest_cwd: str,
    *,
    manifest_path: str | None = None,
    overrides: HydrateOverrides,
    cwd: str,
) -> HydrateResult:
    plan = build_hydration_plan_data(
        data,
        manifest_cwd,
        manifest_path=manifest_path,
        overrides=overrides,
        cwd=cwd,
    )
    return apply_hydration_plan(plan)


def build_hydration_plan(
    manifest_path: str,
    *,
    overrides: HydrateOverrides,
    cwd: str,
) -> HydratePlan:
    import yaml

    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping with 'config' and 'components'")
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    return build_hydration_plan_data(
        data,
        manifest_dir,
        manifest_path=manifest_path,
        overrides=overrides,
        cwd=cwd,
    )


def build_hydration_plan_data(
    data: dict[str, Any],
    manifest_cwd: str,
    *,
    manifest_path: str | None = None,
    overrides: HydrateOverrides,
    cwd: str,
) -> HydratePlan:
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping with 'config' and 'components'")

    cfg = data.get("config") or {}
    if not isinstance(cfg, dict):
        raise ValueError("'config' must be a mapping")

    components = data.get("components")
    if not isinstance(components, list):
        raise ValueError("'components' must be a list")
    components = normalize_components(components)

    base_dir = _resolve_base_dir(cfg, manifest_cwd, manifest_path)
    context_cfg = _resolve_context_config(cfg, overrides, cwd)
    context_dir = context_cfg["dir"]
    use_external_root = context_cfg[
        "path_strategy"
    ] == "on-disk" and _manifest_has_local_sources(components)
    flatten_groups: set[tuple[str, ...]] = set()
    if context_cfg["path_strategy"] == "by-component":
        flatten_groups = _find_flatten_groups(components)

    used_paths: set[str] = set()
    identity_paths: dict[tuple[Any, ...], Path] = {}
    files_to_write: list[tuple[Path, str]] = []
    index_components: dict[str, list[dict[str, Any]]] = {}
    normalized_components: list[dict[str, Any]] = []

    global_strip_prefix: Path | None = None
    if context_cfg["path_strategy"] == "by-component":
        global_strip_prefix = _find_global_subpath_prefix(components, base_dir)

    if context_cfg["agents_text"] is not None:
        _queue_agent_files(
            files_to_write,
            context_cfg["dir"],
            context_cfg["agents_files"] or ["AGENTS.md"],
            context_cfg["agents_text"],
        )

    for comp in components:
        comp_name = comp["name"]
        comp_files = comp.get("files")
        comp_text = comp.get("text")
        comp_prefix = comp.get("prefix")
        comp_suffix = comp.get("suffix")
        component_root = _build_component_root(
            comp_name,
            comp.get(GROUP_PATH_KEY),
            comp.get(GROUP_BASE_KEY),
            context_cfg["path_strategy"],
            flatten_groups=flatten_groups,
        )

        if (
            not comp_files
            and comp_text is None
            and comp_prefix is None
            and comp_suffix is None
        ):
            raise ValueError(f"Component '{comp_name}' has no content")

        normalized_comp = _build_normalized_component(comp, comp_name)
        normalized_files: list[Any] = []

        for note_name, note_value in (
            ("text-001.md", comp_text),
            ("prefix.md", comp_prefix),
            ("suffix.md", comp_suffix),
        ):
            if note_value is None:
                continue
            if not isinstance(note_value, str):
                raise ValueError(
                    f"Component '{comp_name}' note '{note_name}' must be a string"
                )
            note_root = component_root or Path(comp_name)
            note_path = context_dir / note_root / "notes" / note_name
            files_to_write.append((note_path, note_value))

        if comp_files is not None and not isinstance(comp_files, list):
            raise ValueError(f"Component '{comp_name}' files must be a list")
        if comp_files:
            resolved_items: list[
                tuple[
                    ResolvedItem,
                    str,
                    str | None,
                    list[tuple[int, int]] | None,
                    list[str] | None,
                    list[tuple[int, int]] | None,
                    str | None,
                    dict[str, Any],
                ]
            ] = []
            for file_spec in comp_files:
                raw_spec, file_opts = coerce_file_spec(file_spec)
                spec_comment = _parse_comment(file_opts.pop("comment", None))
                range_value = file_opts.pop("range", None)
                symbols_value = file_opts.pop("symbols", None)
                range_spec = _parse_range_value(range_value)
                symbols_spec = _parse_symbols_value(symbols_value)
                raw_spec, path_symbols = split_spec_symbols(raw_spec)
                if path_symbols:
                    symbols_spec = _merge_symbols(symbols_spec, path_symbols)

                for item in _resolve_spec_items(
                    raw_spec,
                    base_dir,
                    component_name=comp_name,
                    filename_hint=file_opts.get("filename"),
                ):
                    ranges = range_spec[:] if range_spec else None
                    symbols = symbols_spec[:] if symbols_spec else None

                    if symbols:
                        ranges, symbols, should_skip = resolve_symbol_ranges(
                            item.context_subpath,
                            symbols,
                            text=item.content,
                            ranges=ranges,
                            warn_label=item.context_subpath,
                            append_to_ranges=True,
                            keep_missing=False,
                            skip_on_missing=True,
                            warn_on_partial=False,
                        )
                        if should_skip:
                            continue

                    content = (
                        extract_ranges(item.content, ranges) if ranges else item.content
                    )
                    suffix = _build_suffix(ranges, symbols)
                    resolved_items.append(
                        (
                            item,
                            content,
                            suffix,
                            ranges,
                            symbols,
                            range_spec,
                            spec_comment,
                            file_opts,
                        )
                    )

            for (
                item,
                content,
                suffix,
                ranges,
                symbols,
                range_spec,
                spec_comment,
                file_opts,
            ) in resolved_items:
                rel_path, should_write = _resolve_context_path(
                    comp_name,
                    item,
                    suffix,
                    used_paths,
                    identity_paths,
                    context_cfg["path_strategy"],
                    ranges,
                    symbols,
                    component_root=component_root,
                    use_external_root=use_external_root,
                    strip_prefix=global_strip_prefix,
                )
                if should_write:
                    files_to_write.append((context_dir / rel_path, content))
                index_components.setdefault(comp_name, []).append(
                    _build_index_entry(
                        rel_path,
                        item,
                        ranges,
                        symbols,
                        content,
                    )
                )
                normalized_files.append(
                    _build_manifest_file_entry(
                        item,
                        range_spec,
                        symbols,
                        spec_comment,
                        file_opts,
                    )
                )

        if normalized_files:
            normalized_comp["files"] = normalized_files
        normalized_components.append(normalized_comp)

    if context_cfg["include_meta"]:
        normalized_manifest = {
            "config": _build_normalized_config(cfg, context_cfg, base_dir),
            "components": normalized_components,
        }
        manifest_text = _dump_manifest(normalized_manifest)
        files_to_write.append((context_dir / "manifest.yaml", manifest_text))

    if context_cfg["include_meta"]:
        index_data = {"version": 1, "components": index_components}
        index_text = _dump_index(index_data)
        files_to_write.append((context_dir / "index.json", index_text))

    return HydratePlan(
        context_dir=context_dir,
        files_to_write=files_to_write,
        used_paths=used_paths,
        component_count=len(components),
        include_meta=context_cfg["include_meta"],
        access=context_cfg["access"],
    )


def _resolve_base_dir(
    cfg: dict[str, Any], manifest_cwd: str, manifest_path: str | None
) -> str:
    if "root" in cfg:
        raw_root = cfg.get("root") or "~"
        base_dir = os.path.expanduser(raw_root)
    else:
        base_dir = manifest_cwd if manifest_path or manifest_cwd else os.getcwd()
    return os.path.abspath(base_dir)


def _resolve_context_config(
    cfg: dict[str, Any], overrides: HydrateOverrides, cwd: str
) -> dict[str, Any]:
    context_cfg = cfg.get("context") or {}
    if not isinstance(context_cfg, dict):
        raise ValueError("'config.context' must be a mapping")

    raw_access = overrides.access or context_cfg.get("access") or "writable"
    if not isinstance(raw_access, str):
        raise ValueError("context access must be a string")
    access = raw_access.lower()
    if access not in {"read-only", "writable"}:
        raise ValueError("context access must be 'read-only' or 'writable'")

    raw_dir_value = overrides.context_dir or context_cfg.get("dir") or ".context"
    dir_value = str(raw_dir_value)
    dir_path = Path(os.path.expanduser(dir_value))
    if not dir_path.is_absolute():
        dir_path = Path(os.path.abspath(os.path.join(cwd, dir_path)))

    raw_path_strategy = (
        overrides.path_strategy or context_cfg.get("path-strategy") or "on-disk"
    )
    if not isinstance(raw_path_strategy, str):
        raise ValueError("path-strategy must be a string")
    path_strategy = raw_path_strategy.lower()
    if path_strategy not in {"on-disk", "by-component"}:
        raise ValueError("path-strategy must be 'on-disk' or 'by-component'")

    include_meta_value = context_cfg.get("include-meta", True)
    if not isinstance(include_meta_value, bool):
        raise ValueError("include-meta must be a boolean")
    include_meta = False if overrides.omit_meta else include_meta_value

    agents_text, agents_files = _resolve_agents(context_cfg, overrides)
    return {
        "dir": dir_path,
        "dir_value": dir_value,
        "access": access,
        "include_meta": bool(include_meta),
        "path_strategy": path_strategy,
        "agents_text": agents_text,
        "agents_files": agents_files,
    }


def _resolve_agents(
    context_cfg: dict[str, Any], overrides: HydrateOverrides
) -> tuple[str | None, list[str] | None]:
    if overrides.agents_prompt is not None:
        files = list(overrides.agents_filenames) or ["AGENTS.md"]
        return overrides.agents_prompt, files

    agents_cfg = context_cfg.get("agents")
    if not isinstance(agents_cfg, dict):
        return None, None
    text = agents_cfg.get("text")
    if text is None:
        return None, None
    if not isinstance(text, str):
        raise ValueError("context.agents.text must be a string")
    files = agents_cfg.get("files") or ["AGENTS.md"]
    if isinstance(files, str):
        files = [files]
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        raise ValueError("context.agents.files must be a list of strings")
    if not files:
        files = ["AGENTS.md"]
    return text, files


def _queue_agent_files(
    files: list[tuple[Path, str]],
    context_dir: Path,
    filenames: list[str],
    content: str,
) -> None:
    for filename in filenames:
        _validate_agent_filename(filename)
        files.append((context_dir / filename, content))


def _validate_agent_filename(name: str) -> None:
    if not name or name in {".", ".."}:
        raise ValueError("Agent filename must be a non-empty name")
    if "/" in name or "\\" in name:
        raise ValueError("Agent filename must not contain path separators")


def _assign_component_names(components: list[dict[str, Any]]) -> None:
    used = set()
    for comp in components:
        if not isinstance(comp, dict):
            raise ValueError("Components must be mappings")
        name = comp.get("name")
        if name is None:
            continue
        if not isinstance(name, str):
            raise ValueError("Component name must be a string")
        _validate_component_name(name)
        if name in used:
            raise ValueError(f"Duplicate component name: {name}")
        used.add(name)

    counter = 1
    for comp in components:
        if comp.get("name"):
            continue
        while True:
            candidate = f"component-{counter:03d}"
            counter += 1
            if candidate not in used:
                break
        _validate_component_name(candidate)
        comp["name"] = candidate
        used.add(candidate)


def _validate_component_name(name: str) -> None:
    parts = Path(name).parts
    if len(parts) != 1 or name in {".", ".."}:
        raise ValueError(f"Invalid component name: {name}")
    if "/" in name or "\\" in name:
        raise ValueError(f"Invalid component name: {name}")


def _build_normalized_component(comp: dict[str, Any], name: str) -> dict[str, Any]:
    normalized = {
        k: v
        for k, v in comp.items()
        if k not in {"files", "name", GROUP_PATH_KEY, GROUP_BASE_KEY} and v is not None
    }
    normalized["name"] = name
    return normalized


def _find_common_subpath_prefix(subpaths: list[str]) -> Path | None:
    if not subpaths:
        return None

    parents = [Path(sp).parent for sp in subpaths]
    if any(p == Path(".") or not p.parts for p in parents):
        return None

    if len(parents) == 1:
        return parents[0] if parents[0].parts else None

    first = parents[0].parts
    common_parts: list[str] = []
    for i, part in enumerate(first):
        if all(len(p.parts) > i and p.parts[i] == part for p in parents):
            common_parts.append(part)
        else:
            break

    if not common_parts:
        return None

    return Path(*common_parts)


def _find_global_subpath_prefix(
    components: list[dict[str, Any]], base_dir: str
) -> Path | None:
    all_subpaths: list[str] = []
    for comp in components:
        comp_name = comp["name"]
        comp_files = comp.get("files")
        if not comp_files or not isinstance(comp_files, list):
            continue
        for file_spec in comp_files:
            raw_spec, file_opts = coerce_file_spec(file_spec)
            raw_spec, _ = split_spec_symbols(raw_spec)
            try:
                for item in _resolve_spec_items(
                    raw_spec,
                    base_dir,
                    component_name=comp_name,
                    filename_hint=file_opts.get("filename"),
                ):
                    all_subpaths.append(item.context_subpath)
            except (FileNotFoundError, ValueError):
                pass
    return _find_common_subpath_prefix(all_subpaths)


def _find_flatten_groups(components: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for comp in components:
        group_path = comp.get(GROUP_PATH_KEY)
        if not group_path:
            continue
        grouped.setdefault(tuple(group_path), []).append(comp)

    flatten: set[tuple[str, ...]] = set()
    for group_path, comps in grouped.items():
        if len(comps) < 2:
            continue
        shared_root: str | None = None
        for comp in comps:
            root = _component_external_root(comp)
            if root is None:
                shared_root = None
                break
            if shared_root is None:
                shared_root = root
            elif shared_root != root:
                shared_root = None
                break
        if shared_root is not None:
            flatten.add(group_path)
    return flatten


def _component_external_root(comp: dict[str, Any]) -> str | None:
    if comp.get("text") is not None or comp.get("prefix") is not None:
        return None
    if comp.get("suffix") is not None:
        return None
    files = comp.get("files")
    if not files or not isinstance(files, list):
        return None
    root: str | None = None
    for file_spec in files:
        raw_spec, _ = coerce_file_spec(file_spec)
        key = _external_root_key(raw_spec)
        if key is None:
            return None
        if root is None:
            root = key
        elif root != key:
            return None
    return root


def _external_root_key(raw_spec: str) -> str | None:
    spec = os.path.expanduser(raw_spec)
    if is_http_url(spec):
        opts = parse_target_spec(spec)
        url = opts.get("target", spec)
        tgt = parse_git_url_target(url)
        if tgt:
            rev = tgt.rev or ""
            return f"git:{tgt.repo_url}@{rev}"
        return f"http:{_url_origin(url)}"
    tgt = parse_git_target(spec)
    if tgt:
        rev = tgt.rev or ""
        return f"git:{tgt.repo_url}@{rev}"
    return None


def _url_origin(url: str) -> str:
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc or parsed.hostname or ""
    if netloc:
        return f"{scheme}://{netloc}"
    return url


def _manifest_has_local_sources(components: list[dict[str, Any]]) -> bool:
    for comp in components:
        files = comp.get("files")
        if not files or not isinstance(files, list):
            continue
        for file_spec in files:
            raw_spec, _ = coerce_file_spec(file_spec)
            if not _is_external_spec(raw_spec):
                return True
    return False


def _is_external_spec(raw_spec: str) -> bool:
    spec = os.path.expanduser(raw_spec)
    if is_http_url(spec):
        return True
    return parse_git_target(spec) is not None


def _parse_comment(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("File comment must be a string")
    return value


def _parse_range_value(value: Any) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("Range lines must be >= 1")
        return [(value, value)]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        ranges = [_parse_range_part(part) for part in parts]
        return ranges or None
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(v, int) for v in value):
            return [_parse_range_tuple(value[0], value[1])]
        ranges: list[tuple[int, int]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Range must be a pair of integers")
            ranges.append(_parse_range_tuple(item[0], item[1]))
        return ranges or None
    raise ValueError("Range must be a string or list")


def _parse_range_part(part: str) -> tuple[int, int]:
    match = _RANGE_RE.match(part)
    if match:
        return _parse_range_tuple(int(match.group(1)), int(match.group(2)))
    if part.isdigit():
        return _parse_range_tuple(int(part), int(part))
    raise ValueError(f"Invalid range value: {part}")


def _parse_range_tuple(start: int, end: int) -> tuple[int, int]:
    if start <= 0 or end <= 0:
        raise ValueError("Range lines must be >= 1")
    if start > end:
        raise ValueError("Range start must be <= end")
    return start, end


def _parse_symbols_value(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        symbols = [part.strip() for part in value.split(",") if part.strip()]
        return symbols or None
    if isinstance(value, (list, tuple)):
        symbols = [str(item).strip() for item in value if str(item).strip()]
        return symbols or None
    raise ValueError("Symbols must be a string or list")


def _merge_symbols(primary: list[str] | None, extra: list[str]) -> list[str] | None:
    merged = list(primary or [])
    for item in extra:
        if item not in merged:
            merged.append(item)
    return merged or None


def _resolve_spec_items(
    raw_spec: str,
    base_dir: str,
    *,
    component_name: str,
    filename_hint: Any | None,
) -> list[ResolvedItem]:
    spec = os.path.expanduser(raw_spec)

    if is_http_url(spec):
        opts = parse_target_spec(spec)
        url = opts.get("target", spec)
        filename = filename_hint or opts.get("filename")

        tgt = parse_git_url_target(url)
        if tgt:
            return _resolve_git_items(tgt, component_name)
        return [_resolve_http_item(url, filename)]

    tgt = parse_git_target(spec)
    if tgt:
        return _resolve_git_items(tgt, component_name)

    base = "" if os.path.isabs(spec) else base_dir
    paths = expand_git_paths(base, spec)
    resolved: list[ResolvedItem] = []
    for full in paths:
        if not os.path.exists(full):
            raise FileNotFoundError(
                f"Component '{component_name}' path not found: {full}"
            )
        refs = create_file_references([full], ignore_patterns=None, format="raw")[
            "refs"
        ]
        for ref in refs:
            rel_path = _relative_path(ref.path, base_dir)
            resolved.append(
                ResolvedItem(
                    source_type="local",
                    source_ref=base_dir,
                    source_rev=None,
                    source_path=rel_path,
                    context_subpath=rel_path,
                    content=ref.file_content,
                    manifest_spec=rel_path,
                )
            )
    return resolved


def _resolve_git_items(tgt, component_name: str) -> list[ResolvedItem]:
    repo_dir = ensure_repo(tgt)
    if tgt.path:
        paths = expand_git_paths(repo_dir, tgt.path)
    else:
        paths = [repo_dir]
    resolved: list[ResolvedItem] = []
    for full in paths:
        if not os.path.exists(full):
            raise FileNotFoundError(
                f"Component '{component_name}' path not found: {full}"
            )
        refs = create_file_references([full], ignore_patterns=None, format="raw")[
            "refs"
        ]
        for ref in refs:
            rel_path = _relative_path(ref.path, repo_dir)
            manifest_spec = _format_git_spec(tgt.repo_url, tgt.rev, rel_path)
            resolved.append(
                ResolvedItem(
                    source_type="git",
                    source_ref=tgt.repo_url,
                    source_rev=tgt.rev,
                    source_path=rel_path,
                    context_subpath=rel_path,
                    content=ref.file_content,
                    manifest_spec=manifest_spec,
                )
            )
    return resolved


def _resolve_http_item(url: str, filename_hint: Any | None) -> ResolvedItem:
    url_ref = URLReference(url, format="raw")
    origin, url_path = _split_url_path(url)
    context_path = _apply_filename_hint(url_path, filename_hint)
    return ResolvedItem(
        source_type="http",
        source_ref=origin,
        source_rev=None,
        source_path=url_path,
        context_subpath=context_path,
        content=url_ref.file_content,
        manifest_spec=url,
    )


def _split_url_path(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    raw_path = unquote(parsed.path or "")
    if not raw_path or raw_path == "/":
        path = "index"
    elif raw_path.endswith("/"):
        path = f"{raw_path}index"
    else:
        path = raw_path
    path = path.lstrip("/")
    return origin, path or "index"


def _apply_filename_hint(path: str, filename_hint: Any | None) -> str:
    if filename_hint is None:
        return path
    if not isinstance(filename_hint, str) or not filename_hint.strip():
        raise ValueError("filename must be a non-empty string")
    if "/" in filename_hint or "\\" in filename_hint:
        raise ValueError("filename must not contain path separators")
    parent = os.path.dirname(path)
    return f"{parent}/{filename_hint}" if parent else filename_hint


def _relative_path(path: str, root: str) -> str:
    path_obj = Path(path).resolve()
    root_obj = Path(root).resolve()
    try:
        rel = path_obj.relative_to(root_obj)
    except ValueError as exc:
        raise ValueError(f"Path outside root: {path}") from exc
    rel_str = rel.as_posix()
    if rel_str.startswith("../") or rel_str == "..":
        raise ValueError(f"Path outside root: {path}")
    return rel_str


def _format_git_spec(repo_url: str, rev: str | None, path: str) -> str:
    base = f"{repo_url}@{rev}" if rev else repo_url
    return f"{base}:{path}" if path else base


def _build_suffix(
    ranges: list[tuple[int, int]] | None, symbols: list[str] | None
) -> str | None:
    if symbols:
        safe_symbols = [_sanitize_symbol(sym) for sym in symbols]
        return "__S-" + "-".join(safe_symbols)
    if ranges:
        start, end = ranges[0]
        return f"__L{start}-L{end}"
    return None


def _sanitize_symbol(symbol: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", symbol.strip())
    return safe.strip("-") or "sym"


def _build_component_root(
    component_name: str,
    group_path: Any | None,
    base_name: Any | None,
    path_strategy: str,
    *,
    flatten_groups: set[tuple[str, ...]] | None = None,
) -> Path | None:
    if path_strategy != "by-component":
        return None
    if group_path:
        parts = [group_path] if isinstance(group_path, str) else list(group_path)
        group_key = tuple(parts)
        if flatten_groups and group_key in flatten_groups:
            return Path(*parts)
        name = base_name if isinstance(base_name, str) and base_name else component_name
        return Path(*parts, name)
    return Path(component_name)


def _resolve_context_path(
    component_name: str,
    item: ResolvedItem,
    suffix: str | None,
    used_paths: set[str],
    identity_paths: dict[tuple[Any, ...], Path],
    path_strategy: str,
    ranges: list[tuple[int, int]] | None,
    symbols: list[str] | None,
    *,
    component_root: Path | None = None,
    use_external_root: bool,
    strip_prefix: Path | None = None,
) -> tuple[Path, bool]:
    rel_path = _build_base_context_path(
        component_name,
        item,
        suffix,
        path_strategy,
        component_root,
        use_external_root,
        strip_prefix,
    )
    _ensure_relative(rel_path)

    if path_strategy == "on-disk":
        identity = _build_identity_key(item, ranges, symbols)
        existing = identity_paths.get(identity)
        if existing:
            return existing, False
        rel_path = _dedupe_path(rel_path, used_paths)
        identity_paths[identity] = rel_path
        return rel_path, True

    rel_path = _dedupe_path(rel_path, used_paths)
    return rel_path, True


def _strip_subpath_prefix(subpath: Path, prefix: Path | None) -> Path:
    if prefix is None:
        return subpath
    try:
        return subpath.relative_to(prefix)
    except ValueError:
        return subpath


def _build_base_context_path(
    component_name: str,
    item: ResolvedItem,
    suffix: str | None,
    path_strategy: str,
    component_root: Path | None = None,
    use_external_root: bool = True,
    strip_prefix: Path | None = None,
) -> Path:
    if item.source_type == "local":
        subpath = _split_subpath(item.context_subpath)
        if path_strategy == "by-component":
            subpath = _strip_subpath_prefix(subpath, strip_prefix)
        rel_path = (
            subpath
            if path_strategy == "on-disk"
            else (component_root or Path(component_name)) / subpath
        )
    else:
        if item.source_type == "http":
            ext_path = _build_http_external_path(item)
            if use_external_root:
                ext_path = Path("external") / ext_path
        else:
            subpath = _split_subpath(item.context_subpath)
            if path_strategy == "by-component":
                subpath = _strip_subpath_prefix(subpath, strip_prefix)
            ext_path = _build_external_path(item, subpath, use_external_root)
        rel_path = (
            ext_path
            if path_strategy == "on-disk"
            else (component_root or Path(component_name)) / ext_path
        )

    if suffix:
        rel_path = _apply_suffix(rel_path, suffix)
    return rel_path


def _build_identity_key(
    item: ResolvedItem,
    ranges: list[tuple[int, int]] | None,
    symbols: list[str] | None,
) -> tuple[Any, ...]:
    ranges_key = tuple(tuple(r) for r in ranges) if ranges else None
    symbols_key = tuple(symbols) if symbols else None
    return (
        item.source_type,
        item.source_ref,
        item.source_rev,
        item.source_path,
        ranges_key,
        symbols_key,
    )


def _split_subpath(subpath: str) -> Path:
    parts = [p for p in subpath.split("/") if p]
    return Path(*parts) if parts else Path("index")


def _apply_suffix(path: Path, suffix: str) -> Path:
    if path.suffix:
        name = f"{path.stem}{suffix}{path.suffix}"
    else:
        name = f"{path.name}{suffix}"
    return path.with_name(name)


def _dedupe_path(path: Path, used_paths: set[str]) -> Path:
    candidate = path
    counter = 2
    while candidate.as_posix() in used_paths:
        candidate = _append_disambiguator(path, counter)
        counter += 1
    used_paths.add(candidate.as_posix())
    return candidate


def _append_disambiguator(path: Path, counter: int) -> Path:
    suffix = f"__{counter}"
    if path.suffix:
        name = f"{path.stem}{suffix}{path.suffix}"
    else:
        name = f"{path.name}{suffix}"
    return path.with_name(name)


def _ensure_relative(path: Path) -> None:
    if path.is_absolute():
        raise ValueError(f"Invalid context path: {path}")
    if any(part in {"..", ""} for part in path.parts):
        raise ValueError(f"Invalid context path: {path}")


def _build_external_path(
    item: ResolvedItem, subpath: Path, use_external_root: bool
) -> Path:
    prefix = Path("external") if use_external_root else Path()
    if item.source_type == "git":
        return (
            prefix
            / _build_git_external_root(item.source_ref, item.source_rev)
            / subpath
        )
    return (
        prefix
        / _build_generic_external_root(item.source_type, item.source_ref)
        / subpath
    )


def _build_http_external_path(item: ResolvedItem) -> Path:
    host = _parse_http_host(item.source_ref)
    leaf = _pick_http_leaf(item.source_path)
    slug = f"{host}-{leaf}" if host else leaf
    slug = _sanitize_path_segment(slug, fallback="external")
    return Path(slug)


def _parse_http_host(source_ref: str) -> str:
    parsed = urlparse(source_ref)
    host = parsed.hostname or parsed.netloc or source_ref
    if parsed.port:
        host = f"{host}-{parsed.port}"
    return host


def _pick_http_leaf(source_path: str) -> str:
    path = Path(source_path)
    name = path.name
    if not name or name == "index":
        parent = path.parent.name
        if parent and parent != ".":
            return parent
        return "index"
    return name


def _build_git_external_root(source_ref: str, source_rev: str | None) -> Path:
    _, host, repo_path = _parse_git_source_ref(source_ref)
    repo_parts = [part for part in repo_path.split("/") if part]
    if repo_parts:
        slug = repo_parts[-1]
    else:
        slug = host or "repo"
    if source_rev:
        slug = f"{slug}@{_format_rev_for_path(source_rev)}"
    slug = _sanitize_path_segment(slug, fallback="repo")
    return Path(slug)


def _build_generic_external_root(source_type: str, source_ref: str) -> Path:
    safe_type = _sanitize_path_segment(source_type, fallback="external")
    safe_ref = _sanitize_path_segment(source_ref, fallback="source")
    return Path(f"{safe_type}-{safe_ref}")


def _parse_git_source_ref(source_ref: str) -> tuple[str, str, str]:
    if source_ref.startswith("git@"):
        host_path = source_ref[4:]
        host, _, path = host_path.partition(":")
        scheme = "ssh"
    else:
        parsed = urlparse(source_ref)
        scheme = (parsed.scheme or "https").lower()
        host = parsed.hostname or parsed.netloc or ""
        path = parsed.path.lstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if not host:
        host = source_ref
    return scheme, host, path


def _format_rev_for_path(value: str) -> str:
    rev = value.strip()
    if re.fullmatch(r"[0-9a-f]{7,40}", rev):
        return rev[:8]
    return rev


def _sanitize_path_segment(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _build_index_entry(
    rel_path: Path,
    item: ResolvedItem,
    ranges: list[tuple[int, int]] | None,
    symbols: list[str] | None,
    content: str,
) -> dict[str, Any]:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "context_path": rel_path.as_posix(),
        "source_type": item.source_type,
        "source_ref": item.source_ref,
        "source_path": item.source_path,
        "source_rev": item.source_rev,
        "range": ranges,
        "symbols": symbols,
        "hash": f"sha256:{digest}",
    }


def _build_manifest_file_entry(
    item: ResolvedItem,
    range_spec: list[tuple[int, int]] | None,
    symbols: list[str] | None,
    comment: str | None,
    extras: dict[str, Any],
) -> dict[str, Any] | str:
    key = "url" if item.source_type == "http" else "path"
    entry = {key: item.manifest_spec}
    entry.update({k: v for k, v in extras.items() if k not in {"range", "symbols"}})
    if range_spec:
        entry["range"] = _format_ranges(range_spec)
    if symbols:
        entry["symbols"] = symbols
    if comment is not None:
        entry["comment"] = comment
    if len(entry) == 1 and key == "path":
        return item.manifest_spec
    return entry


def _format_ranges(ranges: list[tuple[int, int]]) -> str | list[str]:
    formatted = [f"{start}-{end}" for start, end in ranges]
    if len(formatted) == 1:
        return formatted[0]
    return formatted


def _build_normalized_config(
    cfg: dict[str, Any], context_cfg: dict[str, Any], base_dir: str
) -> dict[str, Any]:
    normalized = dict(cfg)
    normalized["root"] = base_dir
    context = dict(cfg.get("context") or {})
    context.pop("dir", None)
    context.pop("access", None)
    if "path-strategy" in context:
        context["path-strategy"] = context_cfg["path_strategy"]
    if "include-meta" in context:
        context["include-meta"] = context_cfg["include_meta"]
    if context_cfg["agents_text"] is not None:
        context["agents"] = {
            "files": context_cfg["agents_files"] or ["AGENTS.md"],
            "text": context_cfg["agents_text"],
        }
    else:
        context.pop("agents", None)
    if context:
        normalized["context"] = context
    else:
        normalized.pop("context", None)
    return normalized


def _dump_manifest(data: dict[str, Any]) -> str:
    import yaml

    class _LiteralDumper(yaml.SafeDumper):
        pass

    def _repr_str(dumper, value):
        if "\n" in value:
            return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", value)

    _LiteralDumper.add_representer(str, _repr_str)
    return yaml.dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        Dumper=_LiteralDumper,
    )


def _dump_index(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, indent=2, sort_keys=True)


def _write_files(files: list[tuple[Path, str]]) -> None:
    for path, content in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _apply_read_only(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            _chmod(Path(dirpath) / name, 0o444)
        for name in dirnames:
            _chmod(Path(dirpath) / name, 0o555)
    _chmod(root, 0o555)


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        return


def _collect_parent_dirs(rel_path: str, target: set[str]) -> None:
    path = Path(rel_path)
    parent = path.parent
    while parent and parent.as_posix() != ".":
        target.add(parent.as_posix())
        parent = parent.parent


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _all_read_only(
    context_dir: Path, expected_dirs: set[str], expected_files: list[str] | set[str]
) -> bool:
    for rel in expected_files:
        path = context_dir / rel
        if not _is_read_only(path):
            return False
    for rel in expected_dirs:
        path = context_dir / rel
        if not _is_read_only(path):
            return False
    return True


def _is_read_only(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    return (
        (mode & stat.S_IWUSR) == 0
        and (mode & stat.S_IWGRP) == 0
        and (mode & stat.S_IWOTH) == 0
    )


def _clear_context_dir(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        raise ValueError(f"Context dir exists and is not a directory: {path}")
    if not path.exists():
        return
    _make_writable(path)
    for item in path.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


def find_untracked_files(path: Path) -> list[str]:
    if not path.exists() or not path.is_dir():
        return []

    index_path = path / "index.json"
    tracked_paths: set[str] = set()

    if index_path.exists():
        try:
            import json

            with open(index_path, "r", encoding="utf-8") as fh:
                index_data = json.load(fh)
            components = index_data.get("components", {})
            for entries in components.values():
                for entry in entries:
                    ctx_path = entry.get("context_path")
                    if ctx_path:
                        tracked_paths.add(ctx_path)
            tracked_paths.add("manifest.yaml")
            tracked_paths.add("index.json")
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass

    untracked: list[str] = []
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                rel_path = file_path.relative_to(path).as_posix()
            except ValueError:
                continue
            if rel_path not in tracked_paths:
                untracked.append(rel_path)

    return untracked


def _make_writable(path: Path) -> None:
    for root, dirs, files in os.walk(path, topdown=False):
        root_path = Path(root)
        for name in files:
            target = root_path / name
            if target.is_symlink():
                continue
            _chmod(target, 0o666)
        for name in dirs:
            target = root_path / name
            if target.is_symlink():
                continue
            _chmod(target, 0o777)
    if not path.is_symlink():
        _chmod(path, 0o777)
