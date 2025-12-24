import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..git.cache import ensure_repo, expand_git_paths, parse_git_target
from .links import add_markdown_link_refs
from .references import URLReference, create_file_references
from .utils import wrap_text


def _parse_url_spec(spec: str) -> dict[str, Any]:
    """Parse URL spec like 'https://example.com::filename=file.py::wrap=md' into an options dict."""
    parts = spec.split("::")
    opts: dict[str, str | None] = {}
    target_parts: list[str] = []

    for part in parts:
        m = re.fullmatch(r'(filename|params|root|wrap)=(?:"([^"]*)"|([^"]*))', part)
        if m:
            opts[m.group(1)] = m.group(2) or m.group(3)
        else:
            target_parts.append(part)

    opts["target"] = "::".join(target_parts)
    return opts


class _SimpleReference:
    def __init__(self, output: str):
        self.output = output


@dataclass
class _PathReference:
    path: str


_DEFAULT_MAP_TOKENS = 10000


def _coerce_file_spec(spec: Any) -> Tuple[str, Dict[str, Any]]:
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


def _format_comment(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Comment must be a string, got: {type(value)}")
    text = value.strip()
    if not text:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    import json

    escaped = json.dumps(text, ensure_ascii=False)
    return f"comment={escaped}"


def _combine_comment(comment: str | None, output: str) -> str:
    if comment:
        return f"{comment}\n{output}"
    return output


def _wrapped_url_reference(
    url: str,
    *,
    filename: Optional[str],
    wrap: Optional[str],
    inject: bool,
    depth: int,
) -> _SimpleReference:
    url_ref = URLReference(
        url,
        format="raw",
        label=filename or url,
        inject=inject,
        depth=depth,
    )
    wrapped = wrap_text(url_ref.output, wrap or "md", filename or url)
    return _SimpleReference(wrapped)


def _resolve_spec_to_paths(
    raw_spec: str,
    base_dir: str,
    *,
    component_name: str,
) -> list[str]:
    spec = os.path.expanduser(raw_spec)

    if spec.startswith("http://") or spec.startswith("https://"):
        opts = _parse_url_spec(spec)
        url = opts.get("target", spec)
        tgt = parse_git_target(url)
        if not tgt or (
            tgt.path is None
            and not tgt.repo_url.endswith(".git")
            and tgt.repo_url == url
        ):
            return []
        repo_dir = ensure_repo(tgt)
        paths = [repo_dir] if not tgt.path else expand_git_paths(repo_dir, tgt.path)
    else:
        tgt = parse_git_target(spec)
        if tgt:
            repo_dir = ensure_repo(tgt)
            paths = [repo_dir] if not tgt.path else expand_git_paths(repo_dir, tgt.path)
        else:
            base = "" if os.path.isabs(spec) else base_dir
            paths = expand_git_paths(base, spec)

    resolved = []
    for full in paths:
        if not os.path.exists(full):
            raise FileNotFoundError(
                f"Component '{component_name}' path not found: {full}"
            )
        resolved.append(full)
    return resolved


def _compute_map_root(paths: list[str]) -> str | None:
    if not paths:
        return None
    if len(paths) == 1:
        path = paths[0]
        if os.path.isdir(path):
            return path
        return os.path.dirname(path)
    try:
        return os.path.commonpath(paths)
    except ValueError:
        return None


def _generate_repo_map_output(
    paths: list[str],
    *,
    token_target: str,
) -> str:
    from .repomap import generate_repo_map_data

    root = _compute_map_root(paths)
    result = generate_repo_map_data(
        paths,
        _DEFAULT_MAP_TOKENS,
        "raw",
        ignore=None,
        annotate_tokens=False,
        token_target=token_target,
        root=root,
    )
    if "error" in result:
        return result["error"]
    return result["repo_map"]


def _resolve_spec_to_seed_refs(
    raw_spec: Any,
    file_opts: Dict[str, Any],
    base_dir: str,
    *,
    inject: bool,
    depth: int,
    component_name: str,
) -> List[Any]:
    """Resolve a single file/url/git spec into a list of seed refs."""
    spec = os.path.expanduser(raw_spec)
    seed_refs: List[Any] = []

    if spec.startswith("http://") or spec.startswith("https://"):
        opts = _parse_url_spec(spec)
        url = opts.get("target", spec)
        filename = file_opts.get("filename") or opts.get("filename")
        wrap = file_opts.get("wrap") or opts.get("wrap")

        tgt = parse_git_target(url)
        if tgt and (
            tgt.path is not None or tgt.repo_url.endswith(".git") or tgt.repo_url != url
        ):
            repo_dir = ensure_repo(tgt)
            paths = [repo_dir] if not tgt.path else expand_git_paths(repo_dir, tgt.path)
            for full in paths:
                if not os.path.exists(full):
                    raise FileNotFoundError(
                        f"Component '{component_name}' path not found: {full}"
                    )
                refs = create_file_references(
                    [full],
                    ignore_patterns=None,
                    format="md",
                    label="relative",
                    inject=inject,
                    depth=depth,
                )["refs"]
                seed_refs.extend(refs)
        else:
            seed_refs.append(
                _wrapped_url_reference(
                    url,
                    filename=filename,
                    wrap=wrap,
                    inject=inject,
                    depth=depth,
                )
            )
        return seed_refs

    # git or local
    tgt = parse_git_target(spec)
    if tgt:
        repo_dir = ensure_repo(tgt)
        paths = [repo_dir] if not tgt.path else expand_git_paths(repo_dir, tgt.path)
    else:
        base = "" if os.path.isabs(spec) else base_dir
        paths = expand_git_paths(base, spec)

    for full in paths:
        if not os.path.exists(full):
            raise FileNotFoundError(
                f"Component '{component_name}' path not found: {full}"
            )
        custom_label = file_opts.get("filename")
        if custom_label and os.path.isfile(full):
            from .references import FileReference

            fr = FileReference(
                full,
                format="md",
                label=str(custom_label),
                inject=inject,
                depth=depth,
            )
            seed_refs.append(fr)
        else:
            refs = create_file_references(
                [full],
                ignore_patterns=None,
                format="md",
                label="relative",
                inject=inject,
                depth=depth,
            )["refs"]
            seed_refs.extend(refs)

    return seed_refs


def _render_attachment_block(
    name: str,
    refs: List[Any],
    wrap_mode: Optional[str],
    prefix: str,
    suffix: str,
    comment: str | None,
) -> str:
    """Render an <attachment> block with optional wrap/prefix/suffix."""
    attachment_lines = [f'<attachment label="{name}">']
    if comment:
        attachment_lines.append(comment)
        if refs:
            attachment_lines.append("")
    for idx, ref in enumerate(refs):
        attachment_lines.append(ref.output)
        if idx < len(refs) - 1:
            attachment_lines.append("")
    attachment_lines.append("</attachment>")
    inner = "\n".join(attachment_lines)

    if wrap_mode:
        if wrap_mode.lower() == "md":
            inner = "```\n" + inner + "\n```"
        else:
            inner = f"<{wrap_mode}>\n{inner}\n</{wrap_mode}>"

    block_lines: List[str] = []
    if prefix:
        block_lines.append(prefix)
    block_lines.append(inner)
    if suffix:
        block_lines.append(suffix)
    return "\n".join(block_lines)


def assemble_payload(
    components: List[Dict[str, Any]],
    base_dir: str,
    *,
    inject: bool = False,
    depth: int = 5,
) -> str:
    return build_payload(
        components,
        base_dir,
        inject=inject,
        depth=depth,
        link_depth=0,
        link_scope="all",
        link_skip=None,
    ).payload


def _append_refs_with_comment(
    target: list[Any],
    refs: list[Any],
    comment: str | None,
) -> None:
    if comment:
        if refs:
            combined = _combine_comment(comment, refs[0].output)
            target.append(_SimpleReference(combined))
            target.extend(refs[1:])
        else:
            target.append(_SimpleReference(comment))
        return
    target.extend(refs)


def _should_map_component(
    name: str | None,
    *,
    map_mode: bool,
    map_keys: set[str],
) -> bool:
    if not name:
        return False
    if map_mode:
        if map_keys:
            return name in map_keys
        return True
    if map_keys:
        return name in map_keys
    return False


def _build_payload_impl(
    components: List[Dict[str, Any]],
    base_dir: str,
    *,
    inject: bool = False,
    depth: int = 5,
    link_depth_default: int = 0,
    link_scope_default: str = "all",
    link_skip_default: List[str] = None,
    exclude_keys: list[str] | None = None,
    map_mode: bool = False,
    map_keys: list[str] | None = None,
    token_target: str = "cl100k_base",
):
    parts: List[str] = []
    all_input_refs = []
    all_trace_items = []
    all_skipped_paths = set()
    all_skip_impact = {}
    exclude_set = {k.strip() for k in (exclude_keys or []) if k and k.strip()}
    map_key_set = {k.strip() for k in (map_keys or []) if k and k.strip()}
    overlap = sorted(exclude_set & map_key_set)
    if overlap:
        names = ", ".join(overlap)
        raise ValueError(f"Components cannot be both mapped and excluded: {names}")

    for comp in components:
        name = comp.get("name")
        if name and name in exclude_set:
            continue
        wrap_mode = comp.get("wrap")
        component_comment = _format_comment(comp.get("comment"))

        # text-only component passthrough
        if "text" in comp:
            text = comp["text"].rstrip()
            if component_comment:
                text = (
                    _combine_comment(component_comment, text)
                    if text
                    else component_comment
                )
            if wrap_mode:
                if wrap_mode.lower() == "md":
                    text = "```\n" + text + "\n```"
                else:
                    text = f"<{wrap_mode}>\n{text}\n</{wrap_mode}>"
            parts.append(text)
            continue

        files = comp.get("files")
        if not name or not files:
            raise ValueError(
                f"Component must have either 'text' or both 'name' & 'files': {comp}"
            )

        prefix = comp.get("prefix", "").rstrip()
        suffix = comp.get("suffix", "").lstrip()
        map_component = _should_map_component(
            name, map_mode=map_mode, map_keys=map_key_set
        )

        comp_link_depth = int(comp.get("link-depth", link_depth_default) or 0)
        comp_link_scope = (comp.get("link-scope", link_scope_default) or "all").lower()

        comp_link_skip = comp.get("link-skip", link_skip_default)
        if comp_link_skip is None:
            comp_link_skip = []
        elif isinstance(comp_link_skip, str):
            comp_link_skip = [comp_link_skip]

        resolved_link_skip_default: List[str] = []
        for skip_path in comp_link_skip:
            skip_path = os.path.expanduser(skip_path)
            if not os.path.isabs(skip_path):
                skip_path = os.path.join(base_dir, skip_path)
            resolved_link_skip_default.append(skip_path)

        refs_for_attachment = []
        input_refs_for_comp = []
        seen_input_paths: set[str] = set()

        for spec in files:
            spec, file_opts = _coerce_file_spec(spec)
            spec = os.path.expanduser(spec)
            item_comment = _format_comment(file_opts.get("comment"))

            if map_component:
                map_paths = _resolve_spec_to_paths(
                    spec,
                    base_dir,
                    component_name=name,
                )
                if map_paths:
                    map_output = _generate_repo_map_output(
                        map_paths, token_target=token_target
                    )
                    _append_refs_with_comment(
                        refs_for_attachment,
                        [_SimpleReference(map_output)],
                        item_comment,
                    )
                    for path in map_paths:
                        abs_path = os.path.abspath(path)
                        if abs_path in seen_input_paths:
                            continue
                        seen_input_paths.add(abs_path)
                        input_refs_for_comp.append(_PathReference(abs_path))
                elif item_comment:
                    _append_refs_with_comment(
                        refs_for_attachment,
                        [],
                        item_comment,
                    )
                continue

            per_file_link_depth = file_opts.get("link-depth")
            per_file_link_scope = (
                (file_opts.get("link-scope") or comp_link_scope).lower()
                if file_opts
                else comp_link_scope
            )
            per_file_link_skip = file_opts.get("link-skip") if file_opts else None
            resolved_link_skip = list(resolved_link_skip_default)
            if per_file_link_skip:
                if isinstance(per_file_link_skip, str):
                    per_file_link_skip = [per_file_link_skip]
                for skip_path in per_file_link_skip:
                    skip_path = os.path.expanduser(skip_path)
                    if not os.path.isabs(skip_path):
                        skip_path = os.path.join(base_dir, skip_path)
                    resolved_link_skip.append(skip_path)

            seed_refs = _resolve_spec_to_seed_refs(
                spec,
                file_opts,
                base_dir,
                inject=inject,
                depth=depth,
                component_name=name,
            )

            input_refs_for_comp.extend([r for r in seed_refs if hasattr(r, "path")])

            effective_link_depth = int(
                per_file_link_depth
                if per_file_link_depth is not None
                else comp_link_depth
            )
            if effective_link_depth > 0:
                expanded_refs, comp_trace_items, comp_skip_impact = (
                    add_markdown_link_refs(
                        seed_refs,
                        link_depth=effective_link_depth,
                        scope=per_file_link_scope,
                        format_="md",
                        label="relative",
                        inject=inject,
                        link_skip=resolved_link_skip if resolved_link_skip else None,
                    )
                )
                _append_refs_with_comment(
                    refs_for_attachment, expanded_refs, item_comment
                )
                all_trace_items.extend(comp_trace_items)
                if resolved_link_skip:
                    for skip_path in resolved_link_skip:
                        abs_skip_path = os.path.abspath(skip_path)
                        if os.path.exists(abs_skip_path):
                            all_skipped_paths.add(abs_skip_path)
                if comp_skip_impact:
                    all_skip_impact.update(comp_skip_impact)
            else:
                _append_refs_with_comment(refs_for_attachment, seed_refs, item_comment)

        all_input_refs.extend(input_refs_for_comp)

        if refs_for_attachment or prefix or suffix or component_comment:
            parts.append(
                _render_attachment_block(
                    name,
                    refs_for_attachment,
                    wrap_mode,
                    prefix,
                    suffix,
                    component_comment,
                )
            )

    return (
        "\n\n".join(parts),
        all_input_refs,
        all_trace_items,
        base_dir,
        list(all_skipped_paths),
        all_skip_impact,
    )


@dataclass
class PayloadResult:
    payload: str
    input_refs: List[Any]
    trace_items: List[Any]
    base_dir: str
    skipped_paths: List[str]
    skip_impact: Dict[str, Any]


def build_payload(
    components: List[Dict[str, Any]],
    base_dir: str,
    *,
    inject: bool = False,
    depth: int = 5,
    link_depth: int = 0,
    link_scope: str = "all",
    link_skip: Optional[List[str]] = None,
    exclude_keys: list[str] | None = None,
    map_mode: bool = False,
    map_keys: list[str] | None = None,
    token_target: str = "cl100k_base",
) -> PayloadResult:
    payload, input_refs, trace_items, base, skipped, impact = _build_payload_impl(
        components,
        base_dir,
        inject=inject,
        depth=depth,
        link_depth_default=link_depth,
        link_scope_default=link_scope,
        link_skip_default=link_skip or [],
        exclude_keys=exclude_keys,
        map_mode=map_mode,
        map_keys=map_keys,
        token_target=token_target,
    )
    return PayloadResult(payload, input_refs, trace_items, base, skipped, impact)


def render_from_yaml(
    manifest_path: str,
    *,
    inject: bool = False,
    depth: int = 5,
) -> str:
    """Render manifest to payload text."""
    return render_manifest(manifest_path, inject=inject, depth=depth).payload


def render_manifest(
    manifest_path: str,
    *,
    inject: bool = False,
    depth: int = 5,
    exclude_keys: list[str] | None = None,
    map_mode: bool = False,
    map_keys: list[str] | None = None,
    token_target: str = "cl100k_base",
) -> PayloadResult:
    """
    Load YAML and assemble payload with mdlinks.
    Respects top-level config keys:
      - root: base directory for relative paths
      - link-depth: default depth for Markdown link traversal
      - link-scope: "first" or "all" (default: all)
      - link-skip: list of paths to skip when resolving Markdown links
    Returns (payload_text, input_refs, trace_items, base_dir, skipped_paths, skip_impact).
    """
    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping with 'config' and 'components'")

    cfg = data.get("config", {})
    if "root" in cfg:
        raw = cfg.get("root") or "~"
        base_dir = os.path.expanduser(raw)
    else:
        base_dir = os.path.dirname(os.path.abspath(manifest_path))

    comps = data.get("components")
    if not isinstance(comps, list):
        raise ValueError("'components' must be a list")

    link_depth_default = int(cfg.get("link-depth", 0) or 0)
    link_scope_default = (cfg.get("link-scope", "all") or "all").lower()
    link_skip_default = cfg.get("link-skip", [])
    if isinstance(link_skip_default, str):
        link_skip_default = [link_skip_default]

    return build_payload(
        comps,
        base_dir,
        inject=inject,
        depth=depth,
        link_depth=link_depth_default,
        link_scope=link_scope_default,
        link_skip=link_skip_default,
        exclude_keys=exclude_keys,
        map_mode=map_mode,
        map_keys=map_keys,
        token_target=token_target,
    )


def render_manifest_data(
    data: Dict[str, Any],
    manifest_cwd: str,
    *,
    inject: bool = False,
    depth: int = 5,
    exclude_keys: list[str] | None = None,
    map_mode: bool = False,
    map_keys: list[str] | None = None,
    token_target: str = "cl100k_base",
) -> PayloadResult:
    """
    Assemble from an already-parsed YAML mapping (used for stdin case).
    Returns (payload_text, input_refs, trace_items, base_dir, skipped_paths, skip_impact).
    """
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping with 'config' and 'components'")

    cfg = data.get("config", {})
    if "root" in cfg:
        base_dir = os.path.expanduser(cfg.get("root") or "~")
    else:
        base_dir = manifest_cwd

    comps = data.get("components")
    if not isinstance(comps, list):
        raise ValueError("'components' must be a list")

    link_depth_default = int(cfg.get("link-depth", 0) or 0)
    link_scope_default = (cfg.get("link-scope", "all") or "all").lower()
    link_skip_default = cfg.get("link-skip", [])
    if isinstance(link_skip_default, str):
        link_skip_default = [link_skip_default]

    return build_payload(
        comps,
        base_dir,
        inject=inject,
        depth=depth,
        link_depth=link_depth_default,
        link_scope=link_scope_default,
        link_skip=link_skip_default,
        exclude_keys=exclude_keys,
        map_mode=map_mode,
        map_keys=map_keys,
        token_target=token_target,
    )
