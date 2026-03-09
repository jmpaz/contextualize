"""Internal payload building logic."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

from ..concurrency import run_indexed_tasks_fail_fast
from ..git.cache import ensure_repo, expand_git_paths
from ..git.target import parse_git_target
from ..runtime import get_payload_spec_jobs
from ..render.links import add_markdown_link_refs
from .manifest import coerce_file_spec, component_selectors
from ..references import create_file_references
from ..references.helpers import is_http_url, parse_git_url_target, parse_target_spec
from ..utils import wrap_text


class _SimpleReference:
    def __init__(
        self,
        output: str,
        *,
        path: str | None = None,
        trace_path: str | None = None,
        content: str | None = None,
    ):
        self.output = output
        self.path = path or ""
        if trace_path:
            self.trace_path = trace_path
        if content is not None:
            self.file_content = content
            self.original_file_content = content


class _MapReference:
    def __init__(self, path: str, output: str, content: str):
        self.path = path
        self.output = output
        self.file_content = content
        self.original_file_content = content
        self.is_map = True


_DEFAULT_MAP_TOKENS = 10000
_EXTERNAL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


@dataclass
class _SpecResolution:
    refs: list[Any]
    input_refs: list[Any]
    trace_items: list[Any]
    skipped_paths: set[str]
    skip_impact: dict[str, Any]


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


def _resolve_spec_to_paths(
    raw_spec: str,
    base_dir: str,
    *,
    component_name: str,
) -> list[str]:
    spec = os.path.expanduser(raw_spec)

    if is_http_url(spec):
        opts = parse_target_spec(spec)
        url = opts.get("target", spec)
        tgt = parse_git_url_target(url)
        if not tgt:
            return []
        repo_dir = ensure_repo(tgt)
        paths = [repo_dir] if not tgt.path else expand_git_paths(repo_dir, tgt.path)
    else:
        if _EXTERNAL_SCHEME_RE.match(spec) and parse_git_target(spec) is None:
            return []
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
    from ..render.map import generate_repo_map_data

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


def _build_map_reference(
    label: str,
    output: str,
    *,
    label_suffix: str | None,
    token_target: str,
) -> _MapReference:
    from ..render.text import process_text

    rendered = process_text(
        output,
        format="md",
        label=label,
        label_suffix=label_suffix,
        token_target=token_target,
        include_token_count=False,
    )
    return _MapReference(label, rendered, output)


def _resolve_spec_to_seed_refs(
    raw_spec: Any,
    file_opts: Dict[str, Any],
    base_dir: str,
    *,
    inject: bool,
    depth: int,
    component_name: str,
    label_suffix: str | None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> tuple[List[Any], List[Any], List[tuple[str, str, int]]]:
    spec = os.path.expanduser(raw_spec)
    opts = parse_target_spec(spec)
    target = opts.get("target", spec)
    filename = file_opts.get("filename") or opts.get("filename")
    wrap = file_opts.get("wrap") or opts.get("wrap")
    seed_refs: List[Any] = []
    trace_inputs: List[Any] = []

    def _append_wrapped_ref(ref: Any, default_path: str) -> None:
        if filename:
            effective_label = filename
        elif hasattr(ref, "get_label") and callable(getattr(ref, "get_label")):
            effective_label = ref.get_label()  # type: ignore[call-arg]
        else:
            effective_label = getattr(ref, "trace_path", None) or getattr(
                ref, "path", default_path
            )
        if label_suffix:
            effective_label = f"{effective_label} {label_suffix}"
        wrapped = wrap_text(ref.output, wrap or "md", effective_label)
        content = getattr(ref, "file_content", None)
        if not isinstance(content, str):
            content = ""
            if hasattr(ref, "read") and callable(getattr(ref, "read")):
                try:
                    read_value = ref.read()  # type: ignore[call-arg]
                except Exception:
                    read_value = ""
                if isinstance(read_value, str):
                    content = read_value
        simple_ref = _SimpleReference(
            wrapped,
            path=getattr(ref, "path", default_path),
            trace_path=getattr(ref, "trace_path", None),
            content=content,
        )
        seed_refs.append(simple_ref)
        trace_inputs.append(simple_ref)

    if is_http_url(spec):
        url = target

        tgt = parse_git_url_target(url)
        if tgt:
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
                    label_suffix=label_suffix,
                    inject=inject,
                    depth=depth,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                )["refs"]
                seed_refs.extend(refs)
                trace_inputs.extend([ref for ref in refs if hasattr(ref, "path")])
            return seed_refs, trace_inputs, []

        refs = create_file_references(
            [url],
            ignore_patterns=None,
            format="raw",
            label="relative",
            inject=inject,
            depth=depth,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )["refs"]
        for ref in refs:
            _append_wrapped_ref(ref, url)
        return seed_refs, trace_inputs, []

    if _EXTERNAL_SCHEME_RE.match(target) and parse_git_target(target) is None:
        refs = create_file_references(
            [target],
            ignore_patterns=None,
            format="raw",
            label="relative",
            inject=inject,
            depth=depth,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )["refs"]
        for ref in refs:
            _append_wrapped_ref(ref, target)
        return seed_refs, trace_inputs, []

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
            from ..references import FileReference

            fr = FileReference(
                full,
                format="md",
                label=str(custom_label),
                label_suffix=label_suffix,
                inject=inject,
                depth=depth,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
            )
            seed_refs.append(fr)
        else:
            refs = create_file_references(
                [full],
                ignore_patterns=None,
                format="md",
                label="relative",
                label_suffix=label_suffix,
                inject=inject,
                depth=depth,
            )["refs"]
            seed_refs.extend(refs)

    trace_inputs.extend([r for r in seed_refs if hasattr(r, "path")])
    return seed_refs, trace_inputs, []


def _render_attachment_block(
    name: str,
    refs: List[Any],
    wrap_mode: Optional[str],
    prefix: str,
    suffix: str,
    comment: str | None,
) -> str:
    """Render an <attachment> block with optional wrap/prefix/suffix."""
    comment_attr = f" {comment}" if comment else ""
    attachment_lines = [f'<attachment label="{name}"{comment_attr}>']
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


def _append_refs(target: list[Any], refs: list[Any]) -> None:
    target.extend(refs)


def _resolve_spec(
    *,
    raw_spec: str,
    file_opts: Dict[str, Any],
    base_dir: str,
    inject: bool,
    depth: int,
    component_name: str,
    item_comment: str | None,
    map_component: bool,
    token_target: str,
    comp_link_depth: int,
    per_file_link_depth: int | None,
    per_file_link_scope: str,
    resolved_link_skip: list[str],
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> _SpecResolution:
    if map_component:
        map_paths = _resolve_spec_to_paths(
            raw_spec,
            base_dir,
            component_name=component_name,
        )
        if map_paths:
            map_output = _generate_repo_map_output(map_paths, token_target=token_target)
            map_ref = _build_map_reference(
                raw_spec,
                map_output,
                label_suffix=item_comment,
                token_target=token_target,
            )
            return _SpecResolution(
                refs=[map_ref],
                input_refs=[map_ref],
                trace_items=[],
                skipped_paths=set(),
                skip_impact={},
            )

    seed_refs, spec_trace_inputs, arena_channel_trace_items = (
        _resolve_spec_to_seed_refs(
            raw_spec,
            file_opts,
            base_dir,
            inject=inject,
            depth=depth,
            component_name=component_name,
            label_suffix=item_comment,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
    )

    trace_items: list[Any] = list(arena_channel_trace_items)
    effective_link_depth = int(
        per_file_link_depth if per_file_link_depth is not None else comp_link_depth
    )
    if effective_link_depth <= 0:
        return _SpecResolution(
            refs=seed_refs,
            input_refs=spec_trace_inputs,
            trace_items=trace_items,
            skipped_paths=set(),
            skip_impact={},
        )

    expanded_refs, comp_trace_items, comp_skip_impact = add_markdown_link_refs(
        seed_refs,
        link_depth=effective_link_depth,
        scope=per_file_link_scope,
        format_="md",
        label="relative",
        inject=inject,
        link_skip=resolved_link_skip if resolved_link_skip else None,
    )
    trace_items.extend(comp_trace_items)
    skipped_paths: set[str] = set()
    if resolved_link_skip:
        for skip_path in resolved_link_skip:
            abs_skip_path = os.path.abspath(skip_path)
            if os.path.exists(abs_skip_path):
                skipped_paths.add(abs_skip_path)
    return _SpecResolution(
        refs=expanded_refs,
        input_refs=spec_trace_inputs,
        trace_items=trace_items,
        skipped_paths=skipped_paths,
        skip_impact=comp_skip_impact or {},
    )


def _should_map_component(
    selectors: set[str],
    *,
    map_mode: bool,
    map_keys: set[str],
) -> bool:
    if not selectors:
        return False
    if map_mode:
        if map_keys:
            return bool(selectors & map_keys)
        return True
    if map_keys:
        return bool(selectors & map_keys)
    return False


def build_payload_impl(
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
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
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
    spec_jobs = get_payload_spec_jobs()

    for comp in components:
        selectors = component_selectors(comp)
        if selectors and exclude_set and selectors & exclude_set:
            continue
        name = comp.get("name")
        wrap_mode = comp.get("wrap")
        component_comment = _format_comment(comp.get("comment"))

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
            selectors, map_mode=map_mode, map_keys=map_key_set
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
        spec_tasks: list[tuple[int, Any]] = []

        for spec_index, spec in enumerate(files, 1):
            spec, file_opts = coerce_file_spec(spec)
            raw_spec = spec
            item_comment = _format_comment(file_opts.get("comment"))

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

            spec_tasks.append(
                (
                    spec_index,
                    lambda rs=raw_spec, fo=file_opts, ic=item_comment, pld=per_file_link_depth, pls=per_file_link_scope, rls=resolved_link_skip: (
                        _resolve_spec(
                            raw_spec=rs,
                            file_opts=fo,
                            base_dir=base_dir,
                            inject=inject,
                            depth=depth,
                            component_name=name,
                            item_comment=ic,
                            map_component=map_component,
                            token_target=token_target,
                            comp_link_depth=comp_link_depth,
                            per_file_link_depth=pld,
                            per_file_link_scope=pls,
                            resolved_link_skip=rls,
                            use_cache=use_cache,
                            cache_ttl=cache_ttl,
                            refresh_cache=refresh_cache,
                        )
                    ),
                )
            )

        for _, spec_result in run_indexed_tasks_fail_fast(
            spec_tasks,
            max_workers=spec_jobs,
        ):
            _append_refs(refs_for_attachment, spec_result.refs)
            input_refs_for_comp.extend(spec_result.input_refs)
            all_trace_items.extend(spec_result.trace_items)
            all_skipped_paths.update(spec_result.skipped_paths)
            if spec_result.skip_impact:
                all_skip_impact.update(spec_result.skip_impact)

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
