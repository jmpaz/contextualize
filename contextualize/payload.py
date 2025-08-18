import os
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .gitcache import ensure_repo, expand_git_paths, parse_git_target
from .mdlinks import add_markdown_link_refs
from .reference import URLReference, create_file_references
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


def assemble_payload(
    components: List[Dict[str, Any]],
    base_dir: str,
    *,
    inject: bool = False,
    depth: int = 5,
) -> str:
    """Assemble an attachments payload from components."""
    parts: List[str] = []

    for comp in components:
        wrap_mode = comp.get("wrap")

        # 1) text‐only
        if "text" in comp:
            text = comp["text"].rstrip()
            if wrap_mode:
                if wrap_mode.lower() == "md":
                    text = "```\n" + text + "\n```"
                else:
                    text = f"<{wrap_mode}>\n{text}\n</{wrap_mode}>"
            parts.append(text)
            continue

        name = comp.get("name")
        files = comp.get("files")
        if not name or not files:
            raise ValueError(
                f"Component must have either 'text' or both 'name' & 'files': {comp}"
            )

        prefix = comp.get("prefix", "").rstrip()
        suffix = comp.get("suffix", "").lstrip()

        all_refs = []
        input_file_refs = []
        for spec in files:
            spec, file_opts = _coerce_file_spec(spec)
            spec = os.path.expanduser(spec)

            if spec.startswith("http://") or spec.startswith("https://"):
                opts = _parse_url_spec(spec)
                url = opts.get("target", spec)
                filename = file_opts.get("filename") or opts.get("filename")
                wrap = file_opts.get("wrap") or opts.get("wrap")

                tgt = parse_git_target(url)
                if tgt and (
                    tgt.path is not None
                    or tgt.repo_url.endswith(".git")
                    or tgt.repo_url != url
                ):
                    repo_dir = ensure_repo(tgt)
                    paths = (
                        [repo_dir]
                        if not tgt.path
                        else expand_git_paths(repo_dir, tgt.path)
                    )
                    for full in paths:
                        if not os.path.exists(full):
                            raise FileNotFoundError(
                                f"Component '{name}' path not found: {full}"
                            )
                        refs = create_file_references(
                            [full],
                            ignore_paths=None,
                            format="md",
                            label="relative",
                            inject=inject,
                            depth=depth,
                        )["refs"]
                        all_refs.extend(refs)
                        input_file_refs.extend(refs)
                else:
                    all_refs.append(
                        _wrapped_url_reference(
                            url,
                            filename=filename,
                            wrap=wrap,
                            inject=inject,
                            depth=depth,
                        )
                    )
            else:
                tgt = parse_git_target(spec)
                if tgt:
                    repo_dir = ensure_repo(tgt)
                    paths = (
                        [repo_dir]
                        if not tgt.path
                        else expand_git_paths(repo_dir, tgt.path)
                    )
                else:
                    base = "" if os.path.isabs(spec) else base_dir
                    paths = expand_git_paths(base, spec)

                for full in paths:
                    if not os.path.exists(full):
                        raise FileNotFoundError(
                            f"Component '{name}' path not found: {full}"
                        )
                    custom_label = file_opts.get("filename")
                    if custom_label and os.path.isfile(full):
                        from .reference import FileReference

                        fr = FileReference(
                            full,
                            format="md",
                            label=str(custom_label),
                            inject=inject,
                            depth=depth,
                        )
                        all_refs.append(fr)
                        input_file_refs.append(fr)
                    else:
                        refs = create_file_references(
                            [full],
                            ignore_paths=None,
                            format="md",
                            label="relative",
                            inject=inject,
                            depth=depth,
                        )["refs"]
                        all_refs.extend(refs)
                        input_file_refs.extend(refs)

        attachment_lines = [f'<attachment label="{name}">']
        for idx, ref in enumerate(all_refs):
            attachment_lines.append(ref.output)
            if idx < len(all_refs) - 1:
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

        parts.append("\n".join(block_lines))

    return "\n\n".join(parts)


def assemble_payload_with_mdlinks(
    components: List[Dict[str, Any]],
    base_dir: str,
    *,
    inject: bool = False,
    depth: int = 5,
    link_depth_default: int = 0,
    link_scope_default: str = "all",
    link_skip_default: List[str] = None,
):
    """
    Assemble payload like assemble_payload, but also resolve Markdown links per component.
    """
    parts: List[str] = []
    all_input_refs = []
    all_trace_items = []
    all_skipped_paths = set()
    all_skip_impact = {}

    for comp in components:
        wrap_mode = comp.get("wrap")

        # text-only component passthrough
        if "text" in comp:
            text = comp["text"].rstrip()
            if wrap_mode:
                if wrap_mode.lower() == "md":
                    text = "```\n" + text + "\n```"
                else:
                    text = f"<{wrap_mode}>\n{text}\n</{wrap_mode}>"
            parts.append(text)
            continue

        name = comp.get("name")
        files = comp.get("files")
        if not name or not files:
            raise ValueError(
                f"Component must have either 'text' or both 'name' & 'files': {comp}"
            )

        prefix = comp.get("prefix", "").rstrip()
        suffix = comp.get("suffix", "").lstrip()

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

        for spec in files:
            spec, file_opts = _coerce_file_spec(spec)
            spec = os.path.expanduser(spec)

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

            seed_refs = []

            if spec.startswith("http://") or spec.startswith("https://"):
                opts = _parse_url_spec(spec)
                url = opts.get("target", spec)
                filename = file_opts.get("filename") or opts.get("filename")
                wrap = file_opts.get("wrap") or opts.get("wrap")

                tgt = parse_git_target(url)
                if tgt and (
                    tgt.path is not None
                    or tgt.repo_url.endswith(".git")
                    or tgt.repo_url != url
                ):
                    repo_dir = ensure_repo(tgt)
                    paths = (
                        [repo_dir]
                        if not tgt.path
                        else expand_git_paths(repo_dir, tgt.path)
                    )
                    for full in paths:
                        if not os.path.exists(full):
                            raise FileNotFoundError(
                                f"Component '{name}' path not found: {full}"
                            )
                        refs = create_file_references(
                            [full],
                            ignore_paths=None,
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
            else:
                tgt = parse_git_target(spec)
                if tgt:
                    repo_dir = ensure_repo(tgt)
                    paths = (
                        [repo_dir]
                        if not tgt.path
                        else expand_git_paths(repo_dir, tgt.path)
                    )
                else:
                    base = "" if os.path.isabs(spec) else base_dir
                    paths = expand_git_paths(base, spec)

                for full in paths:
                    if not os.path.exists(full):
                        raise FileNotFoundError(
                            f"Component '{name}' path not found: {full}"
                        )
                    custom_label = file_opts.get("filename")
                    if custom_label and os.path.isfile(full):
                        from .reference import FileReference

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
                            ignore_paths=None,
                            format="md",
                            label="relative",
                            inject=inject,
                            depth=depth,
                        )["refs"]
                        seed_refs.extend(refs)

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
                refs_for_attachment.extend(expanded_refs)
                all_trace_items.extend(comp_trace_items)
                if resolved_link_skip:
                    for skip_path in resolved_link_skip:
                        abs_skip_path = os.path.abspath(skip_path)
                        if os.path.exists(abs_skip_path):
                            all_skipped_paths.add(abs_skip_path)
                if comp_skip_impact:
                    all_skip_impact.update(comp_skip_impact)
            else:
                refs_for_attachment.extend(seed_refs)

        all_input_refs.extend(input_refs_for_comp)

        attachment_lines = [f'<attachment label="{name}">']
        for idx, ref in enumerate(refs_for_attachment):
            attachment_lines.append(ref.output)
            if idx < len(refs_for_attachment) - 1:
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

        parts.append("\n".join(block_lines))

    return (
        "\n\n".join(parts),
        all_input_refs,
        all_trace_items,
        base_dir,
        list(all_skipped_paths),
        all_skip_impact,
    )


def render_from_yaml(
    manifest_path: str,
    *,
    inject: bool = False,
    depth: int = 5,
) -> str:
    """
    Load YAML with top-level:
      config:
        root:  # optional, expands ~
      components:
        - text: ...
        - name: ...; prefix/suffix?; files: [...]
        - wrap:  # optional
    """
    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping with 'config' and 'components'")

    cfg = data.get("config", {})
    if "root" in cfg:
        raw = cfg["root"] or "~"
        base_dir = os.path.expanduser(raw)
    else:
        base_dir = os.path.dirname(os.path.abspath(manifest_path))

    comps = data.get("components")
    if not isinstance(comps, list):
        raise ValueError("'components' must be a list")

    return assemble_payload(comps, base_dir, inject=inject, depth=depth)


def render_from_yaml_with_mdlinks(
    manifest_path: str,
    *,
    inject: bool = False,
    depth: int = 5,
):
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

    return assemble_payload_with_mdlinks(
        comps,
        base_dir,
        inject=inject,
        depth=depth,
        link_depth_default=link_depth_default,
        link_scope_default=link_scope_default,
        link_skip_default=link_skip_default,
    )


def assemble_payload_with_mdlinks_from_data(
    data: Dict[str, Any],
    manifest_cwd: str,
    *,
    inject: bool = False,
    depth: int = 5,
):
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

    return assemble_payload_with_mdlinks(
        comps,
        base_dir,
        inject=inject,
        depth=depth,
        link_depth_default=link_depth_default,
        link_scope_default=link_scope_default,
        link_skip_default=link_skip_default,
    )
