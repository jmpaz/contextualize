import os
import re
from typing import Any, Dict, List

import yaml

from .gitcache import ensure_repo, expand_git_paths, parse_git_target
from .reference import URLReference, create_file_references
from .utils import wrap_text


def _parse_url_spec(spec: str) -> dict[str, Any]:
    """Parse URL spec like 'https://example.com::filename=file.py::wrap=md' into options dict."""
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


def assemble_payload(
    components: List[Dict[str, Any]],
    base_dir: str,
    *,
    inject: bool = False,
    depth: int = 5,
) -> str:
    """
    - If a component has a 'text' key, emit that text verbatim.
    - Otherwise it must have 'name' and 'files':
        optional 'prefix' (above the attachment) and 'suffix' (below).
      Files and directories are expanded via create_file_references().
    - if 'wrap' key is present:
        * wrap == "md" → wrap the inner content in a markdown code fence
        * wrap == other string → wrap inner content in <wrap>…</wrap>
    - if `inject` is true, resolve {cx::...} markers before wrapping files
    """
    parts: List[str] = []

    for comp in components:
        wrap_mode = comp.get("wrap")  # may be None, "md", or any tag name

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

        # 2) file component
        name = comp.get("name")
        files = comp.get("files")
        if not name or not files:
            raise ValueError(
                f"Component must have either 'text' or both 'name' & 'files': {comp}"
            )

        prefix = comp.get("prefix", "").rstrip()
        suffix = comp.get("suffix", "").lstrip()

        # collect FileReference objects (recursing into directories)
        all_refs = []
        for spec in files:
            spec = os.path.expanduser(spec)

            if spec.startswith("http://") or spec.startswith("https://"):
                opts = _parse_url_spec(spec)
                url = opts.get("target", spec)
                filename = opts.get("filename")
                wrap = opts.get("wrap")

                # try as git target first, fall back to URL
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
                else:
                    url_ref = URLReference(
                        url,
                        format="raw",
                        label=filename or url,
                        inject=inject,
                        depth=depth,
                    )

                    wrap_format = wrap or "md"
                    wrapped_content = wrap_text(
                        url_ref.output, wrap_format, filename or url
                    )

                    class SimpleReference:
                        def __init__(self, output: str):
                            self.output = output

                    all_refs.append(SimpleReference(wrapped_content))
            else:
                # handle git targets and local files
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
                    refs = create_file_references(
                        [full],
                        ignore_paths=None,
                        format="md",
                        label="relative",
                        inject=inject,
                        depth=depth,
                    )["refs"]
                    all_refs.extend(refs)

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
