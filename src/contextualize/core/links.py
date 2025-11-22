"""Markdown link discovery and {cx::...} content injection helpers."""

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import yaml
from urllib.parse import urlparse

from ..git.cache import ensure_repo, expand_git_paths, parse_git_target
from .references import (
    FileReference,
    URLReference,
    _is_utf8_file,
    create_file_references,
)
from .utils import count_tokens, wrap_text

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s#]+)\)")  # ignore anchors/fragments
_INJECTION_PATTERN = re.compile(r"\{cx::((?:[^{}]|\{[^{}]*\})*)\}")


# Markdown link traversal -----------------------------------------------------
def _strip_fenced_code(md: str) -> str:
    """Remove fenced code blocks to avoid false positives in links."""
    out, in_fence = [], False
    for line in md.splitlines():
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


def _extract_local_hrefs(md: str) -> list[str]:
    """Return hrefs that look local (no scheme/mailto), as written."""
    md = _strip_fenced_code(md)
    hrefs = []
    for _text, href in _LINK_RE.findall(md):
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        if href.startswith("#"):
            continue
        if href.startswith("<") and href.endswith(">"):
            href = href[1:-1]
        hrefs.append(href)
    return hrefs


def _resolve_to_path(href: str, base_dir: str) -> str | None:
    """Resolve a local href to a UTF-8 file path."""
    candidates = []
    if os.path.isabs(href):
        candidates.extend([href, href + ".md"])
    else:
        candidates.extend(
            [
                os.path.join(base_dir, href + ".md"),
                os.path.join(base_dir, href),
            ]
        )
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.exists(c) and _is_utf8_file(c):
            return c
    return None


def _parse_frontmatter_title(text: str) -> str | None:
    """Extract `title` from YAML frontmatter if present."""
    if not text:
        return None
    try:
        if not text.lstrip().startswith("---"):
            return None
        lines = text.splitlines()
        if not lines or not lines[0].strip().startswith("---"):
            return None
        end_idx = None
        for i in range(1, min(len(lines), 200)):
            if lines[i].strip() == "---":
                end_idx = i
                break
        if end_idx is None:
            return None
        yaml_text = "\n".join(lines[1:end_idx])
        if not yaml_text.strip():
            return None
        data = yaml.safe_load(yaml_text)
        if isinstance(data, dict) and "title" in data:
            title = data.get("title")
            if title is None:
                return None
            title_str = str(title).strip().replace("\n", " ")
            return title_str if title_str else None
    except Exception:
        return None
    return None


def compute_input_token_details(input_refs, token_target: str = "cl100k_base"):
    """
    Compute token counts and optional titles for a list of refs.
    Returns (total_tokens, {id(ref): {token_display, token_value, title}}).
    """
    total_tokens = 0
    details: Dict[int, Dict[str, object]] = {}

    for ref in input_refs:
        ref_id = id(ref)
        original_content = getattr(ref, "original_file_content", None)
        final_content = getattr(ref, "file_content", None)

        if final_content is None and original_content is not None:
            final_content = original_content

        title = _parse_frontmatter_title(original_content or final_content or "")

        if original_content and final_content and original_content != final_content:
            original_tokens = count_tokens(original_content, target=token_target)[
                "count"
            ]
            final_tokens = count_tokens(final_content, target=token_target)["count"]
            token_display = (original_tokens, final_tokens)
            token_value = final_tokens
        else:
            token_value = (
                count_tokens(final_content, target=token_target)["count"]
                if final_content
                else 0
            )
            token_display = token_value

        details[ref_id] = {
            "token_display": token_display,
            "token_value": token_value,
            "title": title,
        }
        total_tokens += token_value

    return total_tokens, details


def _collect_linked_paths(seed_path, seed_content, max_depth, seen):
    """
    BFS from seed file following Markdown links up to max_depth.
    'seen' prevents re-crawl across all depths/seeds.
    """
    queue = [(seed_path, seed_content, 0)]
    found = []
    trace_items = []

    while queue:
        path, content, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        base = os.path.dirname(path)
        for href in _extract_local_hrefs(content):
            tgt = _resolve_to_path(href, base)
            if not tgt or tgt in seen:
                continue
            seen.add(tgt)
            found.append(tgt)
            trace_items.append((tgt, path, depth + 1))
            try:
                with open(tgt, "r", encoding="utf-8") as fh:
                    queue.append((tgt, fh.read(), depth + 1))
            except Exception:
                pass

    return found, trace_items


def format_trace_output(
    input_refs,
    trace_items,
    skipped_paths=None,
    skip_impact=None,
    common_prefix=None,
    stdin_data=None,
    injection_traces=None,
    ignored_files=None,
    ignored_folders=None,
    token_target="cl100k_base",
    input_token_details=None,
    sort_inputs_by_tokens=False,
):
    if not input_refs and not trace_items and not stdin_data and not injection_traces:
        return ""

    all_paths = [r.path for r in input_refs] + [item[0] for item in trace_items]
    if skipped_paths:
        all_paths.extend(skipped_paths)
    if ignored_files:
        all_paths.extend([path for path, _ in ignored_files])

    if common_prefix is None:
        try:
            common_prefix = (
                os.path.dirname(os.path.commonpath(all_paths)) if all_paths else ""
            )
        except Exception:
            common_prefix = ""

    formatted_inputs = []
    formatted_discovered = {}
    formatted_skipped = []

    seen_files = set()
    for ref in input_refs:
        path = getattr(ref, "path", None) or getattr(ref, "url", None) or ""
        seen_files.add(os.path.abspath(path))

    def get_rel_path(path):
        return (
            path[len(common_prefix) :].lstrip(os.sep)
            if common_prefix and path.startswith(common_prefix)
            else path
        )

    for ref in input_refs:
        path = getattr(ref, "path", None) or getattr(ref, "url", None) or ""
        rel_path = get_rel_path(path)
        detail = input_token_details.get(id(ref)) if input_token_details else None

        if detail:
            token_display = detail.get("token_display", 0)
            token_value = detail.get("token_value", 0)
            title = detail.get("title")
        else:
            original_content = getattr(ref, "original_file_content", None)
            final_content = getattr(ref, "file_content", None)
            if final_content is None and original_content is not None:
                final_content = original_content
            final_content = final_content or ""
            title = _parse_frontmatter_title(original_content or final_content)

            if original_content and original_content != final_content:
                original_tokens = count_tokens(original_content, target=token_target)[
                    "count"
                ]
                final_tokens = count_tokens(final_content, target=token_target)["count"]
                token_display = (original_tokens, final_tokens)
                token_value = final_tokens
            else:
                token_value = (
                    count_tokens(final_content, target=token_target)["count"]
                    if final_content
                    else 0
                )
                token_display = token_value

        formatted_inputs.append((rel_path, token_display, title, token_value))

    if sort_inputs_by_tokens:
        formatted_inputs.sort(
            key=lambda item: item[3] if item[3] is not None else 0, reverse=True
        )

    by_depth = defaultdict(list)
    parent_map = {}
    for tgt, src, depth in trace_items:
        by_depth[depth].append((tgt, src))
        abs_tgt = os.path.abspath(tgt)
        abs_src = os.path.abspath(src)
        if abs_tgt not in parent_map:
            parent_map[abs_tgt] = abs_src

    def build_source_chain(abs_target_path, max_len=None):
        """
        Return a display string like 'a.md ← b.md ← c.md' from immediate → seed.
        """
        chain_parts = []
        seen_chain = set()
        cur = abs_target_path
        while cur in parent_map and cur not in seen_chain:
            seen_chain.add(cur)
            parent = parent_map[cur]
            display = os.path.basename(get_rel_path(parent))
            chain_parts.append(display)
            cur = parent
            if max_len is not None and len(chain_parts) >= max_len:
                break
        return " ← ".join(chain_parts)

    for depth in sorted(by_depth.keys()):
        depth_items = []
        for tgt, _src in sorted(by_depth[depth]):
            abs_tgt = os.path.abspath(tgt)
            rel_path = get_rel_path(tgt)

            is_duplicate = abs_tgt in seen_files
            if not is_duplicate:
                ref = FileReference(tgt, token_target=token_target)
                token_count = (
                    count_tokens(ref.file_content, target=token_target)["count"]
                    if hasattr(ref, "file_content")
                    else 0
                )
                title = _parse_frontmatter_title(getattr(ref, "file_content", ""))
                seen_files.add(abs_tgt)
            else:
                token_count = None
                title = None

            chain = build_source_chain(abs_tgt, max_len=depth)
            depth_items.append((rel_path, token_count, chain, title))
        formatted_discovered[depth] = depth_items

    if skipped_paths:
        for path in sorted(skipped_paths):
            rel_path = get_rel_path(path)
            if skip_impact and path in skip_impact:
                impact = skip_impact[path]
                file_tokens = impact["file_tokens"]
                downstream_files = impact["downstream_files"]
                downstream_tokens = impact["downstream_tokens"]

                if downstream_files > 0:
                    formatted_skipped.append(
                        (rel_path, file_tokens, downstream_files, downstream_tokens)
                    )
                else:
                    formatted_skipped.append((rel_path, file_tokens, 0, 0))
            else:
                formatted_skipped.append((rel_path, 0, 0, 0))

    lines = ["Inputs:"]

    if stdin_data:
        stdin_token_count = count_tokens(stdin_data, target=token_target)["count"]
        lines.append(f"  stdin ({stdin_token_count} tokens)")

    for rel_path, token_display, title, _token_value in formatted_inputs:
        if isinstance(token_display, tuple):
            original, final = token_display
            token_str = f"({original} → {final} tokens)"
        else:
            token_str = f"({token_display} tokens)"
        if title:
            line = f"  {rel_path} — {title} {token_str}"
        else:
            line = f"  {rel_path} {token_str}"
        lines.append(line)

    for depth in sorted(formatted_discovered.keys()):
        lines.append(f"\nDiscovered (depth {depth}):")

        path_token_widths = []
        for p, t, _chain, _title in formatted_discovered[depth]:
            token_part = "(✓)" if t is None else f"({t})"
            title_part = f" — {_title}" if _title else ""
            left_text = f"{p}{title_part} {token_part}"
            path_token_widths.append(len(left_text))
        max_path_token_width = max(path_token_widths, default=0)

        for rel_path, token_count, source_chain, title in formatted_discovered[depth]:
            token_part = "(✓)" if token_count is None else f"({token_count})"
            title_part = f" — {title}" if title else ""
            left_text = f"{rel_path}{title_part} {token_part}"
            padding = max_path_token_width - len(left_text)

            arrow_and_chain = f" ← {source_chain}" if source_chain else ""
            line = f"  {left_text}{' ' * padding}{arrow_and_chain}"
            lines.append(line)

    if formatted_skipped:
        lines.append("\nSkipped:")
        for (
            rel_path,
            file_tokens,
            downstream_files,
            downstream_tokens,
        ) in formatted_skipped:
            lines.append(
                f"  {rel_path} → {downstream_files} additional files ({downstream_tokens} tokens)"
                if downstream_files > 0
                else f"  {rel_path} ({file_tokens} tokens)"
            )

    if ignored_files or ignored_folders:
        lines.append("\nIgnored:")
        if ignored_folders:
            for folder_path, (file_count, total_tokens) in sorted(
                ignored_folders.items()
            ):
                rel_path = get_rel_path(folder_path)
                lines.append(
                    f"  {rel_path}/ ({file_count} files, {total_tokens} tokens)"
                )
        if ignored_files:
            for file_path, token_count in ignored_files:
                rel_path = get_rel_path(file_path)
                lines.append(f"  {rel_path} ({token_count} tokens)")

    if injection_traces:
        lines.append("\nInjected:")
        injections_by_source = defaultdict(list)
        for trace in injection_traces:
            if trace[0] == "injection":
                _, target, source, pattern, tokens = trace
                injections_by_source[source].append((target, pattern, tokens))

        for source in sorted(injections_by_source.keys()):
            source_rel = (
                source[len(common_prefix) :].lstrip(os.sep)
                if common_prefix and source.startswith(common_prefix)
                else source
            )
            for target, pattern, tokens in injections_by_source[source]:
                if pattern.startswith("{cx::") and pattern.endswith("}"):
                    content_part = pattern[5:-1]
                    display_pattern = (
                        "{cx::" + content_part[:35] + "...}"
                        if len(content_part) > 40
                        else pattern
                    )
                else:
                    display_pattern = pattern

                if target.startswith(("http://", "https://")):
                    parsed = urlparse(target)
                    if len(target) > 60:
                        path_parts = parsed.path.split("/")
                        display_target = (
                            f"{parsed.netloc}/.../{path_parts[-1][:20]}"
                            if len(path_parts) > 3
                            else target[:57] + "..."
                        )
                    else:
                        display_target = target
                elif target.startswith("git@") or ".git" in target:
                    display_target = target
                else:
                    home = os.path.expanduser("~")
                    if target.startswith(home):
                        display_target = "~" + target[len(home) :]
                    else:
                        display_target = (
                            target[len(common_prefix) :].lstrip(os.sep)
                            if common_prefix and target.startswith(common_prefix)
                            else target
                        )

                home = os.path.expanduser("~")
                source_display = (
                    "~" + source_rel[len(home) :]
                    if source_rel.startswith(home)
                    else source_rel
                )
                lines.append(
                    f"  {display_target} ({tokens} tokens) ← {display_pattern} in {source_display}"
                )

    return "\n".join(lines)


def count_downstream(path, content, depth, seen_set=None, token_target="cl100k_base"):
    """Count how many files would be discovered downstream from a skipped file."""
    if seen_set is None:
        seen_set = set([os.path.abspath(path)])
    else:
        seen_set = seen_set.copy()

    queue = [(path, content, 0)]
    found_files = []
    total_tokens = 0

    while queue:
        current_path, current_content, current_depth = queue.pop(0)
        if current_depth >= depth:
            continue

        base_dir = os.path.dirname(current_path)
        for href in _extract_local_hrefs(current_content):
            target = _resolve_to_path(href, base_dir)
            if not target or target in seen_set:
                continue

            seen_set.add(target)
            found_files.append(target)

            try:
                with open(target, "r", encoding="utf-8") as fh:
                    content = fh.read()
                    token_info = count_tokens(content, target=token_target)
                    total_tokens += token_info["count"]
                    queue.append((target, content, current_depth + 1))
            except Exception:
                pass

    return len(found_files), total_tokens


def add_markdown_link_refs(
    refs,
    *,
    link_depth,
    scope="all",
    format_="md",
    label="relative",
    token_target="cl100k_base",
    include_token_count=False,
    inject=False,
    link_skip=None,
):
    """
    Discover Markdown-linked files and append them as additional refs.
    - link_depth: max hop count (0 = off)
    - scope: follow links from only the first file ("first") or from all initial files ("all")
    """
    if link_depth <= 0 or not refs:
        return refs, [], {}

    seen = set()

    skipped_paths = []
    skip_impact = {}
    if link_skip:
        for path in link_skip:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        token_info = count_tokens(content, target=token_target)
                        downstream_count, downstream_tokens = count_downstream(
                            abs_path, content, link_depth, seen, token_target
                        )
                        skip_impact[abs_path] = {
                            "file_tokens": token_info["count"],
                            "downstream_files": downstream_count,
                            "downstream_tokens": downstream_tokens,
                        }
                except Exception:
                    pass

                seen.add(abs_path)
                skipped_paths.append(abs_path)

    for r in refs:
        if isinstance(r, FileReference):
            seen.add(os.path.abspath(r.path))

    seeds: List[FileReference]
    if scope == "all":
        seeds = [r for r in refs if isinstance(r, FileReference)]
    else:
        seeds = [next((r for r in refs if isinstance(r, FileReference)), None)]
        seeds = [s for s in seeds if s is not None]

    # collect new paths
    to_add: list[str] = []
    trace_items: list[tuple[str, str, int]] = []

    for seed in seeds:
        paths, new_traces = _collect_linked_paths(
            seed.path, seed.file_content, link_depth, seen
        )
        to_add.extend(paths)
        trace_items.extend(new_traces)

    if not to_add:
        return refs, trace_items, skip_impact

    # materialize new paths as normal file references
    more = create_file_references(
        to_add,
        ignore_patterns=None,
        format=format_,
        label=label,
        token_target=token_target,
        include_token_count=include_token_count,
        inject=inject,
        depth=link_depth,
    )["refs"]
    refs.extend(more)
    return refs, trace_items, skip_impact


# Injection helpers ----------------------------------------------------------
def _parse_injection_spec(piece: str) -> dict[str, Any]:
    parts = piece.split("::")
    opts: dict[str, str | None] = {}
    target_parts: list[str] = []
    for part in parts:
        m = re.fullmatch(r'(filename|params|root|wrap)=(?:"([^\"]*)"|([^\"]*))', part)
        if m:
            opts[m.group(1)] = m.group(2) or m.group(3)
        else:
            target_parts.append(part)
    opts["target"] = "::".join(target_parts)
    return opts


def _http_fetch(
    url: str,
    name: str | None,
    depth: int,
    wrap: str | None = None,
    trace_collector: Optional[list] = None,
) -> str:
    try:
        ref = URLReference(
            url,
            format="raw",
            label=name or url,
            inject=depth > 0,
            depth=depth,
            trace_collector=trace_collector,
        )
        return wrap_text(ref.output, wrap or "md", name)
    except Exception as e:
        raise Exception(f"http fetch failed for {url}: {e}")


def _git_fetch(
    target: str,
    params: str | None,
    depth: int,
    wrap: str | None = None,
    trace_collector: Optional[list] = None,
) -> str:
    git_opts = params.split() if params else []
    fmt = "md"
    lbl = "relative"
    pull = "--git-pull" in git_opts
    reclone = "--git-reclone" in git_opts
    for i, p in enumerate(git_opts):
        if p == "--format" and i + 1 < len(git_opts):
            fmt = git_opts[i + 1]
        if p == "--label" and i + 1 < len(git_opts):
            lbl = git_opts[i + 1]

    tgt = parse_git_target(target)
    if not tgt:
        raise Exception(f"invalid git target: {target}")
    repo = ensure_repo(tgt, pull=pull, reclone=reclone)
    paths = [repo] if not tgt.path else expand_git_paths(repo, tgt.path)
    refs = create_file_references(
        paths,
        format=fmt,
        label=lbl,
        inject=True,
        depth=depth,
        trace_collector=trace_collector,
    )
    result = refs["concatenated"]
    return wrap_text(result, wrap) if wrap else result


def _local_fetch(
    path: str,
    root: str | None,
    params: str | None,
    depth: int,
    wrap: str | None = None,
    filename: str | None = None,
    trace_collector: Optional[list] = None,
) -> str:
    fmt = "md"
    lbl = "relative"
    if params:
        parts = params.split()
        for i, p in enumerate(parts):
            if p == "--format" and i + 1 < len(parts):
                fmt = parts[i + 1]
            if p == "--label" and i + 1 < len(parts):
                lbl = parts[i + 1]
    base = os.path.expanduser(root) if root else os.getcwd()
    if os.path.isabs(path):
        base = ""
    paths = expand_git_paths(base, os.path.expanduser(path))
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        raise Exception(f"path not found: {path}")

    label = filename if filename and len(existing) == 1 else lbl
    refs = create_file_references(
        existing,
        format=fmt,
        label=label,
        inject=True,
        depth=depth,
        trace_collector=trace_collector,
    )
    result = refs["concatenated"]
    return wrap_text(result, wrap, filename) if wrap else result


def _process_injection(
    opts: dict[str, Any],
    depth: int,
    trace_collector: Optional[list] = None,
    source_file: Optional[str] = None,
    pattern_text: Optional[str] = None,
) -> str:
    tgt = opts.get("target") or ""

    def _add_trace(result, resolved_tgt=None):
        if trace_collector is not None and source_file and pattern_text:
            trace_collector.append(
                (
                    "injection",
                    resolved_tgt or tgt,
                    source_file,
                    pattern_text,
                    count_tokens(result).get("count", 0),
                )
            )

    if tgt.startswith(("http://", "https://")):
        try:
            result = _http_fetch(
                tgt, opts.get("filename"), depth, opts.get("wrap"), trace_collector
            )
        except Exception:
            if not parse_git_target(tgt):
                raise
            result = _git_fetch(
                tgt, opts.get("params"), depth, opts.get("wrap"), trace_collector
            )
        _add_trace(result)
    elif parse_git_target(tgt):
        result = _git_fetch(
            tgt, opts.get("params"), depth, opts.get("wrap"), trace_collector
        )
        _add_trace(result)
    else:
        result = _local_fetch(
            tgt,
            opts.get("root"),
            opts.get("params"),
            depth,
            opts.get("wrap"),
            opts.get("filename"),
            trace_collector,
        )
        root = opts.get("root")
        base = os.path.expanduser(root) if root else os.getcwd()
        resolved = (
            tgt
            if "{" in tgt and "}" in tgt
            else (
                os.path.expanduser(tgt)
                if os.path.isabs(tgt)
                else os.path.join(base, os.path.expanduser(tgt))
            )
        )
        _add_trace(result, resolved)
    return result


def inject_content_in_text(
    text: str,
    depth: int = 5,
    trace_collector: Optional[list] = None,
    source_file: Optional[str] = None,
) -> str:
    if depth <= 0:
        return text
    new = _INJECTION_PATTERN.sub(
        lambda m: _process_injection(
            _parse_injection_spec(m.group(1)),
            depth - 1,
            trace_collector,
            source_file,
            m.group(0),
        ),
        text,
    )
    return (
        inject_content_in_text(new, depth - 1, trace_collector, source_file)
        if _INJECTION_PATTERN.search(new)
        else new
    )
