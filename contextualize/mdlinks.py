import os
import re
from collections import defaultdict

from .reference import FileReference, _is_utf8_file, create_file_references
from .tokenize import count_tokens

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s#]+)\)")  # ignore anchors/fragments


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


def _extract_local_hrefs(md):
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


def _resolve_to_path(href, base_dir):
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
):
    if not input_refs and not trace_items and not stdin_data:
        return ""

    all_paths = [r.path for r in input_refs] + [item[0] for item in trace_items]
    if skipped_paths:
        all_paths.extend(skipped_paths)

    common_prefix = common_prefix or os.path.dirname(
        os.path.commonpath(all_paths) if all_paths else ""
    )

    formatted_inputs = []
    formatted_discovered = {}
    formatted_skipped = []

    def get_rel_path(path):
        return (
            path[len(common_prefix) :].lstrip(os.sep)
            if common_prefix and path.startswith(common_prefix)
            else path
        )

    for ref in input_refs:
        rel_path = get_rel_path(ref.path)
        token_count = (
            count_tokens(ref.file_content)["count"]
            if hasattr(ref, "file_content")
            else 0
        )
        formatted_inputs.append((rel_path, token_count, None))

    by_depth = defaultdict(list)
    for tgt, src, depth in trace_items:
        by_depth[depth].append((tgt, src))

    for depth in sorted(by_depth.keys()):
        depth_items = []
        for tgt, src in sorted(by_depth[depth]):
            ref = FileReference(tgt)
            rel_path = get_rel_path(tgt)
            token_count = (
                count_tokens(ref.file_content)["count"]
                if hasattr(ref, "file_content")
                else 0
            )
            source_name = os.path.basename(src)
            depth_items.append((rel_path, token_count, source_name))
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
        stdin_token_count = count_tokens(stdin_data)["count"]
        lines.append(f"  stdin ({stdin_token_count} tokens)")

    for rel_path, token_count, _ in formatted_inputs:
        lines.append(f"  {rel_path} ({token_count} tokens)")

    for depth in sorted(formatted_discovered.keys()):
        lines.append(f"\nDiscovered (depth {depth}):")

        path_token_widths = [
            len(f"{p} ({t})") for p, t, _ in formatted_discovered[depth]
        ]
        max_path_token_width = max(path_token_widths, default=0)

        for rel_path, token_count, source_name in formatted_discovered[depth]:
            path_with_tokens = f"{rel_path} ({token_count})"
            padding = max_path_token_width - len(path_with_tokens)

            lines.append(f"  {path_with_tokens}{' ' * padding} ← {source_name}")

    if formatted_skipped:
        lines.append("\nSkipped:")
        for (
            rel_path,
            file_tokens,
            downstream_files,
            downstream_tokens,
        ) in formatted_skipped:
            if downstream_files > 0:
                lines.append(
                    f"  {rel_path} → {downstream_files} additional files ({downstream_tokens} tokens)"
                )
            else:
                lines.append(f"  {rel_path} ({file_tokens} tokens)")

    return "\n".join(lines)


def count_downstream(path, content, depth, seen_set=None):
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
                    token_info = count_tokens(content)
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
    inject=False,
    link_skip=None,
):
    """
    Discover Markdown-linked files and append them as additional refs.
    - link_depth: max hop count (0 = off)
    - scope: follow links from only the first file ("first") or from all initial files ("all")
    """
    if link_depth <= 0 or not refs:
        return refs, []

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
                        token_info = count_tokens(content)
                        downstream_count, downstream_tokens = count_downstream(
                            abs_path, content, link_depth, seen
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
    to_add = []
    trace_items = []

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
        ignore_paths=None,
        format=format_,
        label=label,
        inject=inject,
        depth=link_depth,
    )["refs"]
    refs.extend(more)
    return refs, trace_items, skip_impact
