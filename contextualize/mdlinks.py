import os
import os.path
import re
from collections import defaultdict
from urllib.parse import urlparse

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
    injection_traces=None,
    global_seen=None,
    heading=None,
):
    if not input_refs and not trace_items and not stdin_data and not injection_traces:
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

    seen_files = set()
    # seed with previously seen files (for multi-payload dedupe)
    if global_seen:
        for p in global_seen:
            try:
                seen_files.add(os.path.abspath(p))
            except Exception:
                pass
    for ref in input_refs:
        seen_files.add(os.path.abspath(ref.path))

    def get_rel_path(path):
        return (
            path[len(common_prefix) :].lstrip(os.sep)
            if common_prefix and path.startswith(common_prefix)
            else path
        )

    for ref in input_refs:
        rel_path = get_rel_path(ref.path)
        original_content = getattr(ref, "original_file_content", None)
        final_content = getattr(ref, "file_content", "")
        abs_path = os.path.abspath(ref.path)

        # if already seen globally, show as duplicate ✓
        if global_seen and abs_path in {os.path.abspath(p) for p in global_seen}:
            token_display = None
        else:
            if original_content and original_content != final_content:
                original_tokens = count_tokens(original_content)["count"]
                final_tokens = count_tokens(final_content)["count"]
                token_display = (original_tokens, final_tokens)
            else:
                token_count = count_tokens(final_content)["count"] if final_content else 0
                token_display = token_count

        formatted_inputs.append((rel_path, token_display, None))

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
                ref = FileReference(tgt)
                token_count = (
                    count_tokens(ref.file_content)["count"]
                    if hasattr(ref, "file_content")
                    else 0
                )
                seen_files.add(abs_tgt)
            else:
                token_count = None

            chain = build_source_chain(abs_tgt, max_len=depth)
            depth_items.append((rel_path, token_count, chain))
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

    if heading:
        lines = [f"\x1b[1m{heading}\x1b[0m", "Inputs:"]
    else:
        lines = ["Inputs:"]

    if stdin_data:
        stdin_token_count = count_tokens(stdin_data)["count"]
        lines.append(f"  stdin ({stdin_token_count} tokens)")

    for rel_path, token_display, _ in formatted_inputs:
        if token_display is None:
            lines.append(f"  {rel_path} (✓)")
        elif isinstance(token_display, tuple):
            original, final = token_display
            lines.append(f"  {rel_path} ({original} → {final} tokens)")
        else:
            lines.append(f"  {rel_path} ({token_display} tokens)")

    for depth in sorted(formatted_discovered.keys()):
        lines.append(f"\nDiscovered (depth {depth}):")

        path_token_widths = []
        for p, t, _ in formatted_discovered[depth]:
            if t is None:
                path_token_widths.append(len(f"{p} (✓)"))
            else:
                path_token_widths.append(len(f"{p} ({t})"))
        max_path_token_width = max(path_token_widths, default=0)

        for rel_path, token_count, source_chain in formatted_discovered[depth]:
            if token_count is None:
                path_with_tokens = f"{rel_path} (✓)"
            else:
                path_with_tokens = f"{rel_path} ({token_count})"
            padding = max_path_token_width - len(path_with_tokens)

            arrow_and_chain = f" ← {source_chain}" if source_chain else ""
            lines.append(f"  {path_with_tokens}{' ' * padding}{arrow_and_chain}")

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
    seen=None,
):
    """
    Discover Markdown-linked files and append them as additional refs.
    - link_depth: max hop count (0 = off)
    - scope: follow links from only the first file ("first") or from all initial files ("all")
    """
    if link_depth <= 0 or not refs:
        return refs, []

    # allow a shared 'seen' set to be provided by caller so multiple
    # payloads can share traversal state and dedupe across runs
    seen = set() if seen is None else seen

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
