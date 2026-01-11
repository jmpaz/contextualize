"""Markdown link discovery and traversal."""

import os
import re
from typing import List

from ..references import FileReference, create_file_references, is_utf8_file
from ..utils import count_tokens

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s#]+)\)")


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
        if os.path.exists(c) and is_utf8_file(c):
            return c
    return None


def _walk_markdown_links(seed_path, seed_content, max_depth, seen):
    queue = [(seed_path, seed_content, 0)]
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
            try:
                with open(tgt, "r", encoding="utf-8") as fh:
                    next_content = fh.read()
            except Exception:
                next_content = None
            yield tgt, path, depth + 1, next_content
            if next_content is not None:
                queue.append((tgt, next_content, depth + 1))


def count_downstream(path, content, depth, seen_set=None, token_target="cl100k_base"):
    """Count how many files would be discovered downstream from a skipped file."""
    if seen_set is None:
        seen_set = set([os.path.abspath(path)])
    else:
        seen_set = seen_set.copy()

    found_files = 0
    total_tokens = 0

    for target, _, _, next_content in _walk_markdown_links(
        path, content, depth, seen_set
    ):
        found_files += 1
        if next_content is None:
            continue
        token_info = count_tokens(next_content, target=token_target)
        total_tokens += token_info["count"]

    return found_files, total_tokens


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

    to_add: list[str] = []
    trace_items: list[tuple[str, str, int]] = []

    for seed in seeds:
        for path, parent, depth, _ in _walk_markdown_links(
            seed.path, seed.file_content, link_depth, seen
        ):
            to_add.append(path)
            trace_items.append((path, parent, depth))

    if not to_add:
        return refs, trace_items, skip_impact

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
