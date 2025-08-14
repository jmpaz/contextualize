import os
import re
from typing import Iterable, List, Optional, Set, Tuple

from .reference import FileReference, _is_utf8_file, create_file_references

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


def _extract_local_hrefs(md: str) -> List[str]:
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


def _resolve_to_path(href: str, base_dir: str) -> Optional[str]:
    """
    Resolve a local href to a UTF-8 file path. Prefer '<href>.md' in same dir,
    then '<href>' verbatim. Supports absolute/relative hrefs.
    """
    candidates: List[str] = []
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


def _collect_linked_paths(
    seed_path: str, seed_content: str, max_depth: int, seen: Set[str]
) -> List[str]:
    """
    BFS from seed file following Markdown links up to max_depth.
    'seen' prevents re-crawl across all depths/seeds.
    """
    queue: List[Tuple[str, str, int]] = [(seed_path, seed_content, 0)]
    found: List[str] = []

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
            try:
                with open(tgt, "r", encoding="utf-8") as fh:
                    queue.append((tgt, fh.read(), depth + 1))
            except Exception:
                pass

    return found


def add_markdown_link_refs(
    refs: List[FileReference],
    *,
    link_depth: int,
    scope: str = "all",
    format_: str = "md",
    label: str = "relative",
    inject: bool = False,
) -> List:
    """
    Discover Markdown-linked files and append them as additional refs.
    - link_depth: max hop count (0 = off)
    - scope: follow links from only the first file ("first") or from all initial files ("all")
    """
    if link_depth <= 0 or not refs:
        return refs

    import os

    seen: Set[str] = set()
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
    to_add: List[str] = []
    for seed in seeds:
        to_add.extend(
            _collect_linked_paths(seed.path, seed.file_content, link_depth, seen)
        )

    if not to_add:
        return refs

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
    return refs

