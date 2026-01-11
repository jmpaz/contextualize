"""Factory function for resolving target strings into references."""

import codecs
import os
from glob import glob
from pathlib import Path

from .file import FileReference, _is_utf8_file, _MARKITDOWN_PREFERRED_EXTENSIONS
from .url import URLReference
from .symbols import split_path_and_symbols
from ..utils import brace_expand


def resolve(
    target: str,
    *,
    format: str = "md",
    label: str = "relative",
    label_suffix: str | None = None,
    include_token_count: bool = False,
    token_target: str = "cl100k_base",
    inject: bool = False,
    depth: int = 5,
    trace_collector: list | None = None,
    ignore_patterns: list[str] | None = None,
    git_pull: bool = False,
    git_reclone: bool = False,
) -> list:
    """
    Resolve a target string into one or more references.

    Handles:
    - Local paths with glob expansion
    - Git targets (github:user/repo:path, https://...)
    - HTTP URLs
    - Symbol extraction (path:SymbolName)

    Args:
        target: The target string to resolve
        format: Output format (md, xml, shell, raw)
        label: Label style (relative, name, ext)
        label_suffix: Optional suffix to add to labels
        include_token_count: Whether to include token counts
        token_target: Token counting target/encoding
        inject: Whether to process {cx::...} markers
        depth: Maximum depth for content injection
        trace_collector: Optional list to collect injection traces
        ignore_patterns: Patterns to ignore when expanding directories
        git_pull: Whether to pull git repositories
        git_reclone: Whether to reclone git repositories

    Returns:
        List of Reference objects
    """
    references = []

    # Expand brace patterns
    expanded_targets = brace_expand(target)

    for expanded_target in expanded_targets:
        refs = _resolve_single(
            expanded_target,
            format=format,
            label=label,
            label_suffix=label_suffix,
            include_token_count=include_token_count,
            token_target=token_target,
            inject=inject,
            depth=depth,
            trace_collector=trace_collector,
            ignore_patterns=ignore_patterns,
            git_pull=git_pull,
            git_reclone=git_reclone,
        )
        references.extend(refs)

    return references


def _resolve_single(
    target: str,
    *,
    format: str,
    label: str,
    label_suffix: str | None,
    include_token_count: bool,
    token_target: str,
    inject: bool,
    depth: int,
    trace_collector: list | None,
    ignore_patterns: list[str] | None,
    git_pull: bool,
    git_reclone: bool,
) -> list:
    """Resolve a single target (after brace expansion)."""
    # Check for URL
    if target.startswith("http://") or target.startswith("https://"):
        return [
            URLReference(
                target,
                format=format,
                _label_style=label,
                label_suffix=label_suffix,
                include_token_count=include_token_count,
                token_target=token_target,
                inject=inject,
                depth=depth,
                trace_collector=trace_collector,
            )
        ]

    # Check for git target
    from ...git.cache import parse_git_target, ensure_repo, expand_git_paths

    git_target = parse_git_target(target)
    if git_target:
        return _resolve_git_target(
            git_target,
            format=format,
            label=label,
            label_suffix=label_suffix,
            include_token_count=include_token_count,
            token_target=token_target,
            inject=inject,
            depth=depth,
            trace_collector=trace_collector,
            git_pull=git_pull,
            git_reclone=git_reclone,
        )

    # Local path
    return _resolve_local_path(
        target,
        format=format,
        label=label,
        label_suffix=label_suffix,
        include_token_count=include_token_count,
        token_target=token_target,
        inject=inject,
        depth=depth,
        trace_collector=trace_collector,
        ignore_patterns=ignore_patterns,
    )


def _resolve_git_target(
    git_target,
    *,
    format: str,
    label: str,
    label_suffix: str | None,
    include_token_count: bool,
    token_target: str,
    inject: bool,
    depth: int,
    trace_collector: list | None,
    git_pull: bool,
    git_reclone: bool,
) -> list:
    """Resolve a git target to references."""
    from ...git.cache import ensure_repo, expand_git_paths
    from .git import GitCacheReference

    repo_dir = ensure_repo(git_target, pull=git_pull, reclone=git_reclone)
    references = []

    if git_target.path:
        paths = expand_git_paths(repo_dir, git_target.path)
    else:
        paths = [repo_dir]

    for path in paths:
        if os.path.isfile(path):
            rel_path = os.path.relpath(path, repo_dir)
            ref = GitCacheReference(
                cache_dir=repo_dir,
                rel_path=rel_path,
                format=format,
                _label_style=label,
                label_suffix=label_suffix,
                include_token_count=include_token_count,
                token_target=token_target,
                inject=inject,
                depth=depth,
                trace_collector=trace_collector,
            )
            references.append(ref)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if _is_utf8_file(file_path) or Path(file_path).suffix.lower() in _MARKITDOWN_PREFERRED_EXTENSIONS:
                        rel_path = os.path.relpath(file_path, repo_dir)
                        ref = GitCacheReference(
                            cache_dir=repo_dir,
                            rel_path=rel_path,
                            format=format,
                            _label_style=label,
                            label_suffix=label_suffix,
                            include_token_count=include_token_count,
                            token_target=token_target,
                            inject=inject,
                            depth=depth,
                            trace_collector=trace_collector,
                        )
                        references.append(ref)

    return references


def _resolve_local_path(
    target: str,
    *,
    format: str,
    label: str,
    label_suffix: str | None,
    include_token_count: bool,
    token_target: str,
    inject: bool,
    depth: int,
    trace_collector: list | None,
    ignore_patterns: list[str] | None,
) -> list:
    """Resolve a local path to references."""
    path, symbols = split_path_and_symbols(target)
    references = []

    if os.path.isfile(path):
        if not _is_utf8_file(path) and Path(path).suffix.lower() not in _MARKITDOWN_PREFERRED_EXTENSIONS:
            raise ValueError(f"Unsupported binary file type (not convertible): {path}")

        # Process symbols if specified
        ranges = None
        if symbols:
            if not _is_utf8_file(path):
                raise ValueError(f"Symbol selection is only supported for text files: {path}")
            try:
                from ..repomap import find_symbol_ranges

                match_map = find_symbol_ranges(path, symbols)
            except Exception:
                match_map = {}

            matched = [s for s in symbols if s in match_map]
            if matched:
                ranges = [match_map[s] for s in matched]
                symbols = matched

        ref = FileReference(
            path,
            ranges=ranges,
            symbols=symbols,
            format=format,
            label=label,
            label_suffix=label_suffix,
            include_token_count=include_token_count,
            token_target=token_target,
            inject=inject,
            depth=depth,
            trace_collector=trace_collector,
        )
        references.append(ref)

    elif os.path.isdir(path):
        references.extend(
            _expand_directory(
                path,
                format=format,
                label=label,
                label_suffix=label_suffix,
                include_token_count=include_token_count,
                token_target=token_target,
                inject=inject,
                depth=depth,
                trace_collector=trace_collector,
                ignore_patterns=ignore_patterns,
            )
        )

    else:
        # Try glob expansion
        matches = glob(path, recursive=True)
        for match in matches:
            if os.path.isfile(match):
                if _is_utf8_file(match) or Path(match).suffix.lower() in _MARKITDOWN_PREFERRED_EXTENSIONS:
                    ref = FileReference(
                        match,
                        format=format,
                        label=label,
                        label_suffix=label_suffix,
                        include_token_count=include_token_count,
                        token_target=token_target,
                        inject=inject,
                        depth=depth,
                        trace_collector=trace_collector,
                    )
                    references.append(ref)

    return references


def _expand_directory(
    path: str,
    *,
    format: str,
    label: str,
    label_suffix: str | None,
    include_token_count: bool,
    token_target: str,
    inject: bool,
    depth: int,
    trace_collector: list | None,
    ignore_patterns: list[str] | None,
) -> list:
    """Expand a directory into file references."""
    from pathspec import PathSpec

    default_ignore_patterns = [
        ".gitignore",
        ".git/",
        "__pycache__/",
        "__init__.py",
    ]

    all_ignore_patterns = default_ignore_patterns[:]
    if ignore_patterns:
        for pattern in ignore_patterns:
            if "{" in pattern and "}" in pattern:
                all_ignore_patterns.extend(brace_expand(pattern))
            else:
                all_ignore_patterns.append(pattern)

    path_spec = PathSpec.from_lines("gitwildmatch", all_ignore_patterns)
    references = []

    for root, dirs, files in os.walk(path):
        # Filter directories
        dirs[:] = [
            d for d in dirs
            if not path_spec.match_file(os.path.join(root, d))
        ]

        for file in files:
            file_path = os.path.join(root, file)
            if path_spec.match_file(file_path):
                continue

            if _is_utf8_file(file_path) or Path(file_path).suffix.lower() in _MARKITDOWN_PREFERRED_EXTENSIONS:
                ref = FileReference(
                    file_path,
                    format=format,
                    label=label,
                    label_suffix=label_suffix,
                    include_token_count=include_token_count,
                    token_target=token_target,
                    inject=inject,
                    depth=depth,
                    trace_collector=trace_collector,
                )
                references.append(ref)

    return references
