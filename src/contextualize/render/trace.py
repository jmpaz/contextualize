"""Trace output formatting for --trace flag."""

import os
from collections import defaultdict
from typing import Dict
from urllib.parse import urlparse

import yaml

from ..references import FileReference
from ..utils import count_tokens


def _trace_path_for_ref(ref) -> str:
    return (
        getattr(ref, "trace_path", None)
        or getattr(ref, "path", None)
        or getattr(ref, "url", None)
        or ""
    )


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

    all_paths = [_trace_path_for_ref(r) for r in input_refs] + [
        item[0] for item in trace_items
    ]
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
        path = _trace_path_for_ref(ref)
        seen_files.add(os.path.abspath(path))

    def get_rel_path(path):
        return (
            path[len(common_prefix) :].lstrip(os.sep)
            if common_prefix and path.startswith(common_prefix)
            else path
        )

    for ref in input_refs:
        path = _trace_path_for_ref(ref)
        rel_path = get_rel_path(path)
        if getattr(ref, "is_map", False):
            rel_path = f"[map] {rel_path}"
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
