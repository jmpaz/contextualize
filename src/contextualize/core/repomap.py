import os
import tempfile

from aider.repomap import (
    RepoMap,
    Tag,
    UPDATING_REPO_MAP_MESSAGE,
    Spinner,
    filter_important_files,
)
from .utils import count_tokens
from grep_ast import TreeContext
from .references import _is_utf8_file
from ..git.rev import list_files_at_rev, read_file_at_rev


def _parse_symbol_spec(spec: str) -> tuple[str, str, int | None]:
    raw = spec.strip()
    if not raw:
        return "", "", None
    line_hint = None
    name_part = raw
    if "#" in raw:
        before, _, maybe_line = raw.rpartition("#")
        if maybe_line.isdigit():
            line_hint = int(maybe_line)
            name_part = before or raw
    parts = [p for p in name_part.split(".") if p]
    name = parts[-1] if parts else name_part
    return raw, name, line_hint


def find_symbol_ranges(
    file_path: str, symbol_specs: list[str], text: str | None = None
) -> dict[str, tuple[int, int]]:
    """
    Return {symbol_spec: (start_line, end_line)} using tree-sitter tags.
    Lines are 1-based and inclusive.
    """
    parsed_specs = [_parse_symbol_spec(spec) for spec in symbol_specs]
    parsed_specs = [spec for spec in parsed_specs if spec[0]]
    if not parsed_specs:
        return {}

    parse_path = file_path
    cleanup_path = None
    code_for_parse = text

    if text is not None:
        suffix = os.path.splitext(file_path)[1]
        fd, tmp_path = tempfile.mkstemp(suffix=suffix or "")
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(text)
        parse_path = tmp_path
        cleanup_path = tmp_path
    elif not os.path.exists(parse_path):
        return {}

    class _SymbolIO:
        def __init__(self, override_text: str | None = None):
            self.override_text = override_text

        def read_text(self, fname):
            if self.override_text is not None and fname == parse_path:
                return self.override_text
            with open(fname, "r", encoding="utf-8") as f:
                return f.read()

        def tool_output(self, *_args, **_kwargs):
            return None

        def tool_error(self, *_args, **_kwargs):
            return None

        def tool_warning(self, *_args, **_kwargs):
            return None

    class _ZeroModel:
        def token_count(self, _text):
            return 0

    io = _SymbolIO(code_for_parse)
    rm = RepoMap(
        map_tokens=0,
        main_model=_ZeroModel(),
        io=io,
        root=os.path.dirname(parse_path) or ".",
    )

    try:
        tags = rm.get_tags(parse_path, os.path.basename(parse_path)) or []
    except Exception:
        tags = []

    try:
        code = io.read_text(parse_path)
        ctx = TreeContext(
            parse_path,
            code,
            color=False,
            line_number=False,
            parent_context=True,
            child_context=True,
            last_line=True,
            margin=0,
            mark_lois=False,
            header_max=10,
            show_top_of_file_parent_scope=True,
            loi_pad=1,
        )
    except Exception:
        ctx = None

    results: dict[str, tuple[int, int]] = {}
    defs = [t for t in tags if getattr(t, "kind", None) == "def"]

    for raw, name, line_hint in parsed_specs:
        candidates = [t for t in defs if t.name == name or t.name == raw]
        if not candidates and "." in raw:
            suffix = raw.split(".")[-1]
            candidates = [t for t in defs if t.name == suffix]
        if not candidates:
            continue

        if line_hint is not None:
            candidates.sort(key=lambda t: abs((t.line + 1) - line_hint))

        tag = candidates[0]
        end_line = tag.line
        if ctx:
            try:
                end_line = ctx.get_last_line_of_scope(tag.line)
            except Exception:
                end_line = tag.line

        start_line = tag.line + 1
        results[raw] = (start_line, end_line + 1)

    if cleanup_path and os.path.exists(cleanup_path):
        try:
            os.remove(cleanup_path)
        except OSError:
            pass

    return results


def _is_ignored(path: str, patterns: list[str]) -> bool:
    from pathspec import PathSpec

    spec = PathSpec.from_lines("gitwildmatch", patterns)
    return spec.match_file(path)


def _collect_files(paths: list[str], ignore_paths: list[str] | None) -> list[str]:
    patterns = [".gitignore", "__pycache__/", "__init__.py"]
    if ignore_paths:
        for p in ignore_paths:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as fh:
                    patterns.extend(fh.read().splitlines())

    files: list[str] = []
    for path in paths:
        if os.path.isfile(path):
            if not _is_ignored(path, patterns) and _is_utf8_file(path):
                files.append(path)
        elif os.path.isdir(path):
            for root, dirs, names in os.walk(path):
                dirs[:] = [
                    d for d in dirs if not _is_ignored(os.path.join(root, d), patterns)
                ]
                for name in names:
                    fp = os.path.join(root, name)
                    if not _is_ignored(fp, patterns) and _is_utf8_file(fp):
                        files.append(fp)
    return files


def _is_section_header(line: str) -> bool:
    """
    Detect repo map section headers (file lines) versus code/context lines.
    """
    if not line or not line.strip():
        return False
    if line[0].isspace():
        return False
    return not line.startswith("│") and not line.startswith("⋮")


def _format_label_with_tokens(label: str, token_count: int) -> str:
    suffix = "token" if token_count == 1 else "tokens"
    trimmed = label.rstrip()
    has_colon = trimmed.endswith(":")
    base = trimmed[:-1] if has_colon else trimmed
    return f"{base} ({token_count} {suffix}){':' if has_colon else ''}"


def _annotate_repo_map(
    repo_map: str,
    token_lookup: dict[str, int],
    token_target: str,
    symbol_lookup: dict[str, list[dict]] | None = None,
) -> str:
    """
    Insert token counts into each section header of the repo map output.
    Prefers full-file counts from token_lookup; falls back to snippet counts.
    Optionally inserts per-symbol token breakdowns when available.
    """
    if not repo_map:
        return repo_map

    preamble = ""
    body = repo_map
    trailing_newline = body.endswith("\n")
    if body.startswith("❯") and "\n" in body:
        preamble, body = body.split("\n", 1)

    sections = []
    current_header = None
    current_lines: list[str] = []
    for line in body.splitlines():
        if _is_section_header(line):
            if current_header is not None:
                sections.append((current_header, current_lines))
            current_header = line
            current_lines = []
        else:
            if current_header is None:
                continue
            current_lines.append(line)
    if current_header is not None:
        sections.append((current_header, current_lines))

    if not sections:
        return repo_map

    annotated_lines: list[str] = []
    for idx, (header, content_lines) in enumerate(sections):
        if idx > 0:
            annotated_lines.append("")
        content_text = "\n".join(content_lines)
        key = header.rstrip()
        if key.endswith(":"):
            key = key[:-1]
        tokens = token_lookup.get(key)
        if tokens is None:
            tokens = (
                count_tokens(content_text, target=token_target)["count"]
                if content_lines
                else 0
            )
        annotated_lines.append(_format_label_with_tokens(header, tokens))

        if symbol_lookup:
            symbols = symbol_lookup.get(key, [])
        else:
            symbols = []

        # insert symbol token markers directly above matching lines
        remaining_symbols = list(symbols)
        for line in content_lines:
            to_insert = []
            stripped = line.lstrip("│ ").rstrip()
            i = 0
            while i < len(remaining_symbols):
                sym = remaining_symbols[i]
                code_line = sym.get("code_line", "")
                if code_line and stripped.startswith(code_line):
                    to_insert.append(sym)
                    remaining_symbols.pop(i)
                else:
                    i += 1
            if to_insert:
                for sym in to_insert:
                    count = sym.get("tokens", 0)
                    suffix = "token" if count == 1 else "tokens"
                    code_preview = sym.get("code_line") or sym.get("name", "")
                    annotated_lines.append(f"├── [{count} {suffix}] {code_preview}")
                # skip original line since it is already included in the marker
                continue
            annotated_lines.append(line)

        # any unmatched symbols just get appended at the end of the section
        for sym in remaining_symbols:
            count = sym.get("tokens", 0)
            suffix = "token" if count == 1 else "tokens"
            code_preview = sym.get("code_line") or sym.get("name", "")
            annotated_lines.append(f"├── [{count} {suffix}] {code_preview}")

    annotated_body = "\n".join(annotated_lines)
    if trailing_newline:
        annotated_body += "\n"
    if preamble:
        return preamble + "\n" + annotated_body
    return annotated_body


class ContextualRepoMap(RepoMap):
    """
    RepoMap wrapper that records the tag subset used for the final map so we can
    compute per-symbol token counts.
    """

    def get_ranked_tags_map_uncached(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        spin = Spinner(UPDATING_REPO_MAP_MESSAGE)

        ranked_tags = self.get_ranked_tags(
            chat_fnames,
            other_fnames,
            mentioned_fnames,
            mentioned_idents,
            progress=spin.step,
        )

        other_rel_fnames = sorted(
            set(self.get_rel_fname(fname) for fname in other_fnames)
        )
        special_fnames = filter_important_files(other_rel_fnames)
        ranked_tags_fnames = set(tag[0] for tag in ranked_tags)
        special_fnames = [fn for fn in special_fnames if fn not in ranked_tags_fnames]
        special_fnames = [(fn,) for fn in special_fnames]

        ranked_tags = special_fnames + ranked_tags

        spin.step()

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0
        best_tags_subset = None

        chat_rel_fnames = set(self.get_rel_fname(fname) for fname in chat_fnames)

        self.tree_cache = dict()

        middle = min(int(max_map_tokens // 25), num_tags)
        while lower_bound <= upper_bound:
            if middle > 1500:
                show_tokens = f"{middle / 1000.0:.1f}K"
            else:
                show_tokens = str(middle)
            spin.step(f"{UPDATING_REPO_MAP_MESSAGE}: {show_tokens} tokens")

            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)

            pct_err = abs(num_tokens - max_map_tokens) / max_map_tokens
            ok_err = 0.15
            if (
                num_tokens <= max_map_tokens and num_tokens > best_tree_tokens
            ) or pct_err < ok_err:
                best_tree = tree
                best_tree_tokens = num_tokens
                best_tags_subset = ranked_tags[:middle]

                if pct_err < ok_err:
                    break

            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = int((lower_bound + upper_bound) // 2)

        spin.end()
        self.best_ranked_tags = best_tags_subset or []
        return best_tree

    def _symbol_context(self, rel_fname: str, code: str) -> TreeContext | None:
        if not code:
            return None
        if not hasattr(self, "_symbol_context_cache"):
            self._symbol_context_cache = {}

        cached = self._symbol_context_cache.get(rel_fname)
        if cached and cached.get("code") == code:
            return cached["context"]

        try:
            ctx = TreeContext(
                rel_fname,
                code,
                color=False,
                line_number=False,
                parent_context=True,
                child_context=True,
                last_line=True,
                margin=0,
                mark_lois=False,
                header_max=10,
                show_top_of_file_parent_scope=True,
                loi_pad=1,
            )
        except Exception:
            return None

        self._symbol_context_cache[rel_fname] = {"context": ctx, "code": code}
        return ctx

    def symbol_token_count(
        self, rel_fname: str, code: str, start_line: int, token_target: str
    ) -> int:
        ctx = self._symbol_context(rel_fname, code)
        if not ctx:
            return 0

        try:
            end_line = ctx.get_last_line_of_scope(start_line)
        except Exception:
            end_line = start_line

        lines = code.splitlines()
        if start_line >= len(lines):
            return 0
        end_line = min(end_line, len(lines) - 1)
        symbol_text = "\n".join(lines[start_line : end_line + 1])
        return count_tokens(symbol_text, target=token_target)["count"]


def generate_repo_map_data(
    paths,
    max_tokens,
    fmt,
    ignore=None,
    annotate_tokens=False,
    token_target="cl100k_base",
):
    """
    Generate a repository map and return a dict containing:
      - repo_map: The generated repository map as a string.
      - summary: A summary string with file/token info.
      - messages: Any warnings/errors collected.
      - error: Present if no map could be generated.
    """
    files = _collect_files(paths, ignore)

    class CollectorIO:
        def __init__(self):
            self.messages = []

        def tool_output(self, msg):
            self.messages.append(msg)

        def tool_error(self, msg):
            self.messages.append(f"ERROR: {msg}")

        def tool_warning(self, msg):
            self.messages.append(f"WARNING: {msg}")

        def read_text(self, fname):
            try:
                with open(fname, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                self.tool_warning(f"Error reading file {fname}: {str(e)}")
                return ""

    class TokenCounter:
        def token_count(self, text):
            result = count_tokens(text, target=token_target)
            return result["count"]

    io = CollectorIO()
    token_counter = TokenCounter()

    root = os.getcwd()

    rm = ContextualRepoMap(
        map_tokens=max_tokens, main_model=token_counter, io=io, root=root
    )
    repo_map = rm.get_repo_map(chat_files=[], other_files=files)

    if not repo_map:
        error_message = "\n".join(io.messages) or "No repository map was generated."
        return {"error": error_message}

    if fmt == "shell":
        repo_map = f"❯ repo-map {' '.join(paths)}\n{repo_map}"

    if annotate_tokens:
        token_lookup = {}
        symbol_lookup: dict[str, list[dict]] = {}
        file_texts: dict[str, str] = {}

        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
            rel = os.path.relpath(path, root)
            token_lookup[rel] = count_tokens(text, target=token_target)["count"]
            file_texts[rel] = text

        best_tags = getattr(rm, "best_ranked_tags", None) or []
        tags_by_file: dict[str, list[Tag]] = {}
        for t in best_tags:
            if isinstance(t, Tag):
                tags_by_file.setdefault(t.rel_fname, []).append(t)

        for rel_path, tags in tags_by_file.items():
            seen = set()
            entries = []
            for t in tags:
                if getattr(t, "kind", None) and t.kind != "def":
                    continue
                if t.line in seen:
                    continue
                seen.add(t.line)
                file_text = file_texts.get(rel_path, "")
                tokens = rm.symbol_token_count(
                    rel_path, file_text, t.line, token_target
                )
                try:
                    source_line = file_text.splitlines()[t.line]
                except Exception:
                    source_line = ""
                code_line = source_line[:100].strip()
                entries.append(
                    {
                        "name": t.name,
                        "line": t.line + 1,
                        "tokens": tokens,
                        "code_line": code_line,
                    }
                )
            if entries:
                symbol_lookup[rel_path] = entries

        repo_map = _annotate_repo_map(
            repo_map, token_lookup, token_target, symbol_lookup=symbol_lookup
        )

    token_info = count_tokens(repo_map, target=token_target)
    num_files = len(files)
    summary_str = f"Map of {num_files} files ({token_info['count']} tokens)"

    return {
        "repo_map": repo_map,
        "summary": summary_str,
        "messages": io.messages,
    }


def generate_repo_map_data_from_git(
    repo_root,
    path_specs,
    rev,
    max_tokens,
    fmt,
    ignore=None,
    annotate_tokens=False,
    token_target="cl100k_base",
):
    patterns = [".gitignore", "__pycache__/", "__init__.py"]
    if ignore:
        for p in ignore:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as fh:
                    patterns.extend(fh.read().splitlines())

    from pathspec import PathSpec

    spec = PathSpec.from_lines("gitwildmatch", patterns)
    rel_files = list_files_at_rev(repo_root, rev, path_specs)
    filtered = []
    for rel in rel_files:
        if spec.match_file(rel):
            continue
        if read_file_at_rev(repo_root, rev, rel) is not None:
            filtered.append(rel)

    class IO:
        def __init__(self):
            self.messages = []

        def tool_output(self, msg):
            self.messages.append(msg)

        def tool_error(self, msg):
            self.messages.append(f"ERROR: {msg}")

        def tool_warning(self, msg):
            self.messages.append(f"WARNING: {msg}")

        def read_text(self, fname):
            txt = read_file_at_rev(repo_root, rev, fname)
            if txt is None:
                self.tool_warning(f"Skipping binary or unreadable file: {fname}")
                return ""
            return txt

    class Counter:
        def token_count(self, text):
            return count_tokens(text, target=token_target)["count"]

    io = IO()
    rm = ContextualRepoMap(
        map_tokens=max_tokens, main_model=Counter(), io=io, root=repo_root
    )
    repo_map = rm.get_repo_map(chat_files=[], other_files=filtered)
    if not repo_map:
        return {"error": "\n".join(io.messages) or "No repository map was generated."}
    if fmt == "shell":
        repo_map = f"❯ repo-map {' '.join(path_specs)}\n{repo_map}"
    if annotate_tokens:
        token_lookup: dict[str, int] = {}
        symbol_lookup: dict[str, list[dict]] = {}
        file_texts: dict[str, str] = {}

        for rel in filtered:
            text = read_file_at_rev(repo_root, rev, rel)
            if text is None:
                continue
            token_lookup[rel] = count_tokens(text, target=token_target)["count"]
            file_texts[rel] = text

        best_tags = getattr(rm, "best_ranked_tags", None) or []
        tags_by_file: dict[str, list[Tag]] = {}
        for t in best_tags:
            if isinstance(t, Tag):
                tags_by_file.setdefault(t.rel_fname, []).append(t)

        for rel_path, tags in tags_by_file.items():
            seen = set()
            entries = []
            for t in tags:
                if getattr(t, "kind", None) and t.kind != "def":
                    continue
                if t.line in seen:
                    continue
                seen.add(t.line)
                file_text = file_texts.get(rel_path, "")
                tokens = rm.symbol_token_count(
                    rel_path, file_text, t.line, token_target
                )
                try:
                    source_line = file_text.splitlines()[t.line]
                except Exception:
                    source_line = ""
                code_line = source_line[:100].strip()
                entries.append(
                    {
                        "name": t.name,
                        "line": t.line + 1,
                        "tokens": tokens,
                        "code_line": code_line,
                    }
                )
            if entries:
                symbol_lookup[rel_path] = entries

        repo_map = _annotate_repo_map(
            repo_map, token_lookup, token_target, symbol_lookup=symbol_lookup
        )
    token_info = count_tokens(repo_map, target=token_target)
    return {
        "repo_map": repo_map,
        "summary": f"Map of {len(filtered)} files ({token_info['count']} tokens)",
        "messages": io.messages,
    }
