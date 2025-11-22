import os

from contextualize.tokenize import count_tokens
from .reference import _is_utf8_file
from .gitrev import list_files_at_rev, read_file_at_rev


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
    repo_map: str, token_lookup: dict[str, int], token_target: str
) -> str:
    """
    Insert token counts into each section header of the repo map output.
    Prefers full-file counts from token_lookup; falls back to snippet counts.
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
        annotated_lines.extend(content_lines)

    annotated_body = "\n".join(annotated_lines)
    if trailing_newline:
        annotated_body += "\n"
    if preamble:
        return preamble + "\n" + annotated_body
    return annotated_body


def generate_repo_map_data(
    paths, max_tokens, fmt, ignore=None, annotate_tokens=False, token_target="cl100k_base"
):
    """
    Generate a repository map and return a dict containing:
      - repo_map: The generated repository map as a string.
      - summary: A summary string with file/token info.
      - messages: Any warnings/errors collected.
      - error: Present if no map could be generated.
    """
    from aider.repomap import RepoMap

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

    rm = RepoMap(map_tokens=max_tokens, main_model=token_counter, io=io, root=root)
    repo_map = rm.get_repo_map(chat_files=[], other_files=files)

    if not repo_map:
        error_message = "\n".join(io.messages) or "No repository map was generated."
        return {"error": error_message}

    if fmt == "shell":
        repo_map = f"❯ repo-map {' '.join(paths)}\n{repo_map}"

    if annotate_tokens:
        token_lookup = {}
        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
            rel = os.path.relpath(path, root)
            token_lookup[rel] = count_tokens(text, target=token_target)["count"]

        repo_map = _annotate_repo_map(repo_map, token_lookup, token_target)

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

    from aider.repomap import RepoMap

    io = IO()
    rm = RepoMap(map_tokens=max_tokens, main_model=Counter(), io=io, root=repo_root)
    repo_map = rm.get_repo_map(chat_files=[], other_files=filtered)
    if not repo_map:
        return {"error": "\n".join(io.messages) or "No repository map was generated."}
    if fmt == "shell":
        repo_map = f"❯ repo-map {' '.join(path_specs)}\n{repo_map}"
    if annotate_tokens:
        token_lookup = {}
        for rel in filtered:
            text = read_file_at_rev(repo_root, rev, rel)
            if text is None:
                continue
            token_lookup[rel] = count_tokens(text, target=token_target)["count"]
        repo_map = _annotate_repo_map(repo_map, token_lookup, token_target)
    token_info = count_tokens(repo_map, target=token_target)
    return {
        "repo_map": repo_map,
        "summary": f"Map of {len(filtered)} files ({token_info['count']} tokens)",
        "messages": io.messages,
    }
