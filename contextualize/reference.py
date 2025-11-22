import codecs
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from .utils import brace_expand


def _is_utf8_file(path: str, sample_size: int = 4096) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_size)
    except OSError:
        return False

    if not chunk:
        return True

    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        decoder.decode(chunk, final=False)
    except UnicodeDecodeError:
        return False

    return True


def create_file_references(
    paths,
    ignore_patterns=None,
    format="md",
    label="relative",
    include_token_count=False,
    token_target="cl100k_base",
    inject=False,
    depth=5,
    trace_collector=None,
):
    """
    Build a list of file references from the specified paths.
    if `inject` is true, {cx::...} markers are resolved before wrapping.
    """
    from .tokenize import count_tokens

    def is_ignored(path, gitignore_patterns):
        # We'll import pathspec only if needed:
        from pathspec import PathSpec

        path_spec = PathSpec.from_lines("gitwildmatch", gitignore_patterns)
        return path_spec.match_file(path)

    def get_file_token_count(file_path):
        """Get token count for a file if it's readable."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return count_tokens(content, target=token_target)["count"]
        except:
            return 0

    file_references = []
    ignored_files = []
    ignored_folders = {}
    dirs_with_non_ignored_files = set()
    default_ignore_patterns = [
        ".gitignore",
        ".git/",
        "__pycache__/",
        "__init__.py",
    ]

    all_ignore_patterns = default_ignore_patterns[:]
    if ignore_patterns:
        expanded_ignore_patterns = []
        for pattern in ignore_patterns:
            if "{" in pattern and "}" in pattern:
                expanded_ignore_patterns.extend(brace_expand(pattern))
            else:
                expanded_ignore_patterns.append(pattern)
        all_ignore_patterns.extend(expanded_ignore_patterns)

    expanded_user_patterns = []
    if ignore_patterns:
        for pattern in ignore_patterns:
            if "{" in pattern and "}" in pattern:
                expanded_user_patterns.extend(brace_expand(pattern))
            else:
                expanded_user_patterns.append(pattern)

    for path in paths:
        if path.startswith("http://") or path.startswith("https://"):
            file_references.append(
                URLReference(
                    path,
                    format=format,
                    label=label,
                    include_token_count=include_token_count,
                    token_target=token_target,
                    inject=inject,
                    depth=depth,
                    trace_collector=trace_collector,
                )
            )
        elif os.path.isfile(path):
            if is_ignored(path, all_ignore_patterns):
                if (
                    expanded_user_patterns
                    and is_ignored(path, expanded_user_patterns)
                    and _is_utf8_file(path)
                ):
                    token_count = get_file_token_count(path)
                    ignored_files.append((path, token_count))
            elif _is_utf8_file(path):
                file_references.append(
                    FileReference(
                        path,
                        format=format,
                        label=label,
                        include_token_count=include_token_count,
                        token_target=token_target,
                        inject=inject,
                        depth=depth,
                        trace_collector=trace_collector,
                    )
                )
        elif os.path.isdir(path):
            dir_ignored_files = {}
            for root, dirs, files in os.walk(path):
                dirs[:] = [
                    d
                    for d in dirs
                    if not is_ignored(os.path.join(root, d), all_ignore_patterns)
                ]
                for file in files:
                    file_path = os.path.join(root, file)
                    if is_ignored(file_path, all_ignore_patterns):
                        if (
                            expanded_user_patterns
                            and is_ignored(file_path, expanded_user_patterns)
                            and _is_utf8_file(file_path)
                        ):
                            token_count = get_file_token_count(file_path)
                            if root not in dir_ignored_files:
                                dir_ignored_files[root] = []
                            dir_ignored_files[root].append((file_path, token_count))
                    elif _is_utf8_file(file_path):
                        dirs_with_non_ignored_files.add(root)
                        parent = os.path.dirname(root)
                        while (
                            parent
                            and parent != path
                            and parent not in dirs_with_non_ignored_files
                        ):
                            dirs_with_non_ignored_files.add(parent)
                            new_parent = os.path.dirname(parent)
                            if new_parent == parent:
                                break
                            parent = new_parent
                        file_references.append(
                            FileReference(
                                file_path,
                                format=format,
                                label=label,
                                include_token_count=include_token_count,
                                token_target=token_target,
                                inject=inject,
                                depth=depth,
                                trace_collector=trace_collector,
                            )
                        )

            for dir_path, files_list in dir_ignored_files.items():
                if dir_path not in dirs_with_non_ignored_files:
                    total_tokens = sum(tokens for _, tokens in files_list)
                    ignored_folders[dir_path] = (len(files_list), total_tokens)
                else:
                    ignored_files.extend(files_list)

    consolidated_folders = {}
    for folder_path in sorted(ignored_folders.keys()):
        parent_is_ignored = False
        parent = os.path.dirname(folder_path)
        while parent:
            if parent in ignored_folders:
                parent_is_ignored = True
                break
            new_parent = os.path.dirname(parent)
            if new_parent == parent:
                break
            parent = new_parent
        if not parent_is_ignored:
            consolidated_folders[folder_path] = ignored_folders[folder_path]

    return {
        "refs": file_references,
        "concatenated": concat_refs(file_references),
        "ignored_files": ignored_files,
        "ignored_folders": consolidated_folders,
    }


def concat_refs(file_references):
    """
    Concatenate references into a single string with the chosen format.
    """
    return "\n\n".join(ref.output for ref in file_references)


class FileReference:
    def __init__(
        self,
        path,
        range=None,
        format="md",
        label="relative",
        clean_contents=False,
        *,
        include_token_count=False,
        token_target="cl100k_base",
        inject=False,
        depth=5,
        trace_collector=None,
    ):
        self.range = range
        self.path = path
        self.format = format
        self.label = label
        self.clean_contents = clean_contents
        self.include_token_count = include_token_count
        self.token_target = token_target
        self.inject = inject
        self.depth = depth
        self.trace_collector = trace_collector
        self.file_content = self.original_file_content = ""
        self.output = self.get_contents()

    def get_contents(self):
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                self.file_content = self.original_file_content = file.read()
        except Exception as e:
            print(f"Error reading file {self.path}: {str(e)}")
            return ""
        if self.inject:
            from .injection import inject_content_in_text

            self.file_content = inject_content_in_text(
                self.file_content, self.depth, self.trace_collector, self.path
            )

        return process_text(
            self.file_content,
            self.clean_contents,
            self.range,
            self.format,
            self.get_label(),
            token_target=self.token_target,
            include_token_count=self.include_token_count,
        )

    def get_label(self):
        if self.label == "relative":
            from .gitcache import CACHE_ROOT

            cache_root = os.path.join(CACHE_ROOT, "")
            if self.path.startswith(cache_root):
                rel = os.path.relpath(self.path, CACHE_ROOT)
                parts = rel.split(os.sep)
                if parts and parts[0] in ("github", "ext"):
                    return os.path.join(*parts[1:])
                return rel
            return self.path
        elif self.label == "name":
            return os.path.basename(self.path)
        elif self.label == "ext":
            return os.path.splitext(self.path)[1]
        else:
            return self.label


@dataclass
class URLReference:
    url: str
    format: str = "md"
    label: str = "relative"
    token_target: str = "cl100k_base"
    include_token_count: bool = False
    inject: bool = False
    depth: int = 5
    trace_collector: list = None

    def __post_init__(self) -> None:
        self.file_content = ""
        self.original_file_content = ""
        self.output = self.get_contents()

    def get_label(self) -> str:
        path = urlparse(self.url).path
        if self.label == "relative":
            return self.url
        if self.label == "name":
            return os.path.basename(path)
        if self.label == "ext":
            return os.path.splitext(path)[1]
        return self.label

    @property
    def path(self) -> str:
        return self.url

    def get_contents(self) -> str:
        import json

        import requests

        r = requests.get(self.url, timeout=30, headers={"User-Agent": "contextualize"})
        r.raise_for_status()
        text = r.text
        self.original_file_content = text
        if "json" in r.headers.get("Content-Type", ""):
            try:
                text = json.dumps(r.json(), indent=2)
            except Exception:
                pass
        if self.inject:
            from .injection import inject_content_in_text

            text = inject_content_in_text(
                text, self.depth, self.trace_collector, self.url
            )
        self.file_content = text
        return process_text(
            text,
            format=self.format,
            label=self.get_label(),
            token_target=self.token_target,
            include_token_count=self.include_token_count,
        )


def process_text(
    text,
    clean=False,
    range=None,
    format="md",
    label="",
    shell_cmd=None,
    rev: str | None = None,
    token_target: str = "cl100k_base",
    token_count: int | None = None,
    include_token_count: bool = False,
):
    if clean:
        text = _clean(text)
    if range:
        text = _extract_range(text, range)
    use_token_count = include_token_count and format in {"md", "xml", "shell"}

    if use_token_count:
        if token_count is None:
            from .tokenize import count_tokens

            token_count = count_tokens(text, target=token_target)["count"]
    else:
        token_count = None
    max_backticks = _count_max_backticks(text)
    return _delimit(
        text, format, label, max_backticks, shell_cmd, rev, token_count
    )


def _clean(text):
    # Example cleaning logic
    return text.replace("    ", "\t")


def _extract_range(text, range_tuple):
    start, end = range_tuple
    lines = text.split("\n")
    return "\n".join(lines[start - 1 : end])


def _count_max_backticks(text):
    max_backticks = 0
    for line in text.split("\n"):
        # If a line starts with backticks, count them
        stripped = line.lstrip("`")
        count = len(line) - len(stripped)
        if count > max_backticks:
            max_backticks = count
    return max_backticks


def _delimit(
    text,
    format,
    label,
    max_backticks=0,
    shell_cmd=None,
    rev: str | None = None,
    token_count: int | None = None,
):
    if format == "md":
        backticks_str = "`" * max(max_backticks + 2, 3)  # at least 3
        info = f"{label}@{rev}" if rev else label
        if token_count is not None:
            info = f"{info} ({token_count} tokens)"
        return f"{backticks_str}{info}\n{text}\n{backticks_str}"
    elif format == "xml":
        token_attr = f" token_count='{token_count}'" if token_count is not None else ""
        rev_attr = f" rev='{rev}'" if rev else ""
        return f"<file path='{label}'{token_attr}{rev_attr}>\n{text}\n</file>"
    elif format == "shell":
        if shell_cmd:
            token_suffix = f" ({token_count} tokens)" if token_count is not None else ""
            return f"❯ {shell_cmd}{token_suffix}\n{text}"
        else:
            info = f"{label}@{rev}" if rev else label
            token_suffix = f" ({token_count} tokens)" if token_count is not None else ""
            return f"❯ cat {info}{token_suffix}\n{text}"
    else:
        return text
