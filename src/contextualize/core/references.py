import codecs
import os
import sys
import re
import subprocess
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse

from .render import process_text
from .utils import brace_expand, count_tokens


def split_path_and_symbols(raw_path: str) -> tuple[str, list[str]]:
    """
    Split a path of the form "path::sym1,sym2" into the base path and symbol list.
    """
    if "::" not in raw_path:
        return raw_path, []
    base, _, suffix = raw_path.partition("::")
    symbols = [part.strip() for part in suffix.split(",") if part.strip()]
    return base or raw_path, symbols


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

    for raw_path in paths:
        path, symbols = split_path_and_symbols(raw_path)

        if raw_path.startswith("http://") or raw_path.startswith("https://"):
            file_references.append(
                URLReference(
                    raw_path,
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
                ranges = None
                if symbols:
                    try:
                        from .repomap import find_symbol_ranges

                        match_map = find_symbol_ranges(path, symbols)
                    except Exception:
                        match_map = {}

                    matched = [s for s in symbols if s in match_map]
                    if not matched:
                        print(
                            f"Warning: symbol(s) not found in {path}: {', '.join(symbols)}",
                            file=sys.stderr,
                        )
                        continue
                    symbols = matched
                    ranges = [match_map[s] for s in matched]

                file_references.append(
                    FileReference(
                        path,
                        ranges=ranges,
                        symbols=symbols,
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
        ranges=None,
        format="md",
        label="relative",
        clean_contents=False,
        *,
        include_token_count=False,
        token_target="cl100k_base",
        inject=False,
        depth=5,
        trace_collector=None,
        symbols=None,
    ):
        self.range = range
        self.ranges = ranges
        if self.range and not self.ranges:
            self.ranges = [self.range]
        self.symbols = [s for s in (symbols or []) if s]

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
            from .links import inject_content_in_text

            self.file_content = inject_content_in_text(
                self.file_content, self.depth, self.trace_collector, self.path
            )

        ranges = self.ranges
        if self.symbols and ranges is None:
            try:
                from .repomap import find_symbol_ranges

                match_map = find_symbol_ranges(
                    self.path, self.symbols, text=self.file_content
                )
            except Exception:
                match_map = {}

            missing = [s for s in self.symbols if s not in match_map]
            if missing:
                print(
                    f"Warning: symbol(s) not found in {self.path}: {', '.join(missing)}",
                    file=sys.stderr,
                )
            if match_map:
                matched = [s for s in self.symbols if s in match_map]
                ranges = [match_map[s] for s in matched]
                self.symbols = matched

        return process_text(
            self.file_content,
            self.clean_contents,
            ranges=ranges,
            format=self.format,
            label=self.get_label(),
            token_target=self.token_target,
            include_token_count=self.include_token_count,
            symbols=self.symbols,
        )

    def get_label(self):
        if self.label == "relative":
            from ..git.cache import CACHE_ROOT

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
            from .links import inject_content_in_text

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


class CommandReference:
    """
    A wrapper that runs a command, captures its output, and formats it
    with the same approach used by FileReference.
    """

    def __init__(
        self,
        command: str,
        format: str = "shell",
        capture_stderr: bool = True,
    ):
        """
        :param command: The raw command string, e.g. "ls --help"
        :param format: "md"/"xml"/"shell"
        :param capture_stderr: Whether to capture stderr as well.
        """
        self.command = command
        self.format = format
        self.capture_stderr = capture_stderr

        self.command_output = self.run_command()
        self.output = self.get_contents()

    def run_command(self) -> str:
        """
        Execute the command and return combined stdout/stderr.
        Captures stdout (and stderr if enabled), and returns a single string with ANSI
        escape sequences removed.
        """
        try:
            result = subprocess.run(
                self.command,
                shell=True,  # allows pipes, redirection, etc.
                capture_output=True,
                text=True,
            )
            stdout = result.stdout
            stderr = result.stderr if self.capture_stderr else ""
            combined = stdout + ("\n" + stderr if stderr else "")
            return remove_ansi(combined)
        except Exception as e:
            return f"Error running command {self.command}: {str(e)}\n"

    def get_contents(self) -> str:
        if self.format == "xml":
            return f'<cmd exec="{self.command}">\n{self.command_output}\n</cmd>'
        else:
            return process_text(
                text=self.command_output,
                clean=False,
                range=None,
                format=self.format,
                label=self.command,
                shell_cmd=self.command if self.format == "shell" else None,
            )


def remove_ansi(text: str) -> str:
    """
    Remove ANSI escape sequences from text.
    """
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def create_command_references(
    commands: List[str],
    format: str = "shell",
    capture_stderr: bool = True,
):
    """
    Runs each command, collects outputs as CommandReference objects,
    and concatenates them similarly to how file references are handled.
    """
    cmd_refs = []
    for cmd in commands:
        cmd_ref = CommandReference(cmd, format=format, capture_stderr=capture_stderr)
        cmd_refs.append(cmd_ref)

    concatenated = "\n\n".join(ref.output for ref in cmd_refs)
    return {
        "refs": cmd_refs,
        "concatenated": concatenated,
    }
