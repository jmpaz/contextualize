import os


def _is_utf8_file(path: str, sample_size: int = 4096) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_size)
        chunk.decode("utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


def create_file_references(paths, ignore_paths=None, format="md", label="relative"):
    """
    Build a list of file references from the specified paths.
    """

    def is_ignored(path, gitignore_patterns):
        # We'll import pathspec only if needed:
        from pathspec import PathSpec

        path_spec = PathSpec.from_lines("gitwildmatch", gitignore_patterns)
        return path_spec.match_file(path)

    file_references = []
    ignore_patterns = [
        ".gitignore",
        ".git/",
        "__pycache__/",
        "__init__.py",
    ]

    # If user supplied paths to ignore (like .gitignore, etc.), read them
    if ignore_paths:
        for path in ignore_paths:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as file:
                    ignore_patterns.extend(file.read().splitlines())

    for path in paths:
        if os.path.isfile(path):
            if not is_ignored(path, ignore_patterns) and _is_utf8_file(path):
                file_references.append(FileReference(path, format=format, label=label))
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs[:] = [
                    d
                    for d in dirs
                    if not is_ignored(os.path.join(root, d), ignore_patterns)
                ]
                for file in files:
                    file_path = os.path.join(root, file)
                    if not is_ignored(file_path, ignore_patterns) and _is_utf8_file(
                        file_path
                    ):
                        file_references.append(
                            FileReference(file_path, format=format, label=label)
                        )

    return {
        "refs": file_references,
        "concatenated": concat_refs(file_references),
    }


def concat_refs(file_references):
    """
    Concatenate references into a single string with the chosen format.
    """
    return "\n\n".join(ref.output for ref in file_references)


class FileReference:
    def __init__(
        self, path, range=None, format="md", label="relative", clean_contents=False
    ):
        self.range = range
        self.path = path
        self.format = format
        self.label = label
        self.clean_contents = clean_contents
        self.file_content = ""
        self.output = self.get_contents()

    def get_contents(self):
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                self.file_content = file.read()
        except Exception as e:
            print(f"Error reading file {self.path}: {str(e)}")
            return ""

        return process_text(
            self.file_content,
            self.clean_contents,
            self.range,
            self.format,
            self.get_label(),
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
            return ""


def process_text(text, clean=False, range=None, format="md", label="", shell_cmd=None):
    if clean:
        text = _clean(text)
    if range:
        text = _extract_range(text, range)
    max_backticks = _count_max_backticks(text)
    return _delimit(text, format, label, max_backticks, shell_cmd)


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


def _delimit(text, format, label, max_backticks=0, shell_cmd=None):
    if format == "md":
        backticks_str = "`" * max(max_backticks + 2, 3)  # at least 3
        return f"{backticks_str}{label}\n{text}\n{backticks_str}"
    elif format == "xml":
        return f"<file path='{label}'>\n{text}\n</file>"
    elif format == "shell":
        if shell_cmd:
            return f"❯ {shell_cmd}\n{text}"
        else:
            return f"❯ cat {label}\n{text}"
    else:
        return text
