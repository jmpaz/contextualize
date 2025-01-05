import os

from pathspec import PathSpec


def create_file_references(paths, ignore_paths=None, format="md", label="relative"):
    """FileReference wrapper for creating a list of file references from paths."""
    file_references = []
    ignore_patterns = [
        ".gitignore",
        "__pycache__/",
        "__init__.py",
    ]

    if ignore_paths:
        for path in ignore_paths:
            if os.path.isfile(path):
                with open(path, "r") as file:
                    ignore_patterns.extend(file.read().splitlines())

    def is_ignored(path, gitignore_patterns):
        path_spec = PathSpec.from_lines("gitwildmatch", gitignore_patterns)
        return path_spec.match_file(path)

    for path in paths:
        if os.path.isfile(path):
            if not is_ignored(path, ignore_patterns):
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
                    if not is_ignored(file_path, ignore_patterns):
                        file_references.append(
                            FileReference(file_path, format=format, label=label)
                        )

    return {"refs": file_references, "concatenated": concat_refs(file_references)}


def concat_refs(file_references: list):
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
            with open(self.path, "r") as file:
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
            return self.path
        elif self.label == "name":
            return os.path.basename(self.path)
        elif self.label == "ext":
            return os.path.splitext(self.path)[1]
        else:
            return ""


def _clean(text):
    return text.replace("    ", "\t")


def _extract_range(text, range):
    """Extracts lines from contents based on range tuple."""
    start, end = range
    lines = text.split("\n")
    return "\n".join(lines[start - 1 : end])


def _count_max_backticks(text):
    max_backticks = 0
    lines = text.split("\n")
    for line in lines:
        if line.startswith("`"):
            max_backticks = max(max_backticks, len(line) - len(line.lstrip("`")))
    return max_backticks


def _delimit(text, format, label, max_backticks=0):
    if format == "md":
        backticks_str = "`" * (max_backticks + 2) if max_backticks >= 3 else "```"
        return f"{backticks_str}{label}\n{text}\n{backticks_str}"
    elif format == "xml":
        return f"<file path='{label}'>\n{text}\n</file>"
    elif format == "shell":
        return f"❯ cat {label}\n{text}"
    else:
        return text


def process_text(text, clean=False, range=None, format="md", label=""):
    if clean:
        text = _clean(text)
    if range:
        text = _extract_range(text, range)
    max_backticks = _count_max_backticks(text)
    contents = _delimit(text, format, label, max_backticks)
    return contents
