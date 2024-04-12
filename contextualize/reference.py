import os


class FileReference:
    def __init__(
        self, path, range=None, format="md", label="relative", clean_contents=False
    ):
        self.range = range
        self.path = path
        self.format = format
        self.label = label
        self.clean_contents = clean_contents
        self.output = self.get_contents()

    def get_contents(self):
        try:
            with open(self.path, "r") as file:
                contents = file.read()
        except Exception as e:
            print(f"Error reading file {self.path}: {str(e)}")
            return ""

        return process_text(
            contents, self.clean_contents, self.range, self.format, self.get_label()
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


def concat_refs(file_references: list):
    return "\n\n".join(ref.output for ref in file_references)


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
