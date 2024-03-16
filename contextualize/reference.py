import os


class FileReference:
    def __init__(
        self,
        path: str,
        range: tuple = None,
        format="md",
        label="relative",
        clean_contents=False,
    ):
        self.range = range
        self.path = path
        self.format = format
        self.label = label
        self.clean_contents = clean_contents

        # prepare the reference string
        self.output = self.get_contents()

    def get_contents(self):
        try:
            with open(self.path, "r") as file:
                contents = file.read()
            contents = self.process(contents)
            return contents
        except UnicodeDecodeError:
            print(f"Skipping unreadable file: {self.path}")
            return ""
        except FileNotFoundError:
            print(f"File not found: {self.path}")
            return ""
        except Exception as e:
            print(f"Error occurred while reading file: {self.path}")
            print(f"Error details: {str(e)}")
            return ""

    def process(self, contents):
        if self.clean_contents:
            contents = self.clean(contents)
        if self.range:
            contents = self.extract_range(contents, self.range)
        if self.format == "md":
            max_backticks = self.count_max_backticks(contents)
            contents = self.delineate(
                contents, self.format, self.get_label(), max_backticks
            )
        else:
            contents = self.delineate(contents, self.format, self.get_label())
        return contents

    def extract_range(self, contents, range):
        start, end = range
        lines = contents.split("\n")
        return "\n".join(lines[start - 1 : end])

    def clean(self, contents):
        return contents.replace("    ", "\t")

    def get_label(self):
        if self.label == "relative":
            return self.path
        elif self.label == "name":
            return os.path.basename(self.path)
        elif self.label == "ext":
            return os.path.splitext(self.path)[1]
        else:
            return ""

    def count_max_backticks(self, contents):
        max_backticks = 0
        lines = contents.split("\n")
        for line in lines:
            if line.startswith("`"):
                max_backticks = max(max_backticks, len(line) - len(line.lstrip("`")))
        return max_backticks

    def delineate(self, contents, format, label, max_backticks=0):
        if format == "md":
            backticks_str = "`" * (max_backticks + 2) if max_backticks >= 3 else "```"
            return f"{backticks_str}{label}\n{contents}\n{backticks_str}"
        elif format == "xml":
            return f"<file path='{label}'>\n{contents}\n</file>"
        else:
            return contents


def concat_refs(file_references: list):
    return "\n\n".join(ref.output for ref in file_references)
