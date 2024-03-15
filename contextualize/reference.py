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

        contents = self.delineate(contents, self.format, self.get_label())

        return contents

    def extract_range(self, contents, range):
        start, end = range
        lines = contents.split("\n")
        return "\n".join(lines[start - 1 : end])

    def clean(self, contents):
        # perform cleaning operations, e.g., replace spaces with tabs
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

    def delineate(self, contents, format, label):
        if format == "md":
            return f"```{label}\n{contents}\n```"
        elif format == "xml":
            return f"<file path='{label}'>\n{contents}\n</file>"
        else:
            return contents


def concat_refs(file_references: list):
    return "\n\n".join(ref.output for ref in file_references)
