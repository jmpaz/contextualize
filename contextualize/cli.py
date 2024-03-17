import os
from pathspec import PathSpec
import argparse
from contextualize.reference import FileReference, concat_refs


def create_file_references(paths, ignore_paths=None, format="md", label="relative"):
    file_references = []
    ignore_patterns = [
        # ".git/",
        # "venv/",
        # ".venv/",
        ".gitignore",
        "__pycache__/",
        "__init__.py",
    ]

    if ignore_paths:
        for path in ignore_paths:
            if os.path.isfile(path):
                with open(path, "r") as file:
                    ignore_patterns.extend(file.read().splitlines())

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

    return file_references


def is_ignored(path, gitignore_patterns):
    path_spec = PathSpec.from_lines("gitwildmatch", gitignore_patterns)
    return path_spec.match_file(path)


def cat_cmd(args):
    file_references = create_file_references(
        args.paths, args.ignore, args.format, args.label
    )
    concatenated_refs = concat_refs(file_references)
    print(concatenated_refs)


def main():
    parser = argparse.ArgumentParser(description="File reference CLI")
    subparsers = parser.add_subparsers(dest="command")

    cat_parser = subparsers.add_parser(
        "cat", help="Prepare and concatenate file references"
    )
    cat_parser.add_argument("paths", nargs="+", help="File or folder paths")
    cat_parser.add_argument("--ignore", nargs="*", help="File(s) to ignore")
    cat_parser.add_argument(
        "--format",
        default="md",
        help="Output format (options: 'md', 'xml', default 'md')",
    )
    cat_parser.add_argument(
        "--label",
        default="relative",
        help="Label style (options: 'relative', 'name', 'ext', default 'relative')",
    )
    cat_parser.set_defaults(func=cat_cmd)

    args = parser.parse_args()

    if args.command == "cat":
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
