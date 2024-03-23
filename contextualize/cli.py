import os
from pathspec import PathSpec
from pyperclip import copy
import argparse
from contextualize.reference import FileReference, concat_refs
from contextualize.tokenize import call_tiktoken


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

    if args.output_file:
        with open(args.output_file, "w") as file:
            file.write(concatenated_refs)
        print(f"Contents written to {args.output_file}")

    if args.output == "clipboard":
        try:
            copy(concatenated_refs)
            print("Contents copied to clipboard.")
        except Exception as e:
            print(f"Error copying to clipboard: {e}")
    elif not args.output_file:
        print(concatenated_refs)


def ls_cmd(args):
    file_references = create_file_references(args.paths)
    total_tokens = 0
    encoding = None

    if args.encoding and args.model:
        print(
            "Warning: Both 'encoding' and 'model' arguments provided. Using 'encoding' only."
        )

    for ref in file_references:
        if args.encoding:
            result = call_tiktoken(ref.file_content, encoding_str=args.encoding)
        elif args.model:
            result = call_tiktoken(
                ref.file_content, encoding_str=None, model_str=args.model
            )
        else:
            result = call_tiktoken(ref.file_content)

        output_str = (
            f"{ref.path}: {result['count']} tokens"
            if len(file_references) > 1
            else f"{result['count']} tokens"
        )
        print(output_str)

        total_tokens += result["count"]
        if not encoding:
            encoding = result["encoding"]  # set once for the first file

    if len(file_references) > 1:
        print(f"\nTotal: {total_tokens} tokens ({encoding})")


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
    cat_parser.add_argument(
        "--output",
        default="console",
        help="Output target (options: 'console' (default), 'clipboard')",
    )
    cat_parser.add_argument("--output-file", help="Optional output file path")
    cat_parser.set_defaults(func=cat_cmd)
    ls_parser = subparsers.add_parser("ls", help="List token counts")
    ls_parser.add_argument("paths", nargs="+", help="File or folder paths")
    ls_parser.add_argument(
        "--encoding",
        help="encoding to use for tokenization, e.g., 'cl100k_base' (default), 'p50k_base', 'r50k_base'",
    )
    ls_parser.add_argument(
        "--model",
        help="Model (e.g., 'gpt-3.5-turbo'/'gpt-4' (default), 'text-davinci-003', 'code-davinci-002') to determine which encoding to use for tokenization. Not used if 'encoding' is provided.",
    )
    ls_parser.set_defaults(func=ls_cmd)

    args = parser.parse_args()

    if args.command == "cat":
        args.func(args)
    elif args.command == "ls":
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
