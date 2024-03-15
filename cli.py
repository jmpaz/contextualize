import os
import fnmatch
import argparse
from contextualize.reference import FileReference, concat_refs


def create_file_references(paths, gitignore_path=None):
    file_references = []
    gitignore_patterns = read_gitignore(gitignore_path)

    for path in paths:
        if os.path.isfile(path):
            if not is_ignored(path, gitignore_patterns):
                file_references.append(FileReference(path))
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs[:] = [
                    d
                    for d in dirs
                    if not is_ignored(os.path.join(root, d), gitignore_patterns)
                ]
                for file in files:
                    file_path = os.path.join(root, file)
                    if not is_ignored(file_path, gitignore_patterns):
                        file_references.append(FileReference(file_path))

    return file_references


def read_gitignore(gitignore_path):
    if gitignore_path and os.path.isfile(gitignore_path):
        with open(gitignore_path, "r") as file:
            return file.read().splitlines()
    elif os.path.isfile(".gitignore"):
        with open(".gitignore", "r") as file:
            return file.read().splitlines()
    else:
        return []


def is_ignored(path, gitignore_patterns):
    for pattern in gitignore_patterns:
        pattern = pattern.strip()
        if not pattern or pattern.startswith("#"):
            continue
        if pattern.startswith("/"):
            if path.startswith(pattern[1:]):
                return True
        elif "/" not in pattern:
            if os.path.basename(path) == pattern:
                return True
        else:
            if fnmatch.fnmatch(path, pattern):
                return True
    return False


def cat_cmd(args):
    file_references = create_file_references(args.paths, args.gitignore)
    concatenated_refs = concat_refs(file_references)
    print(concatenated_refs)


def main():
    parser = argparse.ArgumentParser(description="File reference CLI")
    subparsers = parser.add_subparsers(dest="command")

    cat_parser = subparsers.add_parser(
        "cat", help="Prepare and concatenate file references"
    )
    cat_parser.add_argument("paths", nargs="+", help="File or folder paths")
    cat_parser.add_argument("--gitignore", help="Path to .gitignore file")
    cat_parser.set_defaults(func=cat_cmd)

    args = parser.parse_args()

    if args.command == "cat":
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
