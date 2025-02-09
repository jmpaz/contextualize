#!/usr/bin/env python
import os
import sys

from aider.repomap import RepoMap, find_src_files
from pyperclip import copy

from contextualize.tokenize import count_tokens


# minimal IO helper for RepoMap
class ConsoleIO:
    def tool_output(self, msg):
        print(msg)

    def tool_error(self, msg):
        print("ERROR: " + msg, file=sys.stderr)

    def tool_warning(self, msg):
        print("WARNING: " + msg, file=sys.stderr)

    def read_text(self, fname):
        try:
            with open(fname, "r") as file:
                return file.read()
        except Exception as e:
            self.tool_warning(f"Error reading file {fname}: {str(e)}")
            return ""


class TokenCounter:
    def token_count(self, text):
        result = count_tokens(text, target="cl100k_base")
        return result["count"]


def repomap_cmd(args):
    files = []
    for path in args.paths:
        if os.path.isdir(path):
            files.extend(find_src_files(path))
        else:
            files.append(path)

    io = ConsoleIO()
    main_model = TokenCounter()

    rm = RepoMap(map_tokens=args.max_tokens, main_model=main_model, io=io)

    repo_map = rm.get_repo_map(chat_files=[], other_files=files)
    if not repo_map:
        io.tool_error("No repository map was generated.")
        return

    if getattr(args, "format", "plain") == "shell":
        repo_map = f"❯ repo-map {' '.join(args.paths)}\n{repo_map}"

    token_info = count_tokens(repo_map, target="cl100k_base")
    num_files = len(files)

    if args.output_file:
        with open(args.output_file, "w") as file:
            file.write(repo_map)
        summary_str = (
            f"Wrote map of {num_files} files "
            f"({token_info['count']} tokens) "
            f"to file: {args.output_file}."
        )
        print(summary_str)

    elif args.output == "clipboard":
        try:
            copy(repo_map)
            summary_str = (
                f"Copied map of {num_files} files "
                f"({token_info['count']} tokens) "
                "to clipboard."
            )
            print(summary_str)
        except Exception as e:
            print(f"Error copying to clipboard: {e}", file=sys.stderr)

    else:
        print(repo_map)
