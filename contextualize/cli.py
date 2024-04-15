from pyperclip import copy
import argparse
from contextualize.reference import create_file_references
from contextualize.external import LinearClient, InvalidTokenError
from contextualize.tokenize import call_tiktoken
from contextualize.utils import read_config


def cat_cmd(args):
    references = create_file_references(
        args.paths, args.ignore, args.format, args.label
    )["concatenated"]

    if args.output_file:
        with open(args.output_file, "w") as file:
            file.write(references)
        print(f"Contents written to {args.output_file}")

    if args.output == "clipboard":
        try:
            copy(references)
            token_count = call_tiktoken(references)["count"]
            print(f"Copied {token_count} tokens to clipboard.")
        except Exception as e:
            print(f"Error copying to clipboard: {e}")
    elif not args.output_file:
        print(references)


def ls_cmd(args):
    references = create_file_references(args.paths)["refs"]
    total_tokens = 0
    encoding = None

    if args.encoding and args.model:
        print(
            "Warning: Both 'encoding' and 'model' arguments provided. Using 'encoding' only."
        )

    for ref in references:
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
            if len(references) > 1
            else f"{result['count']} tokens"
        )
        print(output_str)

        total_tokens += result["count"]
        if not encoding:
            encoding = result["encoding"]  # set once for the first file

    if len(references) > 1:
        print(f"\nTotal: {total_tokens} tokens ({encoding})")


def fetch_cmd(args):
    config = read_config(args.config)
    try:
        client = LinearClient(config["LINEAR_TOKEN"])
    except InvalidTokenError as e:
        print(f"Error: {str(e)}")
        return

    issue_ids = []
    for arg in args.issue:
        if arg.startswith("https://linear.app/"):
            issue_id = arg.split("/")[-2]
        else:
            issue_id = arg
        issue_ids.append(issue_id)

    include_properties = (
        args.properties.split(",")
        if args.properties
        else config.get("FETCH_INCLUDE_PROPERTIES", [])
    )

    markdown_outputs = []
    token_counts = {}
    total_tokens = 0

    for issue_id in issue_ids:
        issue = client.get_issue(issue_id)
        if issue is None:
            print(f"Issue {issue_id} not found.")
            continue

        issue_markdown = issue.to_markdown(include_properties=include_properties)
        markdown_outputs.append(issue_markdown)

        token_count = call_tiktoken(issue_markdown)["count"]
        token_counts[issue_id] = token_count
        total_tokens += token_count

    markdown_output = "\n\n".join(markdown_outputs).strip()

    def write_output(content, dest, mode="w"):
        if dest == "clipboard":
            copy(content)
        else:
            with open(dest, mode) as file:
                file.write(content)

    if args.output_file:
        write_output(markdown_output, args.output_file)
        print(f"Wrote {total_tokens} tokens to {args.output_file}")
        if len(issue_ids) > 1:
            for issue_id, count in token_counts.items():
                print(f"- {issue_id}: {count} tokens")
    elif args.output == "clipboard":
        write_output(markdown_output, "clipboard")
        if len(issue_ids) == 1:
            print(f"Copied {total_tokens} tokens to clipboard.")
        else:
            print(f"Copied {total_tokens} tokens to clipboard:")
            for issue_id, count in token_counts.items():
                print(f"- {issue_id}: {count} tokens")
    else:
        print(markdown_output)


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
    fetch_parser = subparsers.add_parser(
        "fetch", help="Fetch and prepare Linear issues"
    )
    fetch_parser.add_argument(
        "issue", nargs="+", help="Issue URL or identifier (e.g., CX-212)"
    )
    fetch_parser.add_argument(
        "--properties",
        help="Comma-separated list of properties to include (e.g., 'labels,project,relations')",
    )
    fetch_parser.add_argument(
        "--output",
        default="console",
        help="Output target (options: 'console' (default), 'clipboard')",
    )
    fetch_parser.add_argument("--output-file", help="Optional output file path")
    fetch_parser.add_argument(
        "--config",
        help="Path to config file to use (default: $XDG_CONFIG_HOME/contextualize/config.yaml)",
    )
    fetch_parser.set_defaults(func=fetch_cmd)

    args = parser.parse_args()

    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
