import click


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """Contextualize CLI"""
    pass


@cli.command("cat")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option("--ignore", multiple=True, help="File(s) to ignore")
@click.option("-f", "--format", default="md", help="Output format (md/xml/shell)")
@click.option(
    "-l", "--label", default="relative", help="Label style (relative/name/ext)"
)
@click.option(
    "-o", "--output", default="console", help="Output target (console/clipboard)"
)
@click.option("--output-file", type=click.Path(), help="Optional output file path")
def cat_cmd(paths, ignore, format, label, output, output_file):
    """Prepare and concatenate file references"""
    from pyperclip import copy

    from .reference import create_file_references
    from .tokenize import count_tokens

    references = create_file_references(paths, ignore, format, label)["concatenated"]

    if output_file:
        with open(output_file, "w") as file:
            file.write(references)
        token_info = count_tokens(references, target="cl100k_base")
        click.echo(
            f"Copied {token_info['count']} tokens to file ({token_info['method']})."
        )
        click.echo(f"Contents written to {output_file}")
    elif output == "clipboard":
        try:
            copy(references)
            token_info = count_tokens(references, target="cl100k_base")
            click.echo(
                f"Copied {token_info['count']} tokens to clipboard ({token_info['method']})."
            )
        except Exception as e:
            click.echo(f"Error copying to clipboard: {e}", err=True)
    else:
        click.echo(references)


@cli.command("ls")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--openai-encoding",
    help="OpenAI encoding to use (e.g., 'cl100k_base', 'p50k_base', 'r50k_base')",
)
@click.option("--openai-model", help="OpenAI model name for token counting")
@click.option("--anthropic-model", help="Anthropic model to use for token counting")
def ls_cmd(paths, openai_encoding, anthropic_model, openai_model):
    """List token counts for files"""
    import os

    from .reference import create_file_references
    from .tokenize import call_tiktoken, count_tokens

    references = create_file_references(paths)["refs"]
    total_tokens = 0
    method = None
    results = []

    if sum(bool(x) for x in [openai_encoding, anthropic_model, openai_model]) > 1:
        click.echo(
            "Error: Only one of --openai-encoding, --openai-model, or --anthropic-model can be specified",
            err=True,
        )
        return

    if openai_encoding:
        target = openai_encoding
    elif anthropic_model:
        target = anthropic_model
        if "ANTHROPIC_API_KEY" not in os.environ:
            click.echo(
                "Warning: ANTHROPIC_API_KEY not set in environment. Falling back to tiktoken.",
                err=True,
            )
            target = "cl100k_base"
    elif openai_model:
        result = call_tiktoken("test", model_str=openai_model)
        target = result["encoding"]
    else:
        target = (
            "cl100k_base"
            if "ANTHROPIC_API_KEY" not in os.environ
            else "claude-3-5-sonnet-latest"
        )

    for ref in references:
        result = count_tokens(ref.file_content, target=target)
        total_tokens += int(result["count"])
        if not method:
            method = result["method"]
        results.append((ref.path, result["count"]))

    results.sort(key=lambda x: x[1], reverse=True)
    for path, count in results:
        output_str = (
            f"{path}: {count} tokens" if len(references) > 1 else f"{count} tokens"
        )
        click.echo(output_str, nl=len(references) != 1)
        if len(references) == 1:
            click.echo(f" ({method})")
    if len(references) > 1:
        click.echo(f"\nTotal: {total_tokens} tokens ({method})")


@cli.command("fetch")
@click.argument("issue", nargs=-1)
@click.option("--properties", help="Comma-separated list of properties to include")
@click.option("--output", default="console", help="Output target (console/clipboard)")
@click.option("--output-file", type=click.Path(), help="Optional output file path")
@click.option("--config", type=click.Path(), help="Path to config file")
def fetch_cmd(issue, properties, output, output_file, config):
    """Fetch and prepare Linear issues"""
    from pyperclip import copy

    from .external import InvalidTokenError, LinearClient
    from .tokenize import call_tiktoken
    from .utils import read_config

    config_data = read_config(config)
    try:
        client = LinearClient(config_data["LINEAR_TOKEN"])
    except InvalidTokenError as e:
        click.echo(f"Error: {str(e)}", err=True)
        return

    issue_ids = []
    for arg in issue:
        if arg.startswith("https://linear.app/"):
            issue_id = arg.split("/")[-2]
        else:
            issue_id = arg
        issue_ids.append(issue_id)

    include_properties = (
        properties.split(",")
        if properties
        else config_data.get("FETCH_INCLUDE_PROPERTIES", [])
    )

    markdown_outputs = []
    token_counts = {}
    total_tokens = 0

    for issue_id in issue_ids:
        issue_obj = client.get_issue(issue_id)
        if issue_obj is None:
            click.echo(f"Issue {issue_id} not found.", err=True)
            continue

        issue_markdown = issue_obj.to_markdown(include_properties=include_properties)
        markdown_outputs.append(issue_markdown)

        token_info = call_tiktoken(issue_markdown)["count"]
        token_counts[issue_id] = token_info
        total_tokens += token_info

    markdown_output = "\n\n".join(markdown_outputs).strip()

    def write_output(content, dest, mode="w"):
        if dest == "clipboard":
            copy(content)
        else:
            with open(dest, mode) as file:
                file.write(content)

    if output_file:
        write_output(markdown_output, output_file)
        click.echo(f"Wrote {total_tokens} tokens to {output_file}")
        if len(issue_ids) > 1:
            for issue_id, count in token_counts.items():
                click.echo(f"- {issue_id}: {count} tokens")
    elif output == "clipboard":
        write_output(markdown_output, "clipboard")
        if len(issue_ids) == 1:
            click.echo(f"Copied {total_tokens} tokens to clipboard.")
        else:
            click.echo(f"Copied {total_tokens} tokens to clipboard:")
            for issue_id, count in token_counts.items():
                click.echo(f"- {issue_id}: {count} tokens")
    else:
        click.echo(markdown_output)


@cli.command("map")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "-t",
    "--max-tokens",
    type=int,
    default=10000,
    help="Maximum tokens for the repo map",
)
@click.option("--output", default="console", help="Output target (console/clipboard)")
@click.option("-f", "--format", default="plain", help="Output format (plain/shell)")
@click.option("--output-file", type=click.Path(), help="Optional output file path")
def map_cmd(paths, max_tokens, output, format, output_file):
    """Generate a repository map"""
    from .repomap import repomap_cmd

    repomap_cmd(
        paths=paths,
        max_tokens=max_tokens,
        output=output,
        fmt=format,
        output_file=output_file,
    )


@cli.command("shell")
@click.argument("commands", nargs=-1, required=True)
@click.option(
    "-f",
    "--format",
    default="shell",
    help="Output format (md/xml/shell). Defaults to shell.",
)
@click.option(
    "-l",
    "--label",
    default="cmd",
    help="How to label each command (default is to show the entire command).",
)
@click.option(
    "-o",
    "--output",
    default="console",
    help="Output target (console/clipboard). Defaults to console.",
)
@click.option("--output-file", type=click.Path(), help="Optional output file path")
@click.option(
    "--capture-stderr/--no-capture-stderr",
    default=True,
    help="Capture stderr along with stdout. Defaults to True.",
)
def shell_cmd(commands, format, label, output, output_file, capture_stderr):
    """
    Run arbitrary shell commands. Example:

        contextualize cmd "man waybar" "ls --help"
    """
    from pyperclip import copy

    from .shell import create_command_references
    from .tokenize import count_tokens

    refs_data = create_command_references(
        commands=commands,
        format=format,
        label=label,
        capture_stderr=capture_stderr,
    )
    concatenated = refs_data["concatenated"]

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(concatenated)
        token_info = count_tokens(concatenated, target="cl100k_base")
        click.echo(
            f"Wrote {token_info['count']} tokens ({token_info['method']}) to {output_file}"
        )
    elif output == "clipboard":
        try:
            copy(concatenated)
            token_info = count_tokens(concatenated, target="cl100k_base")
            click.echo(
                f"Copied {token_info['count']} tokens ({token_info['method']}) to clipboard."
            )
        except Exception as e:
            click.echo(f"Error copying to clipboard: {e}", err=True)
    else:
        click.echo(concatenated)


def main():
    cli()


if __name__ == "__main__":
    main()
