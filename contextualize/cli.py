import click


def validate_prompt(ctx, param, value):
    if len(value) > 2:
        raise click.BadParameter("At most two prompt strings are allowed")
    return value


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-p",
    "--prompt",
    multiple=True,
    type=str,
    callback=validate_prompt,
    help="Up to two prompt strings. Supply one --prompt to prepend, and a second --prompt to append (optional).",
)
@click.pass_context
def cli(ctx, prompt):
    """
    Contextualize CLI
    """
    ctx.ensure_object(dict)
    # Store prompt strings in context for use by subcommands
    ctx.obj["prompt"] = list(prompt)


@cli.command("cat")
@click.pass_context
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
@click.option(
    "-w",
    "--wrap",
    "wrap_mode",
    is_flag=False,
    flag_value="md",
    default=None,
    help="Wrap output as 'md' or 'xml'. Defaults to 'xml' if used with no value.",
)
def cat_cmd(ctx, paths, ignore, format, label, output, output_file, wrap_mode):
    """
    Prepare and concatenate file references
    """
    from pyperclip import copy

    from .reference import create_file_references
    from .tokenize import count_tokens
    from .utils import add_prompt_wrappers, wrap_text

    references = create_file_references(paths, ignore, format, label)["concatenated"]
    final_output = wrap_text(references, wrap_mode)

    # Insert prompt strings (if provided)
    prompt_messages = ctx.obj.get("prompt", [])
    final_output = add_prompt_wrappers(final_output, prompt_messages)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(final_output)
        token_info = count_tokens(references, target="cl100k_base")
        click.echo(
            f"Copied {token_info['count']} tokens to file ({token_info['method']})."
        )
        click.echo(f"Contents written to {output_file}")
    elif output == "clipboard":
        try:
            copy(final_output)
            token_info = count_tokens(references, target="cl100k_base")
            click.echo(
                f"Copied {token_info['count']} tokens to clipboard ({token_info['method']})."
            )
        except Exception as e:
            click.echo(f"Error copying to clipboard: {e}", err=True)
    else:
        click.echo(final_output)


@cli.command("ls")
@click.pass_context
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--openai-encoding",
    help="OpenAI encoding to use (e.g., 'cl100k_base', 'p50k_base', 'r50k_base')",
)
@click.option("--openai-model", help="OpenAI model name for token counting")
@click.option("--anthropic-model", help="Anthropic model to use for token counting")
def ls_cmd(ctx, paths, openai_encoding, anthropic_model, openai_model):
    """
    List token counts for files
    """
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
    lines = []
    for path, count in results:
        if len(references) > 1:
            lines.append(f"{path}: {count} tokens")
        else:
            lines.append(f"{count} tokens\n ({method})")

    if len(references) > 1:
        lines.append(f"\nTotal: {total_tokens} tokens ({method})")
    output_str = "\n".join(lines).strip()

    from .utils import add_prompt_wrappers

    prompt_messages = ctx.obj.get("prompt", [])
    output_str = add_prompt_wrappers(output_str, prompt_messages)

    click.echo(output_str)


@cli.command("fetch")
@click.pass_context
@click.argument("issue", nargs=-1)
@click.option("--properties", help="Comma-separated list of properties to include")
@click.option("--output", default="console", help="Output target (console/clipboard)")
@click.option("--output-file", type=click.Path(), help="Optional output file path")
@click.option("--config", type=click.Path(), help="Path to config file")
@click.option(
    "-w",
    "--wrap",
    "wrap_mode",
    is_flag=False,
    flag_value="md",
    default=None,
    help="Wrap output as 'md' or 'xml'. Defaults to 'xml' if used with no value.",
)
def fetch_cmd(ctx, issue, properties, output, output_file, config, wrap_mode):
    """
    Fetch and prepare Linear issues
    """
    from pyperclip import copy

    from .external import InvalidTokenError, LinearClient
    from .tokenize import call_tiktoken
    from .utils import add_prompt_wrappers, read_config, wrap_text

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
    final_output = wrap_text(markdown_output, wrap_mode)

    prompt_messages = ctx.obj.get("prompt", [])
    final_output = add_prompt_wrappers(final_output, prompt_messages)

    def write_output(content, dest, mode="w"):
        if dest == "clipboard":
            copy(content)
        else:
            with open(dest, mode, encoding="utf-8") as file:
                file.write(content)

    if output_file:
        write_output(final_output, output_file)
        click.echo(f"Wrote {total_tokens} tokens to {output_file}")
        if len(issue_ids) > 1:
            for issue_id, count in token_counts.items():
                click.echo(f"- {issue_id}: {count} tokens")
    elif output == "clipboard":
        write_output(final_output, "clipboard")
        if len(issue_ids) == 1:
            click.echo(f"Copied {total_tokens} tokens to clipboard.")
        else:
            click.echo(f"Copied {total_tokens} tokens to clipboard:")
            for issue_id, count in token_counts.items():
                click.echo(f"- {issue_id}: {count} tokens")
    else:
        click.echo(final_output)


@cli.command("map")
@click.pass_context
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
@click.option(
    "-w",
    "--wrap",
    "wrap_mode",
    is_flag=False,
    flag_value="md",
    default=None,
    help="Wrap output as 'md' or 'xml'. Defaults to 'xml' if used with no value.",
)
def map_cmd(ctx, paths, max_tokens, output, format, output_file, wrap_mode):
    """
    Generate a repository map
    """
    from pyperclip import copy

    from contextualize.repomap import generate_repo_map_data

    from .utils import add_prompt_wrappers, wrap_text

    result = generate_repo_map_data(paths, max_tokens, format)
    if "error" in result:
        click.echo(result["error"], err=True)
        return

    repo_map = result["repo_map"]
    final_output = wrap_text(repo_map, wrap_mode)

    prompt_messages = ctx.obj.get("prompt", [])
    final_output = add_prompt_wrappers(final_output, prompt_messages)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_output)
        click.echo(result["summary"] + f" written to: {output_file}.")
    elif output == "clipboard":
        copy(final_output)
        click.echo(result["summary"] + " copied to clipboard.")
    else:
        click.echo(final_output)


@cli.command("shell")
@click.pass_context
@click.argument("commands", nargs=-1, required=True)
@click.option(
    "-f",
    "--format",
    default="shell",
    help="Output format (md/xml/shell). Defaults to shell.",
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
@click.option(
    "-w",
    "--wrap",
    "wrap_mode",
    is_flag=False,
    flag_value="md",
    default=None,
    help="Wrap output as 'md' or 'xml'. Defaults to 'xml' if used with no value.",
)
def shell_cmd(ctx, commands, format, output, output_file, capture_stderr, wrap_mode):
    """
    Run arbitrary shell commands. Example:

        contextualize shell "man waybar" "ls --help"
    """
    from pyperclip import copy

    from .shell import create_command_references
    from .tokenize import count_tokens
    from .utils import add_prompt_wrappers, wrap_text

    refs_data = create_command_references(
        commands=commands,
        format=format,
        capture_stderr=capture_stderr,
    )
    concatenated = refs_data["concatenated"]
    final_output = wrap_text(concatenated, wrap_mode)

    prompt_messages = ctx.obj.get("prompt", [])
    final_output = add_prompt_wrappers(final_output, prompt_messages)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_output)
        token_info = count_tokens(concatenated, target="cl100k_base")
        click.echo(
            f"Wrote {token_info['count']} tokens ({token_info['method']}) to {output_file}"
        )
    elif output == "clipboard":
        try:
            copy(final_output)
            token_info = count_tokens(concatenated, target="cl100k_base")
            click.echo(
                f"Copied {token_info['count']} tokens ({token_info['method']}) to clipboard."
            )
        except Exception as e:
            click.echo(f"Error copying to clipboard: {e}", err=True)
    else:
        click.echo(final_output)


def main():
    cli()


if __name__ == "__main__":
    main()
