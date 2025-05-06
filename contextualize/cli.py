import sys

import click
from pyperclip import copy

from .tokenize import count_tokens
from .utils import add_prompt_wrappers, read_config, wrap_text


def validate_prompt(ctx, param, value):
    """
    Ensure at most two prompt strings are provided.
    """
    if len(value) > 2:
        raise click.BadParameter("At most two prompt strings are allowed.")
    return value


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "-p",
    "--prompt",
    multiple=True,
    callback=validate_prompt,
    help=(
        "Up to two prompt strings. Provide one to prepend, and "
        "an optional second one to append to command output."
    ),
)
@click.option("-w", "wrap_short", is_flag=True, help="Wrap output as 'md'.")
@click.option(
    "--wrap",
    "wrap_mode",
    is_flag=False,
    flag_value="xml",
    default=None,
    help=(
        "Wrap output as 'md' or 'xml'. If used without a value, defaults to 'xml'. "
        "If omitted, no wrapping is done."
    ),
)
@click.option(
    "-c",
    "--copy",
    is_flag=True,
    help="Copy output to clipboard instead of printing to console. Prints labeled token count.",
)
@click.option(
    "--write-file",
    type=click.Path(),
    help="Optional output file path (overrides clipboard/console output).",
)
@click.pass_context
def cli(ctx, prompt, wrap_short, wrap_mode, copy, write_file):
    """
    Contextualize CLI
    """
    ctx.ensure_object(dict)
    ctx.obj["prompt"] = prompt
    ctx.obj["wrap_mode"] = "md" if wrap_short else wrap_mode
    ctx.obj["copy"] = copy
    ctx.obj["write_file"] = write_file

    # handle piped input
    if ctx.invoked_subcommand is None:
        if not sys.stdin.isatty():
            stdin_data = sys.stdin.read()
            ctx.obj["stdin_data"] = stdin_data
        else:
            click.echo(ctx.get_help())
            ctx.exit()


@cli.result_callback()
@click.pass_context
def process_output(ctx, subcommand_output, *args, **kwargs):
    """
    Process subcommand output or piped input:
      1. If output is empty, try to use any captured stdin.
      2. Apply wrap mode.
      3. Insert prompt string(s).
      4. Count tokens on the fully composed text.
      5. Write the final text to file, clipboard, or console.
    """
    if not subcommand_output:
        subcommand_output = ctx.obj.get("stdin_data", "")

    if not subcommand_output:
        return

    raw_text = subcommand_output
    wrapped_text = wrap_text(raw_text, ctx.obj["wrap_mode"])
    final_output = add_prompt_wrappers(wrapped_text, ctx.obj["prompt"])

    token_info = count_tokens(final_output, target="cl100k_base")
    token_count = token_info["count"]
    token_method = token_info["method"]

    write_file = ctx.obj["write_file"]
    copy_flag = ctx.obj["copy"]

    if write_file:
        with open(write_file, "w", encoding="utf-8") as f:
            f.write(final_output)
        click.echo(f"Wrote {token_count} tokens ({token_method}) to {write_file}")
    elif copy_flag:
        try:
            copy(final_output)
            click.echo(f"Copied {token_count} tokens ({token_method}) to clipboard.")
        except Exception as e:
            click.echo(f"Error copying to clipboard: {e}", err=True)
    else:
        click.echo(final_output)


@cli.command("payload")
@click.argument(
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.pass_context
def payload_cmd(ctx, manifest_path):
    """
    Render a context payload from a provided YAML manifest.
    """
    try:
        from .payload import render_from_yaml
    except ImportError:
        raise click.ClickException(
            "You need `pyyaml` installed (pip install contextualize[payload])"
        )

    # This will:
    # 1. load manifest_path as YAML,
    # 2. extract `components: [...]`,
    # 3. assemble them into fenced blocks,
    # 4. return that string.
    return render_from_yaml(manifest_path)


@cli.command("cat")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option("--ignore", multiple=True, help="File(s) to ignore")
@click.option("-f", "--format", default="md", help="Output format (md/xml/shell)")
@click.option(
    "-l",
    "--label",
    default="relative",
    help="Label style for references (relative/name/ext)",
)
@click.pass_context
def cat_cmd(ctx, paths, ignore, format, label):
    """
    Prepare and concatenate file references (raw).
    """
    if not paths:
        click.echo(ctx.get_help())
        ctx.exit()

    from .reference import create_file_references

    refs = create_file_references(paths, ignore, format, label)
    return refs["concatenated"]


@cli.command("fetch")
@click.argument("issue", nargs=-1)
@click.option("--properties", help="Comma-separated list of properties to include")
@click.option("--config", type=click.Path(), help="Path to config file")
@click.pass_context
def fetch_cmd(ctx, issue, properties, config):
    """
    Fetch and prepare Linear issues (returns raw Markdown).
    """
    if not issue:
        click.echo(ctx.get_help())
        ctx.exit()

    from .external import InvalidTokenError, LinearClient

    config_data = read_config(config)
    try:
        client = LinearClient(config_data["LINEAR_TOKEN"])
    except InvalidTokenError as e:
        click.echo(f"Error: {str(e)}", err=True)
        return ""

    issue_ids = []
    for arg in issue:
        if arg.startswith("https://linear.app/"):
            issue_id = arg.split("/")[-2]
        else:
            issue_id = arg
        issue_ids.append(issue_id)

    include_props = (
        properties.split(",")
        if properties
        else config_data.get("FETCH_INCLUDE_PROPERTIES", [])
    )

    markdown_outputs = []
    for issue_id in issue_ids:
        issue_obj = client.get_issue(issue_id)
        if issue_obj is None:
            markdown_outputs.append(f"Error: Issue '{issue_id}' not found.")
            continue
        md = issue_obj.to_markdown(include_properties=include_props)
        markdown_outputs.append(md)

    return "\n\n".join(markdown_outputs)


@cli.command("map")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "-t",
    "--max-tokens",
    type=int,
    default=10000,
    help="Maximum tokens for the repo map",
)
@click.option(
    "-f",
    "--format",
    default="plain",
    help="Output format for the repo map (plain/shell)",
)
@click.pass_context
def map_cmd(ctx, paths, max_tokens, format):
    """
    Generate a repository map (raw).
    """
    if not paths:
        click.echo(ctx.get_help())
        ctx.exit()

    from contextualize.repomap import generate_repo_map_data

    result = generate_repo_map_data(paths, max_tokens, format)
    if "error" in result:
        return result["error"]
    return result["repo_map"]


@cli.command("shell")
@click.argument("commands", nargs=-1, required=True)
@click.option(
    "-f",
    "--format",
    default="shell",
    help="Output format (md/xml/shell). Defaults to shell.",
)
@click.option(
    "--capture-stderr/--no-capture-stderr",
    default=True,
    help="Capture stderr along with stdout. Defaults to True.",
)
@click.pass_context
def shell_cmd(ctx, commands, format, capture_stderr):
    """
    Run arbitrary shell commands (returns raw combined output).
    """
    from .shell import create_command_references

    refs_data = create_command_references(
        commands=commands,
        format=format,
        capture_stderr=capture_stderr,
    )
    return refs_data["concatenated"]


def main():
    cli()


if __name__ == "__main__":
    main()
