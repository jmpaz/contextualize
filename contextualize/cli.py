import os
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


def preprocess_args():
    """
    Move forwardable options from after subcommand to before it.
    """
    if len(sys.argv) < 2:
        return

    subcommands = {"payload", "cat", "fetch", "map", "shell"}

    # options that should be moved / which take values
    forwardable = {"--prompt", "-p", "--wrap", "-w", "--copy", "-c", "--write-file"}
    value_options = {"--prompt", "-p", "--wrap", "--write-file"}

    # find subcommand position
    subcommand_idx = None
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in subcommands:
            subcommand_idx = i
            break
        elif arg in value_options and i + 1 < len(sys.argv):
            i += 2  # skip the option value
        else:
            i += 1

    if subcommand_idx is None:
        return

    # extract forwardable options
    to_move = []
    remaining = []
    i = subcommand_idx + 1

    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in forwardable:
            to_move.append(arg)
            if (  # check if this option takes a value
                arg in value_options
                and i + 1 < len(sys.argv)
                and not sys.argv[i + 1].startswith("-")
            ):
                to_move.append(sys.argv[i + 1])
                i += 1
        else:
            remaining.append(arg)
        i += 1

    # reconstruct sys.argv
    if to_move:
        sys.argv = (
            sys.argv[:subcommand_idx] + to_move + [sys.argv[subcommand_idx]] + remaining
        )


preprocess_args()


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
@click.option(
    "--position",
    "output_position",
    type=click.Choice(["append", "prepend"], case_sensitive=False),
    default=None,
    help="Where to place this command's output relative to piped stdin",
)
@click.option(
    "--after", "-a", "append_flag", is_flag=True, help="same as --position append"
)
@click.option(
    "--before", "-b", "prepend_flag", is_flag=True, help="same as --position prepend"
)
@click.pass_context
def cli(
    ctx,
    prompt,
    wrap_short,
    wrap_mode,
    copy,
    write_file,
    output_position,
    append_flag,
    prepend_flag,
):
    """
    Contextualize CLI - model context preparation utility
    """
    ctx.ensure_object(dict)
    ctx.obj["prompt"] = prompt
    ctx.obj["wrap_mode"] = "md" if wrap_short else wrap_mode
    ctx.obj["copy"] = copy
    ctx.obj["write_file"] = write_file
    if append_flag and prepend_flag:
        raise click.BadParameter("use -a or -b, not both")

    if append_flag:
        output_pos = "append"
    elif prepend_flag:
        output_pos = "prepend"
    else:
        output_pos = output_position or "append"

    ctx.obj["output_pos"] = output_pos

    stdin_data = ""
    if not sys.stdin.isatty():
        stdin_data = sys.stdin.read()
    ctx.obj["stdin_data"] = stdin_data

    if ctx.invoked_subcommand is None and not stdin_data:
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
    stdin_data = ctx.obj.get("stdin_data", "")
    position = ctx.obj.get("output_pos", "append")
    prompts = ctx.obj["prompt"]
    no_subcmd = ctx.invoked_subcommand is None

    if subcommand_output and stdin_data:
        if position == "append":
            raw_text = stdin_data + "\n\n" + subcommand_output
        else:
            raw_text = subcommand_output + "\n\n" + stdin_data
    elif subcommand_output:
        raw_text = subcommand_output
    elif stdin_data and no_subcmd:
        # no subcommand, just processing stdin
        raw_text = stdin_data
    else:
        return

    if subcommand_output and stdin_data and prompts:
        # we're in a pipeline with both input and output; apply wrapping/prompts only to the new content
        wrapped_new_content = wrap_text(subcommand_output, ctx.obj["wrap_mode"])

        if len(prompts) == 1:
            prompted_content = f"{prompts[0]}\n{wrapped_new_content}"
        else:
            prompted_content = f"{prompts[0]}\n{wrapped_new_content}\n\n{prompts[1]}"

        # combine with stdin
        if position == "append":
            final_output = stdin_data + "\n\n" + prompted_content
        else:
            final_output = prompted_content + "\n\n" + stdin_data
    else:
        # normal case: wrap everything and add prompts
        wrapped_text = wrap_text(raw_text, ctx.obj["wrap_mode"])

        # special case: stdin-only with single prompt
        if len(prompts) == 1 and no_subcmd:
            if ctx.obj["output_pos"] == "append":
                final_output = f"{wrapped_text}\n{prompts[0]}"
            else:
                final_output = f"{prompts[0]}\n{wrapped_text}"
        else:
            final_output = add_prompt_wrappers(wrapped_text, prompts)

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
    required=False,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--inject", is_flag=True, help="Process {cx::...} content injection patterns"
)
@click.pass_context
def payload_cmd(ctx, manifest_path, inject):
    """
    Render a context payload from a provided YAML manifest.
    If no path is given and stdin is piped, read the manifest from stdin.
    """
    try:
        import os

        import yaml

        from .payload import assemble_payload, render_from_yaml
    except ImportError:
        raise click.ClickException(
            "You need `pyyaml` installed (pip install contextualize[payload])"
        )

    if manifest_path:
        payload_content = render_from_yaml(manifest_path, inject=inject)
        return payload_content

    # only use stdin when no manifest file is provided
    stdin_data = ctx.obj.get("stdin_data", "")
    if not stdin_data:  # no input file and no piped data → show help
        click.echo(ctx.get_help())
        ctx.exit(1)

    # clear stdin_data since we're consuming it here
    ctx.obj["stdin_data"] = ""

    raw = stdin_data
    try:
        data = yaml.safe_load(raw)
    except Exception as e:
        raise click.ClickException(f"Invalid YAML on stdin: {e}")

    if not isinstance(data, dict):
        raise click.ClickException(
            "Manifest must be a mapping with 'config' and 'components'"
        )

    # determine base directory
    cfg = data.get("config", {})
    if "root" in cfg:
        base_dir = os.path.expanduser(cfg.get("root") or "~")
    else:
        base_dir = os.getcwd()

    comps = data.get("components")
    if not isinstance(comps, list):
        raise click.ClickException("'components' must be a list")

    # assemble and return the payload string
    payload_content = assemble_payload(comps, base_dir, inject=inject)
    return payload_content


@cli.command("cat")
@click.argument("paths", nargs=-1, type=str)
@click.option("--ignore", multiple=True, help="File(s) to ignore")
@click.option("-f", "--format", default="md", help="Output format (md/xml/shell)")
@click.option(
    "-l",
    "--label",
    default="relative",
    help="Label style for references (relative/name/ext)",
)
@click.option("--git-pull", is_flag=True, help="Pull cached git repos")
@click.option("--git-reclone", is_flag=True, help="Reclone cached git repos")
@click.option(
    "--inject", is_flag=True, help="Process {cx::...} content injection patterns"
)
@click.pass_context
def cat_cmd(ctx, paths, ignore, format, label, git_pull, git_reclone, inject):
    """
    Prepare and concatenate file references (raw).
    """
    if not paths:
        click.echo(ctx.get_help())
        ctx.exit()

    from pathlib import Path

    from .gitcache import ensure_repo, expand_git_paths, parse_git_target
    from .reference import URLReference, concat_refs, create_file_references

    def add_file_refs(paths_list):
        """Helper to add file references for a list of paths"""
        file_refs = create_file_references(paths_list, ignore, format, label, inject=inject, depth=5)["refs"]
        refs.extend(file_refs)

    refs = []
    for p in paths:
        if p.startswith("http://") or p.startswith("https://"):
            tgt = parse_git_target(p)
            if tgt and (tgt.path is not None or tgt.repo_url.endswith(".git") or tgt.repo_url != p):
                repo_dir = ensure_repo(tgt, pull=git_pull, reclone=git_reclone)
                expanded_paths = [str(Path(item)) for item in expand_git_paths(repo_dir, tgt.path)] if tgt.path else [str(Path(repo_dir))]
                for path in expanded_paths:
                    add_file_refs([path])
            else:
                refs.append(URLReference(p, format=format, label=label, inject=inject, depth=5))
        elif os.path.exists(p):
            add_file_refs([p])
        else:
            tgt = parse_git_target(p)
            if tgt:
                repo_dir = ensure_repo(tgt, pull=git_pull, reclone=git_reclone)
                expanded_paths = [str(Path(item)) for item in expand_git_paths(repo_dir, tgt.path)] if tgt.path else [str(Path(repo_dir))]
                for path in expanded_paths:
                    add_file_refs([path])
            else:
                add_file_refs([p])

    return concat_refs(refs)


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
@click.argument("paths", nargs=-1, type=str)
@click.option(
    "-t",
    "--max-tokens",
    type=int,
    default=10000,
    help="Maximum tokens for the repo map",
)
@click.option("--ignore", multiple=True, help="File(s) to ignore")
@click.option(
    "-f",
    "--format",
    default="plain",
    help="Output format for the repo map (plain/shell)",
)
@click.option("--git-pull", is_flag=True, help="Pull cached git repos")
@click.option("--git-reclone", is_flag=True, help="Reclone cached git repos")
@click.pass_context
def map_cmd(ctx, paths, max_tokens, ignore, format, git_pull, git_reclone):
    """
    Generate a repository map (raw).
    """
    if not paths:
        click.echo(ctx.get_help())
        ctx.exit()

    from pathlib import Path

    from contextualize.repomap import generate_repo_map_data

    from .gitcache import ensure_repo, expand_git_paths, parse_git_target

    expanded: list[str] = []
    for p in paths:
        if os.path.exists(p):
            expanded.append(p)
            continue

        tgt = parse_git_target(p)
        if tgt:
            repo_dir = ensure_repo(tgt, pull=git_pull, reclone=git_reclone)
            if tgt.path:
                for item in expand_git_paths(repo_dir, tgt.path):
                    expanded.append(str(Path(item)))
            else:
                expanded.append(str(Path(repo_dir)))
        else:
            expanded.append(p)

    result = generate_repo_map_data(expanded, max_tokens, format, ignore)
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
