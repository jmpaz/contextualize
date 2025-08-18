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
    forwardable = {
        "--prompt",
        "-p",
        "--wrap",
        "-w",
        "--copy",
        "-c",
        "--write-file",
        "--copy-segments",
    }
    value_options = {"--prompt", "-p", "--wrap", "--write-file", "--copy-segments"}

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
    "--copy-segments",
    type=int,
    help="Copy output in segments with max tokens per segment. Mutually exclusive with --copy.",
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
    copy_segments,
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
    ctx.obj["copy_segments"] = copy_segments
    ctx.obj["write_file"] = write_file
    if append_flag and prepend_flag:
        raise click.BadParameter("use -a or -b, not both")
    if copy and copy_segments:
        raise click.BadParameter("--copy and --copy-segments are mutually exclusive")

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

        # handle position parameter for single prompt cases
        if len(prompts) == 1:
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
    copy_segments = ctx.obj.get("copy_segments")
    trace_output = ctx.obj.get("trace_output")

    if write_file:
        with open(write_file, "w", encoding="utf-8") as f:
            f.write(final_output)
        if trace_output:
            click.echo(trace_output)
            click.echo("\n-----\n")
        click.echo(f"Wrote {token_count} tokens ({token_method}) to {write_file}")
    elif copy_segments:
        from .utils import build_segment, segment_output, wait_for_enter

        segments = segment_output(raw_text, copy_segments, ctx.obj.get("format", "md"))
        if not segments:
            click.echo("No content to copy.", err=True)
            return

        try:
            for i, (segment_text, _) in enumerate(segments, 1):
                final_segment = build_segment(
                    segment_text,
                    ctx.obj["wrap_mode"],
                    prompts,
                    ctx.obj["output_pos"],
                    i,
                    len(segments),
                )

                copy(final_segment)
                tokens = count_tokens(final_segment, target="cl100k_base")["count"]

                if i == 1:
                    msg = f"({i}/{len(segments)}) Copied {tokens} tokens to clipboard ({token_method})"
                else:
                    msg = f"({i}/{len(segments)}) Copied {tokens} tokens to clipboard"

                if i < len(segments):
                    click.echo(msg + "...", nl=False)
                    if not wait_for_enter():
                        click.echo("\nCopying interrupted.", err=True)
                        break
                else:
                    click.echo(msg + ".")
            else:
                if trace_output:
                    click.echo("\n-----\n")
                    click.echo(trace_output)
        except Exception as e:
            click.echo(f"Error copying to clipboard: {e}", err=True)
    elif copy_flag:
        try:
            copy(final_output)
            if trace_output:
                click.echo(trace_output)
                click.echo("\n-----\n")
            click.echo(f"Copied {token_count} tokens ({token_method}) to clipboard.")
        except Exception as e:
            click.echo(f"Error copying to clipboard: {e}", err=True)
    else:
        click.echo(final_output)
        if trace_output:
            click.echo("\n-----\n")
            click.echo(trace_output)


@cli.command("payload")
@click.argument(
    "manifest_paths",
    nargs=-1,
    required=False,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--inject", is_flag=True, help="Process {cx::...} content injection patterns"
)
@click.option(
    "--local-wrap",
    "local_wrap_mode",
    is_flag=False,
    flag_value="xml",
    default=None,
    help=(
        "Wrap each payload as 'md' or 'xml'. Omit to disable local wrapping; passing with no value uses 'xml'. "
        "Applied per-payload before global wrapping/prompts."
    ),
)
@click.option(
    "--local-prompt",
    "local_prompts",
    multiple=True,
    help=(
        "Prepend a line directly above each payload (can be provided multiple times; maps by index). "
        "If provided once together with --note (multiple), each payload gets '<local-prompt><note[i]>' respectively."
    ),
)
@click.option(
    "--note",
    "notes",
    multiple=True,
    help=(
        "Optional per-payload note strings (specify multiple). Used with --local-prompt to label payloads."
    ),
)
@click.option(
    "--trace",
    is_flag=True,
    help="Show paths crawled during Markdown link resolution.",
)
@click.pass_context
def payload_cmd(
    ctx,
    manifest_paths,
    inject,
    local_wrap_mode,
    local_prompts,
    notes,
    trace,
):
    """
    Render context payload(s) from one or more YAML manifests.
    - Accepts multiple manifest paths.
    - If no paths are given and stdin is piped, reads one or more YAML manifests from stdin.
    - When multiple payloads are produced, they are joined with a double newline.
    """
    ctx.obj["format"] = "md"  # for segmentation
    try:
        import os

        import yaml

        from .mdlinks import format_trace_output
        from .payload import (
            assemble_payload_with_mdlinks_from_data,
            render_from_yaml_with_mdlinks,
        )
    except ImportError:
        raise click.ClickException("pyyaml is required")

    def _dedupe_input_refs(refs):
        seen = set()
        out = []
        for r in refs:
            p = getattr(r, "path", None)
            if not p:
                out.append(r)
                continue
            ap = os.path.abspath(p)
            if ap in seen:
                continue
            seen.add(ap)
            out.append(r)
        return out

    # Shared 'seen' set for Markdown link traversal across multiple payloads
    shared_seen = set()

    # Collectors for trace data across all payloads
    all_payloads = []
    all_input_refs = []
    all_trace_items = []
    all_skipped_paths = set()
    all_skip_impact = {}
    base_dirs = []

    if manifest_paths:
        for mp in manifest_paths:
            (
                payload_content,
                input_refs,
                trace_items,
                base_dir,
                skipped_paths,
                skip_impact,
            ) = render_from_yaml_with_mdlinks(mp, inject=inject, global_seen=shared_seen)
            all_payloads.append(payload_content)
            all_input_refs.extend(input_refs)
            all_trace_items.extend(trace_items)
            base_dirs.append(base_dir)
            all_skipped_paths.update(skipped_paths or [])
            if skip_impact:
                all_skip_impact.update(skip_impact)

        # Apply per-payload local wrapping/prompting if requested
        if local_wrap_mode or local_prompts:
            processed = []
            for idx, payload in enumerate(all_payloads):
                text = payload
                if local_wrap_mode:
                    text = wrap_text(text, local_wrap_mode)
                if local_prompts:
                    # If exactly one local prompt provided and notes are present, append notes[idx] to it
                    # and only apply to payloads that have a corresponding note.
                    if len(local_prompts) == 1 and notes:
                        if idx < len(notes):
                            label_line = f"{local_prompts[0]}{notes[idx]}"
                            text = f"{label_line}\n{text}"
                    # If multiple local prompts provided, map by index; missing indices get no prompt.
                    elif idx < len(local_prompts):
                        text = f"{local_prompts[idx]}\n{text}"
                processed.append(text)
            all_payloads = processed

        if trace:
            # Build per-payload traces with cross-payload deduping (✓)
            trace_outputs = []
            previously_seen = set()
            multi = len(manifest_paths) > 1

            # Pre-compute per-manifest slices from our accumulators
            # We rebuild by re-rendering each manifest with shared_seen to get its local refs
            for idx, mp in enumerate(manifest_paths):
                # Rebuild trace data for this payload WITHOUT global cross-payload seen,
                # so depth and traversal remain identical to a standalone run.
                (
                    _content,
                    input_refs,
                    trace_items,
                    base_dir,
                    skipped_paths,
                    skip_impact,
                ) = render_from_yaml_with_mdlinks(mp, inject=inject, global_seen=None)

                tr = format_trace_output(
                    input_refs,
                    trace_items,
                    skipped_paths=sorted(set(skipped_paths)) if skipped_paths else None,
                    skip_impact=skip_impact if skip_impact else None,
                    common_prefix=None,
                    stdin_data=None,
                    injection_traces=None,
                    global_seen=previously_seen,
                    heading=(
                        f"Payload {idx+1}/{len(manifest_paths)}" if multi else None
                    ),
                )
                trace_outputs.append(tr)

                # update previously_seen with inputs and discovered targets
                for r in input_refs:
                    if hasattr(r, "path"):
                        previously_seen.add(os.path.abspath(r.path))
                for tgt, _src, _depth in trace_items:
                    previously_seen.add(os.path.abspath(tgt))

            ctx.obj["trace_output"] = "\n\n---\n\n".join(trace_outputs)

        return "\n\n".join(all_payloads)

    # only use stdin when no manifest file is provided
    stdin_data = ctx.obj.get("stdin_data", "")
    if not stdin_data:
        click.echo(ctx.get_help())
        ctx.exit(1)

    # preserve stdin for trace output
    original_stdin = stdin_data
    ctx.obj["stdin_data"] = ""

    raw = stdin_data.strip()

    # Prefer heuristic split (double-newline boundary before a new top-level 'config:')
    # to honor the contract that chained payloads are separated by a blank line.
    import re as _re
    docs = []
    if "---" not in raw:
        config_hits = len(_re.findall(r"(?m)^(?:\s*)config:\s*(?:$)", raw))
        if config_hits >= 2 or "\n\nconfig:" in raw:
            chunks = _re.split(r"\n{2,}(?=config:\s*(?:\n|$))", raw, flags=_re.MULTILINE)
            chunks = [c.strip() for c in chunks if c.strip()]
            for idx, chunk in enumerate(chunks, 1):
                try:
                    d = yaml.safe_load(chunk)
                    if d is not None:
                        docs.append(d)
                except Exception as e:
                    raise click.ClickException(
                        f"Invalid YAML on stdin (chunk {idx}): {e}"
                    )

    # If heuristic split didn't find multiple manifests, fall back to YAML multi-doc
    if not docs:
        try:
            for d in yaml.safe_load_all(raw):
                if d is None:
                    continue
                docs.append(d)
        except Exception as e:
            raise click.ClickException(f"Invalid YAML on stdin: {e}")

    if not docs:
        raise click.ClickException("No YAML manifests found on stdin")

    for d in docs:
        if not isinstance(d, dict):
            raise click.ClickException(
                "Each manifest must be a mapping with 'config' and 'components'"
            )

    # Build each manifest with shared link traversal dedupe
    for d in docs:
        try:
            (
                payload_content,
                input_refs,
                trace_items,
                base_dir,
                skipped_paths,
                skip_impact,
            ) = assemble_payload_with_mdlinks_from_data(
                d, os.getcwd(), inject=inject, global_seen=shared_seen
            )
        except Exception as e:
            raise click.ClickException(str(e))

        all_payloads.append(payload_content)
        all_input_refs.extend(input_refs)
        all_trace_items.extend(trace_items)
        base_dirs.append(base_dir)
        all_skipped_paths.update(skipped_paths or [])
        if skip_impact:
            all_skip_impact.update(skip_impact)

    # Apply per-payload local wrapping/prompting if requested
    if local_wrap_mode or local_prompts:
        processed = []
        for idx, payload in enumerate(all_payloads):
            text = payload
            if local_wrap_mode:
                text = wrap_text(text, local_wrap_mode)
            if local_prompts:
                if len(local_prompts) == 1 and notes:
                    if idx < len(notes):
                        label_line = f"{local_prompts[0]}{notes[idx]}"
                        text = f"{label_line}\n{text}"
                elif idx < len(local_prompts):
                    text = f"{local_prompts[idx]}\n{text}"
            processed.append(text)
        all_payloads = processed

    if trace:
        # Per-payload traces with cross-payload dedupe
        trace_outputs = []
        previously_seen = set()
        multi = len(docs) > 1
        for idx, d in enumerate(docs):
            (
                _content,
                input_refs,
                trace_items,
                base_dir,
                skipped_paths,
                skip_impact,
            ) = assemble_payload_with_mdlinks_from_data(
                # Rebuild trace data for this payload WITHOUT global cross-payload seen,
                # so depth and traversal remain identical to a standalone run.
                d, os.getcwd(), inject=inject, global_seen=None
            )

            tr = format_trace_output(
                input_refs,
                trace_items,
                skipped_paths=sorted(set(skipped_paths)) if skipped_paths else None,
                skip_impact=skip_impact if skip_impact else None,
                common_prefix=None,
                stdin_data=None,
                injection_traces=None,
                global_seen=previously_seen,
                heading=(f"Payload {idx+1}/{len(docs)}" if multi else None),
            )
            trace_outputs.append(tr)

            for r in input_refs:
                if hasattr(r, "path"):
                    previously_seen.add(os.path.abspath(r.path))
            for tgt, _src, _depth in trace_items:
                previously_seen.add(os.path.abspath(tgt))

        ctx.obj["trace_output"] = "\n\n---\n\n".join(trace_outputs)

    return "\n\n".join(all_payloads)


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
@click.option(
    "--link-depth",
    type=int,
    default=0,
    help="Follow Markdown links this many levels deep.",
)
@click.option(
    "--link-skip",
    multiple=True,
    help="Paths to skip when resolving Markdown links. Can be specified multiple times.",
)
@click.option(
    "--trace",
    is_flag=True,
    help="Show paths crawled during execution.",
)
@click.option(
    "--link-scope",
    type=click.Choice(["first", "all"], case_sensitive=False),
    default="all",
    help="Resolve links starting from only the first input ('first') or all inputs ('all').",
)
@click.pass_context
def cat_cmd(
    ctx,
    paths,
    ignore,
    format,
    label,
    git_pull,
    git_reclone,
    inject,
    link_depth,
    link_scope,
    link_skip,
    trace,
):
    """
    Prepare and concatenate file references (raw).
    """
    if not paths:
        click.echo(ctx.get_help())
        ctx.exit()

    ctx.obj["format"] = format  # for segmentation

    from pathlib import Path

    from .gitcache import ensure_repo, expand_git_paths, parse_git_target
    from .mdlinks import add_markdown_link_refs, format_trace_output
    from .reference import (
        FileReference,
        URLReference,
        concat_refs,
        create_file_references,
    )

    injection_trace_items = [] if inject and trace else None

    def add_file_refs(paths_list):
        """Helper to add file references for a list of paths"""
        refs.extend(
            create_file_references(
                paths_list,
                ignore,
                format,
                label,
                inject=inject,
                depth=5,
                trace_collector=injection_trace_items,
            )["refs"]
        )

    refs = []
    for p in paths:
        if p.startswith("http://") or p.startswith("https://"):
            tgt = parse_git_target(p)
            if tgt and (
                tgt.path is not None
                or tgt.repo_url.endswith(".git")
                or tgt.repo_url != p
            ):
                repo_dir = ensure_repo(tgt, pull=git_pull, reclone=git_reclone)
                expanded_paths = (
                    [str(Path(item)) for item in expand_git_paths(repo_dir, tgt.path)]
                    if tgt.path
                    else [str(Path(repo_dir))]
                )
                for path in expanded_paths:
                    add_file_refs([path])
            else:
                refs.append(
                    URLReference(
                        p,
                        format=format,
                        label=label,
                        inject=inject,
                        depth=5,
                        trace_collector=injection_trace_items,
                    )
                )
        elif os.path.exists(p):
            add_file_refs([p])
        else:
            tgt = parse_git_target(p)
            if tgt:
                repo_dir = ensure_repo(tgt, pull=git_pull, reclone=git_reclone)
                expanded_paths = (
                    [str(Path(item)) for item in expand_git_paths(repo_dir, tgt.path)]
                    if tgt.path
                    else [str(Path(repo_dir))]
                )
                for path in expanded_paths:
                    add_file_refs([path])
            else:
                add_file_refs([p])

    skipped_paths = [os.path.abspath(p) for p in link_skip] if link_skip else []
    trace_items = []
    skip_impact = {}

    input_refs = [r for r in refs if isinstance(r, FileReference)]

    # resolve markdown links
    if link_depth > 0:
        refs[:], trace_items, skip_impact = add_markdown_link_refs(
            refs,
            link_depth=link_depth,
            scope=link_scope,
            format_=format,
            label=label,
            inject=inject,
            link_skip=link_skip,
        )

    result = concat_refs(refs)

    if trace:
        stdin_data = ctx.obj.get("stdin_data", "")
        trace_output = format_trace_output(
            input_refs,
            trace_items,
            skipped_paths,
            skip_impact,
            common_prefix=None,
            stdin_data=stdin_data if stdin_data else None,
            injection_traces=injection_trace_items,
        )
        ctx.obj["trace_output"] = trace_output

    return result


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

    ctx.obj["format"] = "md"  # fetch returns markdown

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

    ctx.obj["format"] = format  # for segmentation

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
    ctx.obj["format"] = format  # for segmentation
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
