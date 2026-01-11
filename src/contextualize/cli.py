import os
import sys
from collections.abc import Sequence

import click
from click.formatting import term_len
from pyperclip import copy
from pyperclip import paste as clipboard_paste

from .render.text import process_text
from .utils import add_prompt_wrappers, count_tokens, wrap_text

COMMAND_GROUPS = (
    ("Sources", ("cat", "map", "shell", "paste")),
    ("Manifest", ("hydrate", "payload")),
)
HELP_COL_MAX = 30
HELP_COL_SPACING = 2

GLOBAL_OPTION_LABELS = (
    ("prompt", "--prompt"),
    ("wrap_short", "-w"),
    ("wrap_mode", "--wrap"),
    ("copy", "--copy"),
    ("count_only", "--count"),
    ("copy_segments", "--copy-segments"),
    ("write_file", "--write-file"),
    ("token_target", "--token-target"),
    ("md_model", "--md-model"),
    ("output_position", "--position"),
    ("append_flag", "--after"),
    ("prepend_flag", "--before"),
)

GLOBAL_OPTION_DEFAULTS = {
    "prompt": (),
    "wrap_short": False,
    "wrap_mode": None,
    "copy": False,
    "count_only": False,
    "copy_segments": None,
    "write_file": None,
    "token_target": "cl100k_base",
    "md_model": None,
    "output_position": None,
    "append_flag": False,
    "prepend_flag": False,
}


def _write_bold_section(
    formatter: click.HelpFormatter, title: str, records: list[tuple[str, str]]
) -> None:
    if not records:
        return
    formatter.write("\n")
    formatter.write(click.style(title, bold=True) + "\n")
    formatter.indent()
    formatter.write_dl(records, col_max=HELP_COL_MAX, col_spacing=HELP_COL_SPACING)
    formatter.dedent()


def _collect_used_global_option_labels(ctx: click.Context) -> list[str]:
    parent = ctx.parent
    if parent is None:
        return []
    get_source = getattr(parent, "get_parameter_source", None)
    used: list[str] = []
    if callable(get_source):
        for name, label in GLOBAL_OPTION_LABELS:
            if name not in parent.params:
                continue
            if get_source(name) == click.core.ParameterSource.COMMANDLINE:
                used.append(label)
        return used
    for name, label in GLOBAL_OPTION_LABELS:
        if name not in parent.params:
            continue
        if parent.params.get(name) != GLOBAL_OPTION_DEFAULTS.get(name):
            used.append(label)
    return used


def _command_help_limit(formatter: click.HelpFormatter, names: Sequence[str]) -> int:
    if not names:
        return 45
    max_name = max(term_len(name) for name in names)
    first_col = min(max_name, HELP_COL_MAX) + HELP_COL_SPACING
    return max(formatter.width - first_col - 2, 10)


class OrderedGroup(click.Group):
    def __init__(
        self,
        *args,
        commands_order: Sequence[str] | None = None,
        command_groups: Sequence[tuple[str, Sequence[str]]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._commands_order = list(commands_order or [])
        self._command_groups = [
            (title, set(commands)) for title, commands in (command_groups or [])
        ]

    def list_commands(self, ctx: click.Context) -> list[str]:
        if not self._commands_order:
            return super().list_commands(ctx)
        ordered = [name for name in self._commands_order if name in self.commands]
        remaining = [
            name
            for name in super().list_commands(ctx)
            if name not in self._commands_order
        ]
        return ordered + remaining

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        if not self._command_groups:
            return super().format_commands(ctx, formatter)
        command_entries: list[tuple[str | None, str, click.Command]] = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None or cmd.hidden:
                continue
            group_title = None
            for title, command_set in self._command_groups:
                if subcommand in command_set:
                    group_title = title
                    break
            command_entries.append((group_title, subcommand, cmd))
        if not command_entries:
            return
        grouped_records: dict[str, list[tuple[str, click.Command]]] = {
            title: [] for title, _ in self._command_groups
        }
        other_records: list[tuple[str, click.Command]] = []
        for group_title, name, cmd in command_entries:
            if group_title is None:
                other_records.append((name, cmd))
            else:
                grouped_records[group_title].append((name, cmd))
        for title, _ in self._command_groups:
            entries = grouped_records[title]
            if not entries:
                continue
            names = [name for name, _ in entries]
            limit = _command_help_limit(formatter, names)
            rows = [
                (name, cmd.get_short_help_str(limit=limit)) for name, cmd in entries
            ]
            _write_bold_section(formatter, title.upper(), rows)
        if other_records:
            names = [name for name, _ in other_records]
            limit = _command_help_limit(formatter, names)
            rows = [
                (name, cmd.get_short_help_str(limit=limit))
                for name, cmd in other_records
            ]
            _write_bold_section(formatter, "OTHER", rows)


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

    subcommands = {"payload", "cat", "map", "shell", "paste", "hydrate"}

    # options that should be moved / which take values
    forwardable = {
        "--prompt",
        "-p",
        "--wrap",
        "-w",
        "--copy",
        "-c",
        "--count",
        "--write-file",
        "--copy-segments",
        "--token-target",
        "--md-model",
    }
    value_options = {
        "--prompt",
        "-p",
        "--wrap",
        "--write-file",
        "--copy-segments",
        "--token-target",
        "--md-model",
    }

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
    cls=OrderedGroup,
    commands_order=["cat", "map", "shell", "paste", "hydrate", "payload"],
    command_groups=COMMAND_GROUPS,
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
    "--count",
    "count_only",
    is_flag=True,
    help="Dry run of --copy. Counts and prints # of tokens in output.",
)
@click.option(
    "--copy-segments",
    type=int,
    help="Copy output in segments with max tokens per segment. Mutually exclusive with --copy and --count.",
)
@click.option(
    "--write-file",
    type=click.Path(),
    help="Optional output file path (overrides clipboard/console output).",
)
@click.option(
    "--token-target",
    default="cl100k_base",
    show_default=True,
    help="Encoding/model to use for token counts (e.g., cl100k_base, gpt-4o-mini, claude-3-5-sonnet-20241022).",
)
@click.option(
    "--md-model",
    default=None,
    help="Override OPENAI_MODEL (used for optional image captioning during conversion).",
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
    count_only,
    copy_segments,
    write_file,
    token_target,
    md_model,
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
    ctx.obj["count_only"] = count_only
    ctx.obj["copy_segments"] = copy_segments
    ctx.obj["write_file"] = write_file
    ctx.obj["token_target"] = token_target
    if md_model is not None:
        model = md_model.strip()
        if not model:
            raise click.BadParameter("--md-model cannot be empty")
        os.environ["OPENAI_MODEL"] = model
    ctx.obj["md_model"] = md_model
    if append_flag and prepend_flag:
        raise click.BadParameter("use -a or -b, not both")
    if copy and copy_segments:
        raise click.BadParameter("--copy and --copy-segments are mutually exclusive")
    if copy and count_only:
        raise click.BadParameter("--copy and --count are mutually exclusive")
    if count_only and copy_segments:
        raise click.BadParameter("--count and --copy-segments are mutually exclusive")

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
        if prompt:
            ctx.obj["prompt_only"] = True
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
    stdin_data = ctx.obj.get("stdin_data", "")
    position = ctx.obj.get("output_pos", "append")
    prompts = ctx.obj["prompt"]
    no_subcmd = ctx.invoked_subcommand is None
    prompt_only = ctx.obj.get("prompt_only", False)
    max_tokens_budget = ctx.obj.get("max_tokens")

    if subcommand_output and stdin_data:
        # pipeline: wrap and prompt only the new content
        wrapped_new = wrap_text(subcommand_output, ctx.obj["wrap_mode"])
        prompted = add_prompt_wrappers(wrapped_new, prompts)

        if position == "append":
            final_output = stdin_data + "\n\n" + prompted
        else:
            final_output = prompted + "\n\n" + stdin_data
    else:
        if subcommand_output:
            raw_text = subcommand_output
        elif stdin_data and no_subcmd:
            # no subcommand, just processing stdin
            raw_text = stdin_data
        elif prompt_only and prompts:
            if len(prompts) == 1:
                raw_text = prompts[0]
            else:
                raw_text = f"{prompts[0]}\n\n{prompts[1]}"
        else:
            return

        # normal case: wrap everything and add prompts
        wrapped_text = wrap_text(raw_text, ctx.obj["wrap_mode"])

        if prompt_only:
            final_output = wrapped_text
        else:
            if len(prompts) == 1:
                if ctx.obj["output_pos"] == "append":
                    final_output = f"{wrapped_text}\n{prompts[0]}"
                else:
                    final_output = f"{prompts[0]}\n{wrapped_text}"
            else:
                final_output = add_prompt_wrappers(wrapped_text, prompts)

    token_target = ctx.obj.get("token_target", "cl100k_base")

    token_info = count_tokens(final_output, target=token_target)
    token_count = token_info["count"]
    token_method = token_info["method"]

    write_file = ctx.obj["write_file"]
    copy_flag = ctx.obj["copy"]
    count_flag = ctx.obj.get("count_only")
    copy_segments = ctx.obj.get("copy_segments")
    trace_output = ctx.obj.get("trace_output")
    content_token_total = ctx.obj.get("max_tokens_total")
    if max_tokens_budget and content_token_total is None:
        raise click.ClickException(
            "Token budget was set but content totals were missing."
        )

    if max_tokens_budget and content_token_total > max_tokens_budget:
        breakdown = ctx.obj.get("max_tokens_breakdown")
        msg = f"Token budget exceeded: {content_token_total} tokens > {max_tokens_budget} (target {token_target})."
        if breakdown:
            msg = msg + "\n" + breakdown
        raise click.ClickException(msg)

    if write_file:
        with open(write_file, "w", encoding="utf-8") as f:
            f.write(final_output)
        if trace_output:
            click.echo(trace_output)
            click.echo("\n-----\n")
        click.echo(f"Wrote {token_count} tokens ({token_method}) to {write_file}")
    elif copy_segments:
        from .utils import build_segment, segment_output, wait_for_enter

        segments = segment_output(
            raw_text, copy_segments, ctx.obj.get("format", "md"), token_target
        )
        if not segments:
            click.echo("No content to copy.", err=True)
            return

        try:
            for i, (segment_text, _) in enumerate(segments, 1):
                final_segment = build_segment(
                    segment_text,
                    ctx.obj["wrap_mode"],
                    [] if prompt_only else prompts,
                    ctx.obj["output_pos"],
                    i,
                    len(segments),
                )

                copy(final_segment)
                tokens = count_tokens(final_segment, target=token_target)["count"]

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
    elif count_flag:
        if trace_output:
            click.echo(trace_output)
            click.echo("\n-----\n")
        click.echo(f"Total: {token_count} tokens ({token_method}).")
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
    "manifest_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--inject", is_flag=True, help="Process {cx::...} content injection patterns"
)
@click.option(
    "--trace",
    is_flag=True,
    help="Output an itemized list of files processed.",
)
@click.option(
    "--map",
    "map_components",
    multiple=True,
    help="Render maps only for named components or groups.",
)
@click.option(
    "-m",
    "--map-compatible",
    "map_mode",
    is_flag=True,
    help="Render codemaps when possible; otherwise include file contents.",
)
@click.option(
    "--exclude",
    multiple=True,
    help="Exclude named components or groups from the manifest output.",
)
@click.pass_context
def payload_cmd(ctx, manifest_path, inject, trace, exclude, map_mode, map_components):
    """
    Render a context payload from a manifest.
    If no path is given and stdin is piped, read the manifest from stdin.

    Some common non-text formats are converted to text automatically:
    pdf, docx, pptx, xls/xlsx, csv, epub, msg, images (jpg/jpeg/png),
    audio/video (wav/mp3/m4a/mp4).
    """
    ctx.obj["format"] = "md"  # for segmentation
    token_target = ctx.obj.get("token_target", "cl100k_base")

    def parse_keys(values):
        keys = []
        for value in values:
            for part in value.split(","):
                key = part.strip()
                if key:
                    keys.append(key)
        return keys

    exclude_keys_list = parse_keys(exclude)
    map_keys_list = parse_keys(map_components)
    overlap = sorted(set(exclude_keys_list) & set(map_keys_list))
    if overlap:
        names = ", ".join(overlap)
        raise click.BadParameter(
            f"--exclude cannot be combined with --map for: {names}"
        )
    try:
        import os

        import yaml

        from .render.links import format_trace_output
        from .manifest.payload import (
            render_manifest,
            render_manifest_data,
        )
    except ImportError:
        raise click.ClickException("pyyaml is required")

    if manifest_path:
        from .render.markitdown import MarkItDownConversionError

        try:
            result = render_manifest(
                manifest_path,
                inject=inject,
                exclude_keys=exclude_keys_list,
                map_mode=map_mode,
                map_keys=map_keys_list,
                token_target=token_target,
            )
        except (MarkItDownConversionError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        payload_content = result.payload
        input_refs = result.input_refs
        trace_items = result.trace_items
        base_dir = result.base_dir
        skipped_paths = result.skipped_paths
        skip_impact = result.skip_impact
        if trace:
            stdin_data = ctx.obj.get("stdin_data", "")
            trace_output = format_trace_output(
                input_refs,
                trace_items,
                skipped_paths=skipped_paths,
                skip_impact=skip_impact,
                common_prefix=base_dir,
                stdin_data=stdin_data if stdin_data else None,
                injection_traces=None,  # TODO: add injection trace support for payload
                token_target=token_target,
            )
            ctx.obj["trace_output"] = trace_output
        return payload_content

    # only use stdin when no manifest file is provided
    stdin_data = ctx.obj.get("stdin_data", "")
    if not stdin_data:
        click.echo(ctx.get_help())
        ctx.exit(1)

    # preserve stdin for trace output
    original_stdin = stdin_data
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

    try:
        from .render.markitdown import MarkItDownConversionError

        result = render_manifest_data(
            data,
            os.getcwd(),
            inject=inject,
            exclude_keys=exclude_keys_list,
            map_mode=map_mode,
            map_keys=map_keys_list,
            token_target=token_target,
        )
        payload_content = result.payload
        input_refs = result.input_refs
        trace_items = result.trace_items
        base_dir = result.base_dir
        skipped_paths = result.skipped_paths
        skip_impact = result.skip_impact
    except (MarkItDownConversionError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as e:
        raise click.ClickException(str(e)) from e

    if trace:
        trace_output = format_trace_output(
            input_refs,
            trace_items,
            skipped_paths=skipped_paths,
            skip_impact=skip_impact,
            common_prefix=base_dir,
            stdin_data=None,
            injection_traces=None,  # TODO: add injection trace support for payload
            token_target=token_target,
        )
        ctx.obj["trace_output"] = trace_output

    return payload_content


def _confirm_overwrite(path: str, untracked_count: int = 0) -> bool:
    if untracked_count > 0:
        prompt = (
            f"{path} exists. Replace it and all contents? "
            f"({untracked_count} untracked file{'s' if untracked_count != 1 else ''} "
            f"will also be deleted) [y/N]: "
        )
    else:
        prompt = f"{path} exists. Replace it and all contents? [y/N]: "
    try:
        with open("/dev/tty", "r", encoding="utf-8", errors="ignore") as tty_in:
            while True:
                click.echo(prompt, nl=False, err=True)
                response = tty_in.readline()
                if not response:
                    break
                value = response.strip().lower()
                if value in {"n", "no", ""}:
                    return False
                if value in {"y", "yes"}:
                    return True
    except OSError:
        if untracked_count > 0:
            raise click.ClickException(
                f"{path} contains {untracked_count} untracked file{'s' if untracked_count != 1 else ''}. "
                f"Interactive confirmation required."
            ) from None
        raise click.ClickException(
            f"{path} exists. Use --overwrite to replace it."
        ) from None
    raise click.ClickException(f"{path} exists. Use --overwrite to replace it.")


@cli.command("hydrate")
@click.argument(
    "manifest_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--dir",
    "context_dir",
    type=click.Path(),
    help="Context root directory (default: .context)",
)
@click.option(
    "--access",
    type=click.Choice(["read-only", "writable"], case_sensitive=False),
    help="Context folder access mode (default: writable)",
)
@click.option(
    "--path-strategy",
    type=click.Choice(["on-disk", "by-component"], case_sensitive=False),
    help="Path layout strategy (default: on-disk)",
)
@click.option(
    "--agents-filename",
    "agents_filenames",
    multiple=True,
    help="Filename to write agent prompt content (repeat, default: AGENTS.md with --agents-prompt)",
)
@click.option(
    "--agents-prompt",
    "agents_prompt",
    type=str,
    help="Agent prompt content (default: none)",
)
@click.option(
    "--omit-meta",
    is_flag=True,
    help="Omit manifest and index metadata (default: false)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing context directory without prompting (default: false)",
)
@click.pass_context
def hydrate_cmd(
    ctx,
    manifest_path,
    context_dir,
    access,
    path_strategy,
    agents_filenames,
    agents_prompt,
    omit_meta,
    overwrite,
):
    """
    Materialize a provided YAML manifest into a context folder.
    """
    used_options = _collect_used_global_option_labels(ctx)
    if used_options:
        click.echo(f"ignoring global options [{', '.join(used_options)}]", err=True)
    try:
        import yaml

        from .manifest.hydrate import (
            HydrateOverrides,
            apply_hydration_plan,
            build_hydration_plan,
            build_hydration_plan_data,
            clear_context_dir,
            find_untracked_files,
            plan_matches_existing,
        )
    except ImportError:
        raise click.ClickException("pyyaml is required")

    prompt_value = agents_prompt
    if prompt_value is not None and not prompt_value.strip():
        raise click.BadParameter("--agents-prompt cannot be empty")

    access_value = access.lower() if access else None
    path_strategy_value = path_strategy.lower() if path_strategy else None
    overrides = HydrateOverrides(
        context_dir=context_dir,
        access=access_value,
        path_strategy=path_strategy_value,
        agents_prompt=prompt_value,
        agents_filenames=tuple(agents_filenames),
        omit_meta=omit_meta,
    )
    cwd = os.getcwd()
    data = None
    if not manifest_path:
        stdin_data = ctx.obj.get("stdin_data", "")
        if not stdin_data:
            click.echo(ctx.get_help())
            ctx.exit(1)

        ctx.obj["stdin_data"] = ""
        try:
            data = yaml.safe_load(stdin_data)
        except Exception as exc:
            raise click.ClickException(f"Invalid YAML on stdin: {exc}")
        if not isinstance(data, dict):
            raise click.ClickException(
                "Manifest must be a mapping with 'config' and 'components'"
            )

    try:
        if manifest_path:
            plan = build_hydration_plan(
                manifest_path,
                overrides=overrides,
                cwd=cwd,
            )
        else:
            plan = build_hydration_plan_data(
                data,
                manifest_cwd=cwd,
                overrides=overrides,
                cwd=cwd,
            )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    if plan.context_dir.exists():
        if plan_matches_existing(plan):
            click.echo(f"{plan.context_dir} is already up to date.")
            return None
        untracked = find_untracked_files(plan.context_dir)
        untracked_count = len(untracked)
        if not (overwrite and untracked_count == 0):
            if not _confirm_overwrite(str(plan.context_dir), untracked_count):
                ctx.exit(1)
        try:
            clear_context_dir(plan.context_dir)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    result = apply_hydration_plan(plan)
    click.echo(f"Hydrated {result.file_count} files into {result.context_dir}")
    return None


@cli.command("cat")
@click.argument("paths", nargs=-1, type=str)
@click.option("--ignore", multiple=True, help="File(s) to ignore")
@click.option("-f", "--format", default="md", help="Output format (md/xml/shell/raw)")
@click.option(
    "-l",
    "--label",
    default="relative",
    help="Label style for references (relative/name/ext)",
)
@click.option(
    "--tokens",
    "annotate_tokens",
    is_flag=True,
    help="Annotate each label with a token count for its content.",
)
@click.option("--git-pull", is_flag=True, help="Pull cached git repos")
@click.option("--git-reclone", is_flag=True, help="Reclone cached git repos")
@click.option(
    "--inject", is_flag=True, help="Process {cx::...} content injection patterns"
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Fail if gathered paths exceed this token budget (prints a sorted token breakdown).",
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
    "--rev",
    type=str,
    default=None,
    help="Read local file/directory content from a git revision (e.g., HEAD).",
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
    annotate_tokens,
    git_pull,
    git_reclone,
    inject,
    max_tokens,
    link_depth,
    link_scope,
    link_skip,
    trace,
    rev,
):
    """
    Prepare and concatenate file references (raw).

    Some common non-text formats are converted to text automatically:
    pdf, docx, pptx, xls/xlsx, csv, epub, msg, images (jpg/jpeg/png),
    audio/video (wav/mp3/m4a/mp4).
    """
    token_target = ctx.obj.get("token_target", "cl100k_base")
    if max_tokens is not None and max_tokens <= 0:
        raise click.BadParameter("--max-tokens must be greater than 0")
    ctx.obj["max_tokens"] = max_tokens

    if not paths:
        click.echo(ctx.get_help())
        ctx.exit()

    ctx.obj["format"] = format  # for segmentation

    from pathlib import Path

    from .render.links import (
        add_markdown_link_refs,
        compute_input_token_details,
        format_trace_output,
    )
    from .references import (
        FileReference,
        URLReference,
        concat_refs,
        create_file_references,
        split_path_and_symbols,
    )
    from .git.cache import ensure_repo, expand_git_paths, parse_git_target

    injection_trace_items = [] if inject and trace else None
    ignored_files = []
    ignored_folders = {}

    def add_file_refs(paths_list):
        """Helper to add file references for a list of paths"""
        from .render.markitdown import MarkItDownConversionError

        try:
            result = create_file_references(
                paths_list,
                ignore,
                format,
                label,
                include_token_count=annotate_tokens,
                token_target=token_target,
                inject=inject,
                depth=5,
                trace_collector=injection_trace_items,
            )
        except (MarkItDownConversionError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        refs.extend(result["refs"])
        ignored_files.extend(result.get("ignored_files", []))
        ignored_folders.update(result.get("ignored_folders", {}))

    refs = []
    use_rev = bool(rev)
    repo_root = None
    ignore_spec = None
    from .utils import brace_expand

    if use_rev:
        from pathspec import PathSpec

        from .git.rev import GitRevFileReference, discover_repo_root, list_files_at_rev

        try:
            repo_root = discover_repo_root(paths, cwd=os.getcwd())
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        if not repo_root:
            raise click.ClickException("--rev requires running inside a git repository")
        patterns = [".gitignore", ".git/", "__pycache__/", "__init__.py"]
        if ignore:
            for pat in ignore:
                if "{" in pat and "}" in pat:
                    patterns.extend(brace_expand(pat))
                else:
                    patterns.append(pat)
        ignore_spec = PathSpec.from_lines("gitwildmatch", patterns)

    expanded_all_paths = []
    for p in paths:
        if "{" in p and "}" in p:
            expanded_all_paths.extend(brace_expand(p))
        else:
            expanded_all_paths.append(p)

    for p in expanded_all_paths:
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
                        token_target=token_target,
                        include_token_count=annotate_tokens,
                        inject=inject,
                        depth=5,
                        trace_collector=injection_trace_items,
                    )
                )
        elif use_rev:
            base_path, symbols = split_path_and_symbols(p)
            try:
                spec_rel = os.path.expanduser(base_path)
                if os.path.isabs(spec_rel) or spec_rel.startswith(".."):
                    spec_rel = os.path.relpath(os.path.abspath(spec_rel), repo_root)
                rel_files = list_files_at_rev(repo_root, rev, [spec_rel])
            except Exception as e:
                raise click.ClickException(str(e))
            for relf in rel_files:
                if ignore_spec and ignore_spec.match_file(relf):
                    continue
                ranges = None
                if symbols:
                    text_at_rev = read_file_at_rev(repo_root, rev, relf)
                    try:
                        from .render.map import find_symbol_ranges

                        match_map = find_symbol_ranges(relf, symbols, text=text_at_rev)
                    except Exception:
                        match_map = {}
                    matched = [s for s in symbols if s in match_map]
                    if not matched:
                        click.echo(
                            f"Warning: symbol(s) not found in {relf}@{rev}: {', '.join(symbols)}",
                            err=True,
                        )
                        continue
                    ranges = [match_map[s] for s in matched]
                    symbols = matched

                refs.append(
                    GitRevFileReference(
                        repo_root=repo_root,
                        rev=rev,
                        rel_path=relf,
                        format=format,
                        label=label,
                        include_token_count=annotate_tokens,
                        token_target=token_target,
                        ranges=ranges,
                        symbols=symbols,
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

    if link_depth > 0 and not use_rev:
        from .render.markitdown import MarkItDownConversionError

        try:
            refs[:], trace_items, skip_impact = add_markdown_link_refs(
                refs,
                link_depth=link_depth,
                scope=link_scope,
                format_=format,
                label=label,
                token_target=token_target,
                inject=inject,
                link_skip=link_skip,
                include_token_count=annotate_tokens,
            )
        except (MarkItDownConversionError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

    token_details = None
    breakdown = None
    max_tokens_budget = ctx.obj.get("max_tokens")
    need_token_details = bool(max_tokens_budget) or trace
    if need_token_details:
        budget_refs = []
        for ref in refs:
            path = getattr(ref, "path", None) or getattr(ref, "url", None)
            if not path:
                continue

            if getattr(ref, "file_content", None) is None and hasattr(ref, "output"):
                try:
                    _ = ref.output
                except Exception:
                    pass

            if (
                getattr(ref, "file_content", None) is None
                and getattr(ref, "original_file_content", None) is None
            ):
                continue

            budget_refs.append(ref)

        total_tokens, token_details = compute_input_token_details(
            budget_refs, token_target=token_target
        )

        if max_tokens_budget:
            breakdown = format_trace_output(
                budget_refs,
                [],
                common_prefix=None,
                stdin_data=None,
                token_target=token_target,
                input_token_details=token_details,
                sort_inputs_by_tokens=True,
            )
            ctx.obj["max_tokens_breakdown"] = breakdown
            ctx.obj["max_tokens_details"] = token_details
            ctx.obj["max_tokens_refs"] = budget_refs
            ctx.obj["max_tokens_total"] = total_tokens
            if total_tokens > max_tokens_budget:
                raise click.ClickException(
                    f"Token budget exceeded: {total_tokens} tokens > {max_tokens_budget} (target {token_target}).\n{breakdown}"
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
            ignored_files=ignored_files,
            ignored_folders=ignored_folders,
            token_target=token_target,
            input_token_details=token_details,
            sort_inputs_by_tokens=bool(token_details),
        )
        ctx.obj["trace_output"] = trace_output

    return result


@cli.command("paste")
@click.option(
    "--count",
    "-n",
    type=int,
    default=1,
    show_default=True,
    help="Number of clipboard captures to collect.",
)
@click.option(
    "-f",
    "--format",
    "format_hint",
    type=click.Choice(["md", "xml", "shell", "raw"], case_sensitive=False),
    default="md",
    help="Output format for clipboard entries (md/xml/shell/raw).",
)
@click.option(
    "--tokens",
    "annotate_tokens",
    is_flag=True,
    help="Annotate each captured label with a token count for its content.",
)
@click.pass_context
def paste_cmd(ctx, count, format_hint, annotate_tokens):
    """
    Capture clipboard contents in stages.
    """
    if count < 1:
        raise click.BadParameter("--count must be at least 1")

    ctx.obj["format"] = format_hint
    token_target = ctx.obj.get("token_target", "cl100k_base")

    from .utils import wait_for_enter

    def wait_for_stage_signal():
        try:
            import termios
            import tty

            fd = os.open("/dev/tty", os.O_RDONLY)
        except OSError:
            return "confirm" if wait_for_enter() else "interrupt"

        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = os.read(fd, 1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            os.close(fd)

        if key in (b"\r", b"\n"):
            return "confirm"
        if key == b"\x1b":
            return "undo"
        if key in (b"\x03", b"\x04"):
            return "interrupt"
        return "confirm"

    captured_segments = []
    idx = 1
    while idx <= count:
        prompt = (
            f"[{idx}/{count}] Press Enter after copying the next chunk..."
            if count > 1
            else "Press Enter once your clipboard is ready..."
        )
        click.echo(prompt + " ", nl=False)
        signal = wait_for_stage_signal()
        click.echo()

        if signal == "interrupt":
            click.echo("Clipboard capture interrupted.", err=True)
            ctx.exit(1)

        if signal == "undo":
            if captured_segments:
                captured_segments.pop()
                if idx > 1:
                    idx -= 1
                click.echo("Previous capture removed. Please copy again for this slot.")
            else:
                click.echo("No capture to undo.")
            click.echo()
            continue

        try:
            clipboard_text = clipboard_paste()
        except Exception as exc:
            raise click.ClickException(f"Unable to read from clipboard: {exc}") from exc

        label = "paste" if count == 1 else f"paste #{idx}"
        content_token_count = None
        if annotate_tokens:
            content_token_count = count_tokens(
                clipboard_text or "", target=token_target
            )["count"]
        processed = process_text(
            clipboard_text or "",
            format=format_hint,
            label=label,
            token_target=token_target,
            token_count=content_token_count,
            include_token_count=annotate_tokens,
        )
        captured_segments.append(processed)

        token_info = count_tokens(processed, target=token_target)
        token_count = token_info["count"]
        token_method = token_info["method"]
        prefix = f"[{idx}/{count}] " if count > 1 else ""
        click.echo(f"{prefix}Captured {token_count} tokens ({token_method}).")
        click.echo()

        idx += 1

    return "\n\n".join(captured_segments)


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
    type=click.Choice(["raw", "shell", "md", "xml"], case_sensitive=False),
    default="raw",
    help="Output format for the repo map (raw/shell/md/xml)",
)
@click.option(
    "--tokens",
    "annotate_tokens",
    is_flag=True,
    help="Annotate each file label with a token count for its map snippet.",
)
@click.option("--git-pull", is_flag=True, help="Pull cached git repos")
@click.option("--git-reclone", is_flag=True, help="Reclone cached git repos")
@click.option(
    "--rev",
    type=str,
    default=None,
    help="Generate the map from a git revision (e.g., HEAD)",
)
@click.pass_context
def map_cmd(
    ctx, paths, max_tokens, ignore, format, annotate_tokens, git_pull, git_reclone, rev
):
    """
    Generate a repository map.
    """
    if not paths:
        click.echo(ctx.get_help())
        ctx.exit()

    if max_tokens is not None and max_tokens <= 0:
        raise click.BadParameter("--max-tokens must be greater than 0")

    ctx.obj["format"] = format  # for segmentation

    from pathlib import Path

    from .render.map import (
        generate_repo_map_data,
        generate_repo_map_data_from_git,
    )
    from .git.cache import ensure_repo, expand_git_paths, parse_git_target

    token_target = ctx.obj.get("token_target", "cl100k_base")

    if rev:
        from .git.rev import discover_repo_root

        try:
            repo_root = discover_repo_root(paths, cwd=os.getcwd())
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        if not repo_root:
            raise click.ClickException("--rev requires running inside a git repository")
        # Treat provided paths as path specs relative to repo root if not absolute
        expanded: list[str] = []
        for p in paths:
            expanded_path = os.path.expanduser(p)
            if os.path.isabs(expanded_path) or expanded_path.startswith(".."):
                expanded.append(
                    os.path.relpath(os.path.abspath(expanded_path), repo_root)
                )
            else:
                expanded.append(expanded_path)
        internal_format = "raw" if format in {"md", "xml"} else format
        result = generate_repo_map_data_from_git(
            repo_root,
            expanded,
            rev,
            max_tokens,
            internal_format,
            ignore,
            annotate_tokens=annotate_tokens,
            token_target=token_target,
        )
    else:
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

        internal_format = "raw" if format in {"md", "xml"} else format
        result = generate_repo_map_data(
            expanded,
            max_tokens,
            internal_format,
            ignore,
            annotate_tokens=annotate_tokens,
            token_target=token_target,
        )
    if "error" in result:
        return result["error"]
    repo_map = result["repo_map"]
    if format in {"md", "xml"}:
        from .render.text import process_text

        label = " ".join(paths)
        repo_map = process_text(
            repo_map,
            format=format,
            label=label,
            xml_tag="map" if format == "xml" else None,
            token_target=token_target,
            include_token_count=annotate_tokens,
        )
    return repo_map


@cli.command("shell")
@click.argument("commands", nargs=-1, required=False)
@click.option(
    "-f",
    "--format",
    default="shell",
    help="Output format (md/xml/shell/raw). Defaults to shell.",
)
@click.option(
    "--capture-stderr/--no-capture-stderr",
    default=True,
    help="Capture stderr along with stdout. Defaults to True.",
)
@click.option(
    "--shell",
    "shell_executable",
    default=None,
    help="Shell executable for running commands (defaults to $SHELL).",
)
@click.pass_context
def shell_cmd(ctx, commands, format, capture_stderr, shell_executable):
    """
    Run arbitrary shell commands (returns raw combined output).
    """
    if not commands:
        click.echo(ctx.get_help())
        ctx.exit()
    ctx.obj["format"] = format  # for segmentation
    from .references import create_command_references

    refs_data = create_command_references(
        commands=commands,
        format=format,
        capture_stderr=capture_stderr,
        shell_executable=shell_executable,
    )
    return refs_data["concatenated"]


def main():
    cli()


if __name__ == "__main__":
    main()
