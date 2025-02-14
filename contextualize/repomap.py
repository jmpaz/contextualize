import os


def repomap_cmd(paths, max_tokens, output, fmt, output_file):
    """
    Generate a repository map from the provided paths. If `output_file` is specified,
    write to that file; otherwise print to stdout or copy to clipboard (if output='clipboard').
    """
    import click
    from aider.repomap import RepoMap, find_src_files
    from pyperclip import copy

    from contextualize.tokenize import count_tokens

    # Gather files
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(find_src_files(path))
        else:
            files.append(path)

    class ConsoleIO:
        def tool_output(self, msg):
            click.echo(msg)

        def tool_error(self, msg):
            click.echo(f"ERROR: {msg}", err=True)

        def tool_warning(self, msg):
            click.echo(f"WARNING: {msg}", err=True)

        def read_text(self, fname):
            try:
                with open(fname, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                self.tool_warning(f"Error reading file {fname}: {str(e)}")
                return ""

    class TokenCounter:
        def token_count(self, text):
            result = count_tokens(text, target="cl100k_base")
            return result["count"]

    io = ConsoleIO()
    main_model = TokenCounter()

    rm = RepoMap(map_tokens=max_tokens, main_model=main_model, io=io)
    repo_map = rm.get_repo_map(chat_files=[], other_files=files)

    if not repo_map:
        io.tool_error("No repository map was generated.")
        return

    if fmt == "shell":
        repo_map = f"❯ repo-map {' '.join(paths)}\n{repo_map}"

    token_info = count_tokens(repo_map, target="cl100k_base")
    num_files = len(files)
    summary_str = f"Map of {num_files} files ({token_info['count']} tokens) "

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(repo_map)
        click.echo(f"{summary_str}written to: {output_file}.")
    elif output == "clipboard":
        try:
            copy(repo_map)
            click.echo(f"{summary_str}copied to clipboard.")
        except Exception as e:
            click.echo(f"Error copying to clipboard: {e}", err=True)
    else:
        click.echo(repo_map)
