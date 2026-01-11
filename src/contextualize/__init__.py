from pathlib import Path


def cat(
    paths: list[str],
    *,
    format: str = "md",
    label: str = "relative",
    ignore: list[str] | None = None,
    tokens: bool = False,
    token_target: str = "cl100k_base",
    inject: bool = False,
    depth: int = 5,
) -> str:
    from .references import create_file_references

    result = create_file_references(
        paths,
        ignore_patterns=ignore,
        format=format,
        label=label,
        include_token_count=tokens,
        token_target=token_target,
        inject=inject,
        depth=depth,
    )
    return result["concatenated"]


def map_paths(
    paths: list[str],
    *,
    max_tokens: int = 10000,
    format: str = "raw",
    ignore: list[str] | None = None,
    token_target: str = "cl100k_base",
) -> str:
    from .render.map import generate_repo_map_data

    result = generate_repo_map_data(
        paths,
        max_tokens=max_tokens,
        fmt=format,
        ignore=ignore,
        token_target=token_target,
    )
    return result.get("repo_map", "")


def shell(
    commands: list[str],
    *,
    format: str = "shell",
    capture_stderr: bool = True,
    shell_executable: str | None = None,
) -> str:
    from .references import create_command_references

    result = create_command_references(
        commands,
        format=format,
        capture_stderr=capture_stderr,
        shell_executable=shell_executable,
    )
    return result["concatenated"]


def paste(
    content: str | None = None,
    *,
    format: str = "md",
    annotation: str | None = None,
    tokens: bool = False,
    token_target: str = "cl100k_base",
) -> str:
    from .render.text import process_text

    if content is None:
        try:
            import pyperclip

            content = pyperclip.paste()
        except Exception:
            content = ""

    return process_text(
        content,
        format=format,
        label=annotation or "",
        include_token_count=tokens,
        token_target=token_target,
    )


def payload(
    manifest_path: str | Path,
    *,
    inject: bool = False,
    depth: int | None = None,
    exclude: list[str] | None = None,
    include: list[str] | None = None,
    map_mode: bool = False,
    token_target: str = "cl100k_base",
) -> str:
    from .manifest.payload import render_manifest

    result = render_manifest(
        manifest_path,
        inject=inject,
        depth=depth,
        exclude_keys=exclude,
        include_keys=include,
        map_mode=map_mode,
        token_target=token_target,
    )
    return result.payload


def hydrate(
    manifest_path: str | Path,
    *,
    dir: str | Path = ".context",
    access: str = "writable",
    path_strategy: str = "on-disk",
    cwd: str | None = None,
) -> Path:
    import os
    from .manifest.hydrate import HydrateOverrides, hydrate_manifest

    overrides = HydrateOverrides(
        context_dir=str(dir),
        access=access,
        path_strategy=path_strategy,
    )
    result = hydrate_manifest(
        str(manifest_path),
        overrides=overrides,
        cwd=cwd or os.getcwd(),
    )
    return Path(result.context_dir)


__all__ = [
    "cat",
    "map_paths",
    "shell",
    "paste",
    "payload",
    "hydrate",
]
