from typing import Callable

FormatHandler = Callable[..., str]


def _count_max_backticks(text: str) -> int:
    max_backticks = 0
    for line in text.split("\n"):
        stripped = line.lstrip("`")
        count = len(line) - len(stripped)
        if count > max_backticks:
            max_backticks = count
    return max_backticks


def wrap_raw(content: str, label: str | None = None, **kwargs) -> str:
    return content


def wrap_md(
    content: str,
    label: str | None = None,
    *,
    rev: str | None = None,
    token_count: int | None = None,
    label_suffix: str | None = None,
    symbols: list[str] | None = None,
    **kwargs,
) -> str:
    symbols_list = [s for s in (symbols or []) if s]
    sym_suffix = f":{','.join(symbols_list)}" if symbols_list else ""
    label_with_symbols = f"{label or ''}{sym_suffix}"
    if label_suffix:
        label_with_symbols = f"{label_with_symbols} {label_suffix}"

    max_backticks = _count_max_backticks(content)
    backticks_str = "`" * max(max_backticks + 2, 3)
    info = f"{label_with_symbols}@{rev}" if rev else label_with_symbols
    if token_count is not None:
        info = f"{info} ({token_count} tokens)"
    return f"{backticks_str}{info}\n{content}\n{backticks_str}"


def wrap_xml(
    content: str,
    label: str | None = None,
    *,
    tag_name: str | None = None,
    rev: str | None = None,
    token_count: int | None = None,
    label_suffix: str | None = None,
    symbols: list[str] | None = None,
    **kwargs,
) -> str:
    symbols_list = [s for s in (symbols or []) if s]
    tag = tag_name or "file"

    token_attr = f" token_count='{token_count}'" if token_count is not None else ""
    rev_attr = f" rev='{rev}'" if rev else ""
    symbols_attr = f" symbols='{','.join(symbols_list)}'" if symbols_list else ""
    suffix_attr = f" {label_suffix}" if label_suffix else ""

    return (
        f"<{tag} path='{label or ''}'{symbols_attr}{token_attr}{rev_attr}{suffix_attr}>\n"
        f"{content}\n"
        f"</{tag}>"
    )


def wrap_shell(
    content: str,
    label: str | None = None,
    *,
    shell_cmd: str | None = None,
    rev: str | None = None,
    token_count: int | None = None,
    label_suffix: str | None = None,
    symbols: list[str] | None = None,
    **kwargs,
) -> str:
    symbols_list = [s for s in (symbols or []) if s]
    sym_suffix = f":{','.join(symbols_list)}" if symbols_list else ""
    label_with_symbols = f"{label or ''}{sym_suffix}"
    if label_suffix:
        label_with_symbols = f"{label_with_symbols} {label_suffix}"

    target_label = f"{label_with_symbols}@{rev}" if rev else label_with_symbols
    token_suffix = f" ({token_count} tokens)" if token_count is not None else ""

    if shell_cmd:
        return f"❯ {shell_cmd}{token_suffix}\n{content}"
    return f"❯ cat {target_label}{token_suffix}\n{content}"


FORMATS: dict[str, FormatHandler] = {
    "raw": wrap_raw,
    "md": wrap_md,
    "xml": wrap_xml,
    "shell": wrap_shell,
}


def register_format(name: str, handler: FormatHandler) -> None:
    FORMATS[name] = handler


def format_content(
    content: str,
    fmt: str = "raw",
    label: str | None = None,
    **kwargs,
) -> str:
    handler = FORMATS.get(fmt)
    if handler is None:
        raise ValueError(f"Unknown format: {fmt}")
    return handler(content, label, **kwargs)


def supported_formats() -> list[str]:
    return list(FORMATS.keys())
