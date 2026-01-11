"""Markdown format handler."""


def _count_max_backticks(text: str) -> int:
    """Count the maximum consecutive backticks in text."""
    max_backticks = 0
    for line in text.split("\n"):
        stripped = line.lstrip("`")
        count = len(line) - len(stripped)
        if count > max_backticks:
            max_backticks = count
    return max_backticks


def wrap_md(
    content: str,
    label: str | None = None,
    *,
    rev: str | None = None,
    token_count: int | None = None,
    label_suffix: str | None = None,
    symbols: list[str] | None = None,
    is_excerpt: bool = False,
    **kwargs,
) -> str:
    """Wrap content in a Markdown code fence.

    Args:
        content: The content to wrap
        label: Optional label for the code fence info string
        rev: Optional revision identifier
        token_count: Optional token count to display
        label_suffix: Optional suffix to add to the label
        symbols: Optional list of symbol names
        is_excerpt: Whether this is an excerpt (unused, for interface compat)
        **kwargs: Additional options (ignored)

    Returns:
        Content wrapped in markdown code fence
    """
    max_backticks = _count_max_backticks(content)
    backticks_str = "`" * max(max_backticks + 2, 3)

    symbols_list = [s for s in (symbols or []) if s]
    sym_suffix = f":{','.join(symbols_list)}" if symbols_list else ""
    label_with_symbols = f"{label or ''}{sym_suffix}"
    if label_suffix:
        label_with_symbols = f"{label_with_symbols} {label_suffix}"

    info = f"{label_with_symbols}@{rev}" if rev else label_with_symbols
    if token_count is not None:
        info = f"{info} ({token_count} tokens)"

    return f"{backticks_str}{info}\n{content}\n{backticks_str}"
