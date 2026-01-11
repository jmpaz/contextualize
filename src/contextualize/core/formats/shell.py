"""Shell format handler."""


def wrap_shell(
    content: str,
    label: str | None = None,
    *,
    shell_cmd: str | None = None,
    rev: str | None = None,
    token_count: int | None = None,
    label_suffix: str | None = None,
    symbols: list[str] | None = None,
    is_excerpt: bool = False,
    **kwargs,
) -> str:
    """Wrap content in shell-style output format.

    Args:
        content: The content to wrap
        label: Path or label to display
        shell_cmd: Actual shell command (if different from label)
        rev: Optional revision identifier
        token_count: Optional token count to display
        label_suffix: Optional suffix to add to the label
        symbols: Optional list of symbol names
        is_excerpt: Whether this is an excerpt (unused, for interface compat)
        **kwargs: Additional options (ignored)

    Returns:
        Content with shell-style header
    """
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
