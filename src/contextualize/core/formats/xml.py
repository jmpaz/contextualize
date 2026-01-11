"""XML format handler."""


def wrap_xml(
    content: str,
    label: str | None = None,
    *,
    xml_tag: str | None = None,
    rev: str | None = None,
    token_count: int | None = None,
    label_suffix: str | None = None,
    symbols: list[str] | None = None,
    is_excerpt: bool = False,
    **kwargs,
) -> str:
    """Wrap content in XML tags.

    Args:
        content: The content to wrap
        label: Path or label for the path attribute
        xml_tag: Tag name to use (defaults to "file")
        rev: Optional revision identifier
        token_count: Optional token count to display
        label_suffix: Optional suffix to add as attribute
        symbols: Optional list of symbol names
        is_excerpt: Whether this is an excerpt (unused, for interface compat)
        **kwargs: Additional options (ignored)

    Returns:
        Content wrapped in XML tags
    """
    tag_name = xml_tag or "file"
    symbols_list = [s for s in (symbols or []) if s]

    path_attr = f" path='{label}'" if label else ""
    symbols_attr = f" symbols='{','.join(symbols_list)}'" if symbols_list else ""
    token_attr = f" token_count='{token_count}'" if token_count is not None else ""
    rev_attr = f" rev='{rev}'" if rev else ""
    suffix_attr = f" {label_suffix}" if label_suffix else ""

    return (
        f"<{tag_name}{path_attr}{symbols_attr}{token_attr}{rev_attr}{suffix_attr}>\n"
        f"{content}\n"
        f"</{tag_name}>"
    )
