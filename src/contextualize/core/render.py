"""Output rendering utilities."""

from .formats import format_content


def process_text(
    text,
    clean=False,
    range=None,
    ranges=None,
    format="md",
    label="",
    label_suffix: str | None = None,
    xml_tag: str | None = None,
    shell_cmd=None,
    rev: str | None = None,
    token_target: str = "cl100k_base",
    token_count: int | None = None,
    include_token_count: bool = False,
    symbols=None,
):
    """Process text content with optional cleaning, range extraction, and formatting.

    Args:
        text: The text content to process
        clean: Whether to clean the text (convert spaces to tabs)
        range: Single range tuple (start, end) for line extraction
        ranges: List of range tuples for multi-region extraction
        format: Output format (raw, md, xml, shell)
        label: Label for the content
        label_suffix: Optional suffix to add to the label
        xml_tag: Tag name for XML format
        shell_cmd: Shell command for shell format
        rev: Git revision identifier
        token_target: Token counting target/encoding
        token_count: Pre-computed token count
        include_token_count: Whether to include token count in output
        symbols: List of symbol names

    Returns:
        Formatted text string
    """
    if clean:
        text = _clean(text)
    if range and not ranges:
        ranges = [range]
    if ranges:
        text = _extract_ranges(text, ranges)

    use_token_count = include_token_count and format in {"md", "xml", "shell"}

    if use_token_count:
        if token_count is None:
            from .utils import count_tokens

            token_count = count_tokens(text, target=token_target)["count"]
    else:
        token_count = None

    return format_content(
        text,
        fmt=format,
        label=label,
        label_suffix=label_suffix,
        xml_tag=xml_tag,
        shell_cmd=shell_cmd,
        rev=rev,
        token_count=token_count,
        symbols=symbols,
        is_excerpt=bool(ranges),
    )


def _clean(text):
    """Clean text by converting spaces to tabs."""
    return text.replace("    ", "\t")


def _extract_range(text, range_tuple):
    """Extract a single range from text."""
    start, end = range_tuple
    lines = text.split("\n")
    return "\n".join(lines[start - 1 : end])


def _extract_ranges(text, ranges):
    """Extract and merge multiple ranges from text.

    Adjacent or overlapping ranges are merged, and the resulting
    snippets are joined with '...' separators.
    """
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges, key=lambda r: r[0]):
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    lines = text.split("\n")
    snippets = []
    for start, end in merged:
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)
        snippet = "\n".join(lines[start_idx:end_idx])
        snippets.append(snippet)

    return "\n...\n".join(snippets)


# Legacy function for counting backticks (still used by some modules)
def _count_max_backticks(text):
    """Count the maximum consecutive backticks in text."""
    max_backticks = 0
    for line in text.split("\n"):
        stripped = line.lstrip("`")
        count = len(line) - len(stripped)
        if count > max_backticks:
            max_backticks = count
    return max_backticks
