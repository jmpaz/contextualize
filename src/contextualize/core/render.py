def process_text(
    text,
    clean=False,
    range=None,
    ranges=None,
    format="md",
    label="",
    shell_cmd=None,
    rev: str | None = None,
    token_target: str = "cl100k_base",
    token_count: int | None = None,
    include_token_count: bool = False,
    symbols=None,
):
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
    max_backticks = _count_max_backticks(text)
    return _delimit(
        text,
        format,
        label,
        max_backticks,
        shell_cmd,
        rev,
        token_count,
        symbols=symbols,
        is_excerpt=bool(ranges),
    )


def _clean(text):
    # Example cleaning logic
    return text.replace("    ", "\t")


def _extract_range(text, range_tuple):
    start, end = range_tuple
    lines = text.split("\n")
    return "\n".join(lines[start - 1 : end])


def _extract_ranges(text, ranges):
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


def _count_max_backticks(text):
    max_backticks = 0
    for line in text.split("\n"):
        # If a line starts with backticks, count them
        stripped = line.lstrip("`")
        count = len(line) - len(stripped)
        if count > max_backticks:
            max_backticks = count
    return max_backticks


def _delimit(
    text,
    format,
    label,
    max_backticks=0,
    shell_cmd=None,
    rev: str | None = None,
    token_count: int | None = None,
    *,
    symbols=None,
    is_excerpt=False,
):
    symbols_list = [s for s in (symbols or []) if s]
    sym_suffix = f"::{','.join(symbols_list)}" if symbols_list else ""
    label_with_symbols = f"{label}{sym_suffix}"

    if format == "md":
        backticks_str = "`" * max(max_backticks + 2, 3)  # at least 3
        info = f"{label_with_symbols}@{rev}" if rev else label_with_symbols
        if token_count is not None:
            info = f"{info} ({token_count} tokens)"
        return f"{backticks_str}{info}\n{text}\n{backticks_str}"
    elif format == "xml":
        token_attr = f" token_count='{token_count}'" if token_count is not None else ""
        rev_attr = f" rev='{rev}'" if rev else ""
        symbols_attr = f" symbols='{','.join(symbols_list)}'" if symbols_list else ""
        return (
            f"<file path='{label}'{symbols_attr}{token_attr}{rev_attr}>\n"
            f"{text}\n"
            f"</file>"
        )
    elif format == "shell":
        target_label = f"{label_with_symbols}@{rev}" if rev else label_with_symbols
        token_suffix = f" ({token_count} tokens)" if token_count is not None else ""
        if shell_cmd:
            return f"❯ {shell_cmd}{token_suffix}\n{text}"
        return f"❯ cat {target_label}{token_suffix}\n{text}"
    else:
        return text
