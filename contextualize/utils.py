import os
import re


def get_config_path(custom_path=None):
    if custom_path:
        return custom_path
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return os.path.join(xdg_config_home, "contextualize", "config.yaml")


def read_config(custom_path=None):
    import yaml

    config_path = get_config_path(custom_path)
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}


def wrap_text(content: str, wrap_mode: str, filename: str | None = None) -> str:
    """
    Wrap the given content according to wrap_mode ('xml' or 'md').
    If wrap_mode is None or empty string, return content unmodified.
    For 'md' mode, filename can be included in the code fence.
    """
    if not wrap_mode:
        return content

    if wrap_mode == "xml":
        return f"<paste>\n{content}\n</paste>"

    if wrap_mode in ("md", "markdown"):
        backtick_runs = re.findall(r"`+", content)
        longest = max(len(run) for run in backtick_runs) if backtick_runs else 0

        fence_len = longest + 2 if longest >= 3 else 3
        fence = "`" * fence_len

        fence_header = fence + (filename if filename else "")

        return f"{fence_header}\n{content}\n{fence}"

    return content


def add_prompt_wrappers(content, prompts):
    """
    If no prompt strings are provided, return content unchanged.
    If one prompt string is provided, prepend it (without an extra blank line).
    If two prompt strings are provided, prepend the first and append the second,
    with a single blank line separating the appended prompt.
    """
    if not prompts:
        return content
    if len(prompts) == 1:
        return f"{prompts[0]}\n{content}"
    else:
        return f"{prompts[0]}\n{content}\n\n{prompts[1]}"


def segment_output(text, max_tokens, format_hint):
    """Split text into segments without breaking files."""
    import re

    from .tokenize import count_tokens

    if format_hint == "xml":
        files = re.findall(r"<file path='[^']*'>.*?</file>", text, re.DOTALL)
        remaining = re.sub(
            r"<file path='[^']*'>.*?</file>", "|||FILE|||", text, flags=re.DOTALL
        )
        parts = remaining.split("|||FILE|||")
        result = []
        for i, part in enumerate(parts):
            if part.strip():
                result.append(part)
            if i < len(files):
                result.append(files[i])
        files = result
    elif format_hint in ("md", "markdown", "plain"):
        pattern = r"(^```[^\n]*\n.*?\n```$)"
        files = re.split(pattern, text, flags=re.MULTILINE | re.DOTALL)
        files = [f for f in files if f.strip()]
    elif format_hint == "shell":
        files = re.split(r"(?=^❯ )", text, flags=re.MULTILINE)
        files = [f for f in files if f.strip()]
    else:
        files = text.split("\n\n")

    segments = []
    current = []
    current_tokens = 0

    for file_content in files:
        tokens = count_tokens(file_content, target="cl100k_base")["count"]

        if tokens > max_tokens:
            if current:
                segments.append(("\n\n".join(current), current_tokens))
                current, current_tokens = [], 0
            segments.append((file_content, tokens))
        elif current_tokens + tokens + (100 if current else 0) > max_tokens:
            if current:
                segments.append(("\n\n".join(current), current_tokens))
            current, current_tokens = [file_content], tokens
        else:
            current.append(file_content)
            current_tokens += tokens + (100 if len(current) > 1 else 0)

    if current:
        segments.append(("\n\n".join(current), current_tokens))

    return segments


def wait_for_enter():
    """Wait for Enter key press, return False if interrupted."""
    try:
        with open("/dev/tty", "r") as tty:
            tty.readline()
        return True

    except (KeyboardInterrupt, EOFError, OSError):
        return False


def build_segment(text, wrap_mode, prompts, output_pos, index, total):
    """Build a segment with appropriate wrapping and prompts."""
    wrapped = wrap_text(text, wrap_mode)

    if not prompts:
        return wrapped

    is_first = index == 1
    is_last = index == total

    if len(prompts) == 1:
        if output_pos == "append" and is_last:
            return f"{wrapped}\n{prompts[0]}"
        elif output_pos != "append" and is_first:
            return f"{prompts[0]}\n{wrapped}"
    elif len(prompts) == 2:
        prefix = f"{prompts[0]}\n" if is_first else ""
        suffix = f"\n\n{prompts[1]}" if is_last else ""
        return f"{prefix}{wrapped}{suffix}"

    return wrapped


def _split_brace_options(s: str) -> list[str]:
    """Split comma-separated options within braces, handling nested braces."""
    opts = []
    buf = ""
    depth = 0
    for ch in s:
        if ch == "," and depth == 0:
            opts.append(buf)
            buf = ""
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        buf += ch
    opts.append(buf)
    return opts


def brace_expand(pattern: str) -> list[str]:
    """Expand shell-style brace patterns like {a,b,c} into multiple strings."""
    m = re.search(r"\{", pattern)
    if not m:
        return [pattern]
    start = m.start()
    depth = 0
    end = start
    for i in range(start, len(pattern)):
        if pattern[i] == "{":
            depth += 1
        elif pattern[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    inside = pattern[start + 1 : end]
    rest = pattern[end + 1 :]
    prefix = pattern[:start]
    out = []
    for opt in _split_brace_options(inside):
        for expanded in brace_expand(opt + rest):
            out.append(prefix + expanded)
    return out
