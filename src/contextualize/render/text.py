from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..references import Reference
from ..utils import extract_ranges
from .formats import format_content, supported_formats


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
    if clean:
        text = _clean(text)
    if range and not ranges:
        ranges = [range]
    if ranges:
        text = extract_ranges(text, ranges)
    use_token_count = include_token_count and format in {"md", "xml", "shell"}

    if use_token_count:
        if token_count is None:
            from ..utils import count_tokens

            token_count = count_tokens(text, target=token_target)["count"]
    else:
        token_count = None
    if format not in supported_formats():
        return text
    return format_content(
        text,
        fmt=format,
        label=label,
        rev=rev,
        token_count=token_count,
        label_suffix=label_suffix,
        tag_name=xml_tag,
        shell_cmd=shell_cmd,
        symbols=symbols,
    )


def _clean(text):
    return text.replace("    ", "\t")


def render_references(
    refs: list["Reference"],
    format: str = "raw",
    tokens: bool = False,
    label_style: str = "relative",
    separator: str = "\n\n",
) -> str:
    outputs = []
    for ref in refs:
        outputs.append(ref.output)
    return separator.join(outputs)


def render_map(
    refs: list["Reference"],
    max_tokens: int = 10000,
    format: str = "raw",
) -> str:
    from .map import RepoMap

    paths = [ref.path for ref in refs if hasattr(ref, "path")]
    if not paths:
        return ""

    repo_map = RepoMap(max_map_tokens=max_tokens)
    return repo_map.get_map(paths)
