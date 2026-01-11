from .formats import FORMATS, format_content, register_format, supported_formats
from .text import process_text, render_map, render_references


def __getattr__(name):
    if name == "generate_repo_map_data":
        from .map import generate_repo_map_data

        return generate_repo_map_data
    if name == "generate_repo_map_data_from_git":
        from .map import generate_repo_map_data_from_git

        return generate_repo_map_data_from_git
    if name == "add_markdown_link_refs":
        from .links import add_markdown_link_refs

        return add_markdown_link_refs
    if name == "compute_input_token_details":
        from .trace import compute_input_token_details

        return compute_input_token_details
    if name == "count_downstream":
        from .links import count_downstream

        return count_downstream
    if name == "format_trace_output":
        from .trace import format_trace_output

        return format_trace_output
    if name == "inject_content_in_text":
        from .inject import inject_content_in_text

        return inject_content_in_text
    if name == "MarkItDownConversionError":
        from .markitdown import MarkItDownConversionError

        return MarkItDownConversionError
    if name == "MarkItDownResult":
        from .markitdown import MarkItDownResult

        return MarkItDownResult
    if name == "convert_path_to_markdown":
        from .markitdown import convert_path_to_markdown

        return convert_path_to_markdown
    if name == "convert_response_to_markdown":
        from .markitdown import convert_response_to_markdown

        return convert_response_to_markdown
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FORMATS",
    "format_content",
    "register_format",
    "supported_formats",
    "generate_repo_map_data",
    "generate_repo_map_data_from_git",
    "process_text",
    "render_map",
    "render_references",
    "add_markdown_link_refs",
    "compute_input_token_details",
    "count_downstream",
    "format_trace_output",
    "inject_content_in_text",
    "MarkItDownConversionError",
    "MarkItDownResult",
    "convert_path_to_markdown",
    "convert_response_to_markdown",
]
