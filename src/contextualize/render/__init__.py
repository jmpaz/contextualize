import importlib

from .formats import FORMATS, format_content, register_format, supported_formats
from .text import process_text, render_map, render_references

_LAZY_EXPORTS = {
    "generate_repo_map_data": ("map", "generate_repo_map_data"),
    "generate_repo_map_data_from_git": ("map", "generate_repo_map_data_from_git"),
    "add_markdown_link_refs": ("links", "add_markdown_link_refs"),
    "compute_input_token_details": ("trace", "compute_input_token_details"),
    "count_downstream": ("links", "count_downstream"),
    "format_trace_output": ("trace", "format_trace_output"),
    "inject_content_in_text": ("inject", "inject_content_in_text"),
    "MarkItDownConversionError": ("markitdown", "MarkItDownConversionError"),
    "MarkItDownResult": ("markitdown", "MarkItDownResult"),
    "convert_path_to_markdown": ("markitdown", "convert_path_to_markdown"),
    "convert_response_to_markdown": ("markitdown", "convert_response_to_markdown"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, attr_name)


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
