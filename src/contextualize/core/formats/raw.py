"""Raw format handler - no wrapping."""


def wrap_raw(content: str, label: str | None = None, **kwargs) -> str:
    """Return content as-is without any wrapping.

    Args:
        content: The content to format
        label: Ignored for raw format
        **kwargs: Additional options (ignored)

    Returns:
        The content unchanged
    """
    return content
