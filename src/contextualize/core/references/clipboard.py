"""Clipboard reference implementation."""

from ..render import process_text
from ..utils import count_tokens


class ClipboardReference:
    """Reference to content captured from the clipboard."""

    def __init__(
        self,
        content: str,
        *,
        format: str = "md",
        label: str = "clipboard",
        label_suffix: str | None = None,
        include_token_count: bool = False,
        token_target: str = "cl100k_base",
    ):
        self._content = content
        self.format = format
        self._label = label
        self.label_suffix = label_suffix
        self.include_token_count = include_token_count
        self.token_target = token_target
        self._output = self._get_contents()

    @property
    def path(self) -> str:
        return "clipboard"

    @property
    def file_content(self) -> str:
        return self._content

    @property
    def output(self) -> str:
        return self._output

    @property
    def label(self) -> str:
        return self._label

    def read(self) -> str:
        """Read and return the clipboard content."""
        return self._content

    def exists(self) -> bool:
        """Check if content exists."""
        return bool(self._content)

    def token_count(self, encoding: str = "cl100k_base") -> int:
        """Count tokens in the content."""
        return count_tokens(self._content, target=encoding)["count"]

    def _get_contents(self) -> str:
        return process_text(
            self._content,
            format=self.format,
            label=self._label,
            label_suffix=self.label_suffix,
            token_target=self.token_target,
            include_token_count=self.include_token_count,
        )


def get_clipboard_content() -> str:
    """Get content from the system clipboard.

    Returns empty string if clipboard is unavailable or empty.
    """
    try:
        import pyperclip

        return pyperclip.paste() or ""
    except Exception:
        return ""


def create_clipboard_reference(
    format: str = "md",
    label: str = "clipboard",
    include_token_count: bool = False,
    token_target: str = "cl100k_base",
) -> ClipboardReference:
    """Create a ClipboardReference from the current clipboard content."""
    content = get_clipboard_content()
    return ClipboardReference(
        content,
        format=format,
        label=label,
        include_token_count=include_token_count,
        token_target=token_target,
    )
