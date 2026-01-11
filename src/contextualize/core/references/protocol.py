"""Reference protocol and base types."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reference(Protocol):
    """Protocol for content references.

    All reference types must implement this interface to be usable
    in the rendering pipeline.
    """

    @property
    def label(self) -> str:
        """Human-readable label for this reference."""
        ...

    @property
    def path(self) -> str:
        """Path or URL identifying this reference."""
        ...

    @property
    def output(self) -> str:
        """Formatted output content."""
        ...

    @property
    def file_content(self) -> str:
        """Raw content without formatting."""
        ...

    def read(self) -> str:
        """Read and return the raw content."""
        ...

    def exists(self) -> bool:
        """Check if the referenced content exists."""
        ...

    def token_count(self, encoding: str = "cl100k_base") -> int:
        """Count tokens in the content."""
        ...
