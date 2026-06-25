"""Public entry points for embedding contextualize in other tools.

`resolve_refs` is a thin, stable wrapper over the same resolution the `cat`
command uses: it turns targets (files, URLs, plugin refs) into documents that
carry their authored `prose`. Consumers (e.g. an analyzer plugin) call this
rather than reaching into internal resolution functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResolvedRef:
    source: str
    label: str
    content: str
    prose: str | None
    prose_authors: list[str]
    metadata: dict[str, Any]


def _to_resolved_ref(ref: Any) -> ResolvedRef:
    document = getattr(ref, "document", None)
    if document is not None:
        source = getattr(ref, "source", "") or ""
        return ResolvedRef(
            source=source,
            label=getattr(document, "label", "") or source,
            content=getattr(document, "content", "") or "",
            prose=getattr(document, "prose", None),
            prose_authors=list(getattr(document, "prose_authors", []) or []),
            metadata=dict(getattr(document, "metadata", {}) or {}),
        )
    content = (
        getattr(ref, "original_file_content", None)
        or getattr(ref, "file_content", None)
        or ""
    )
    source = getattr(ref, "path", None) or getattr(ref, "source", "") or ""
    label = getattr(ref, "label", None)
    label = label if isinstance(label, str) and label else source
    return ResolvedRef(
        source=source,
        label=label,
        content=content,
        prose=None,
        prose_authors=[],
        metadata={},
    )


def resolve_refs(targets, **kwargs) -> list[ResolvedRef]:
    """Resolve targets to documents carrying prose.

    Each item exposes `.source`, `.label`, `.content`, `.prose` (str | None),
    `.prose_authors`, and `.metadata`. Plain files carry `prose=None`; the caller
    decides how to treat undeclared prose. Extra keyword arguments are forwarded
    to the underlying resolver.
    """
    from .references.factory import create_file_references

    if isinstance(targets, str):
        targets = [targets]
    result = create_file_references(list(targets), **kwargs)
    refs = result["refs"] if isinstance(result, dict) else result
    return [_to_resolved_ref(ref) for ref in refs]
