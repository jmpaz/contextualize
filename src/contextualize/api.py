"""Public entry points for embedding contextualize in other tools.

`resolve_refs` is a thin, stable wrapper over the same resolution the `cat`
command uses: it turns targets (files, URLs, plugin refs) into documents that
carry their authored `prose`. Consumers (e.g. an analyzer plugin) call this
rather than reaching into internal resolution functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .manifest.edition import (
    AuthoredEdition,
    compile_authored_context,
    compile_authored_manifest,
    compile_authored_registry,
)


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
    get_label = getattr(ref, "get_label", None)
    label = None
    if callable(get_label):
        try:
            label = get_label()
        except Exception:
            label = None
    if not (isinstance(label, str) and label):
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


def resolve_refs(
    targets, *, describe_media: bool = True, **kwargs
) -> list[ResolvedRef]:
    """Resolve targets to documents carrying prose.

    Each item exposes `.source`, `.label`, `.content`, `.prose` (str | None),
    `.prose_authors`, and `.metadata`. Plain files carry `prose=None`; the caller
    decides how to treat undeclared prose. Extra keyword arguments are forwarded
    to the underlying resolver.

    `describe_media=False` resolves for text only: audio/video transcription
    still runs (it is authored prose), but the LLM image/frame/media
    *descriptions* are suppressed across providers and the raw-image path. A
    consumer that only scores prose (e.g. an analyzer) passes this to avoid
    paying for descriptions it discards; a later `cat` of the same target still
    describes, and the cached transcript is reused rather than recomputed.
    """
    from .references.factory import create_file_references
    from .runtime import reset_describe_media, set_describe_media

    if isinstance(targets, str):
        targets = [targets]
    token = set_describe_media(describe_media)
    try:
        result = create_file_references(
            list(targets), describe_media=describe_media, **kwargs
        )
    finally:
        reset_describe_media(token)
    refs = result["refs"] if isinstance(result, dict) else result
    return [_to_resolved_ref(ref) for ref in refs]
