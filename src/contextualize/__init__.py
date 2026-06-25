"""contextualize package."""

from .api import ResolvedRef, resolve_refs

__all__: list[str] = ["resolve_refs", "ResolvedRef"]
