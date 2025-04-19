"""
payload: assemble context payloads from a YAML manifest.

Usage:
    from contextualize.payload import render_payload, render_from_yaml

    # manual components:
    components = [
      {"label": "description", "source": {"type": "file", "path": "data/desc.txt"}},
      {"label": "notes",       "source": {"type": "inline", "content": "…"}}
    ]
    payload = render_payload(components)

    # or load directly from YAML:
    s = render_from_yaml("manifest.yaml")
"""

import os
from typing import Any, Dict, List

import yaml

from .utils import wrap_text


def assemble_payload(
    components: List[Dict[str, Any]],
    *,
    base_dir: str | None = None,
) -> str:
    """
    Given a list of {"label": str, "source": {"type": "file"|"inline", ...}},
    return a string where each component is:

      label:
      ```
      <content>
      ```

    joined by blank lines.  (No outer <paste> wrapper here.)
    """
    base = base_dir or os.getcwd()
    parts: List[str] = []

    for comp in components:
        label = comp["label"]
        src = comp["source"]
        kind = src["type"]

        if kind == "file":
            rel = src["path"]
            path = rel if os.path.isabs(rel) else os.path.join(base, rel)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Component '{label}' file not found: {path}")
            with open(path, "r", encoding="utf-8") as fh:
                body = fh.read()
        elif kind == "inline":
            body = src.get("content", "")
        else:
            raise ValueError(f"Unsupported source type for '{label}': {kind}")

        fenced = wrap_text(body, "md")
        parts.append(f"{label}:\n{fenced}")

    return "\n\n".join(parts)


def render_from_yaml(
    manifest_path: str,
    *,
    base_dir: str | None = None,
) -> str:
    """
    Load a YAML at `manifest_path` containing:

      components:
        - label: "..."
          source:
            type: file|inline
            path: ...       # if file
            content: "..."  # if inline

    and return assemble_payload(manifest['components']).
    """
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)

    comps = manifest.get("components")
    if not isinstance(comps, list):
        raise ValueError(
            f"YAML '{manifest_path}' must have a top‑level list under 'components'"
        )

    bd = base_dir or os.path.dirname(os.path.abspath(manifest_path))
    return assemble_payload(comps, base_dir=bd)
