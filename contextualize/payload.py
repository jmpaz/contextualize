import os
from typing import Any, Dict, List

import yaml

from .reference import create_file_references


def assemble_payload(
    components: List[Dict[str, Any]],
    base_dir: str,
) -> str:
    """
    - If a component has a 'text' key, emit that text verbatim.
    - Otherwise it must have 'name' and 'files':
        optional 'prefix' (above the attachment) and 'suffix' (below).
      Files and directories are expanded via create_file_references().
    """
    parts: List[str] = []

    for comp in components:
        # 1) text‐only
        if "text" in comp:
            text = comp["text"].rstrip()
            parts.append(text)
            continue

        # 2) file component
        name = comp.get("name")
        files = comp.get("files")
        if not name or not files:
            raise ValueError(
                f"Component must have either 'text' or both 'name' & 'files': {comp}"
            )

        prefix = comp.get("prefix", "").rstrip()
        suffix = comp.get("suffix", "").lstrip()

        # collect FileReference objects (recursing into directories)
        all_refs = []
        for rel in files:
            full = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
            if not os.path.exists(full):
                raise FileNotFoundError(f"Component '{name}' path not found: {full}")
            refs = create_file_references(
                [full], ignore_paths=None, format="md", label="relative"
            )["refs"]
            all_refs.extend(refs)

        # build block
        block: List[str] = []
        if prefix:
            block.append(prefix)

        block.append(f'<attachment label="{name}">')
        for idx, ref in enumerate(all_refs):
            block.append(ref.output)
            if idx < len(all_refs) - 1:
                block.append("")
        block.append("</attachment>")

        if suffix:
            block.append(suffix)

        parts.append("\n".join(block))

    return "\n\n".join(parts)


def render_from_yaml(
    manifest_path: str,
) -> str:
    """
    Load YAML with top-level:
      config:
        root:  # optional, expands ~
      components:
        - text: ...
        - name: ...; prefix/suffix?; files: [...]
    """
    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping with 'config' and 'components'")

    cfg = data.get("config", {})
    if "root" in cfg:
        raw = cfg["root"] or "~"
        base_dir = os.path.expanduser(raw)
    else:
        base_dir = os.path.dirname(os.path.abspath(manifest_path))

    comps = data.get("components")
    if not isinstance(comps, list):
        raise ValueError("'components' must be a list")

    return assemble_payload(comps, base_dir)
