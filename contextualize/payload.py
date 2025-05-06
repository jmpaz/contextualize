import os
from typing import Any, Dict, List

import yaml

from .reference import FileReference


def assemble_payload(
    components: List[Dict[str, Any]],
    base_dir: str,
) -> str:
    """
    Given a list of components each with:
      - name: str
      - prompt: str
      - files: List[str]

    Return a string of:
      prompt
      <attachment label="name">
      ```path/to/file1
      <contents>
      ```

      ```path/to/file2
      <contents>
      ```
      </attachment>
    joined by blank lines.
    """
    parts: List[str] = []

    for comp in components:
        name = comp.get("name")
        prompt = comp.get("prompt", "").rstrip()
        files = comp.get("files")

        if not name or files is None:
            raise ValueError(f"Each component needs 'name' and 'files': {comp}")

        block: List[str] = []
        if prompt:
            block.append(prompt)

        block.append(f'<attachment label="{name}">')

        for i, relpath in enumerate(files):
            fullpath = (
                relpath if os.path.isabs(relpath) else os.path.join(base_dir, relpath)
            )
            if not os.path.exists(fullpath):
                raise FileNotFoundError(
                    f"Component '{name}' file not found: {fullpath}"
                )

            # produce fenced block with the filename as info string
            ref = FileReference(fullpath, format="md", label="relative")
            block.append(ref.output)

            # blank line
            if i < len(files) - 1:
                block.append("")

        block.append("</attachment>")
        parts.append("\n".join(block))

    return "\n\n".join(parts)


def render_from_yaml(
    manifest_path: str,
) -> str:
    """
    Load a YAML manifest of the form:

      config:
        root: /some/path   # optional; expands ~, defaults to manifest folder
      components:
        - name: …
          prompt: …
          files: [ … ]

    and return the assembled payload as a single Markdown string.
    """
    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping with 'config' and 'components'")

    # Determine base_dir: either config.root or the manifest’s directory
    cfg = data.get("config", {})
    root = cfg.get("root")
    if root:
        base_dir = os.path.expanduser(root)
    else:
        base_dir = os.path.dirname(os.path.abspath(manifest_path))

    comps = data.get("components")
    if not isinstance(comps, list):
        raise ValueError("'components' must be a list of items")

    return assemble_payload(comps, base_dir)
