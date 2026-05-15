# Plugins

`contextualize` discovers plugins from installed Python packages through the
`contextualize.plugins` entry-point group.

Install the maintained provider bundle with:

```bash
uv pip install "contextualize[plugins]"
```

Each plugin module should export:

- `PLUGIN_API_VERSION = "1"`
- `PLUGIN_NAME = "my-plugin"`
- `PLUGIN_PRIORITY = 200`
- optional `PLUGIN_KIND = "source" | "processor"` (defaults to `"source"`)
- `can_resolve(target: str, context: dict) -> bool`
- `resolve(target: str, context: dict) -> list[dict]`
- optional `list_targets(target: str, context: dict) -> list[dict]`
- optional `materialize(target: str, context: dict) -> list[dict]`
- optional `register_auth_command(group) -> None`

`resolve` should return items shaped like:

```python
{
  "source": "scheme://target",
  "label": "provider/path-or-id",
  "content": "plain text payload",
  "metadata": {"trace_path": "provider/path-or-id"},
}
```

Plugins are checked in priority order (highest first). The first plugin that
matches and returns valid documents wins. If a plugin errors, contextualize
warns and falls through to the next plugin or default resolver.

`list_targets` powers `contextualize cat --list` for plugins that can enumerate
available refs without reading all content. It should return items shaped like:

```python
{
  "target": "scheme://target/item",
  "label": "optional display label",
  "kind": "optional item kind",
  "metadata": {},
}
```

`materialize` lets a plugin expose a listed child target as one or more ordinary
files so the normal resolver stack can claim them. This is for embedded targets
such as a Discord attachment that should be re-read as a zip, image, or audio
file instead of being rendered by the parent provider. It should return items
shaped like:

```python
{
  "source": "scheme://target/item",
  "label": "optional display label",
  "filename": "export.zip",
  "content": b"...",
  "content_type": "application/zip",
  "metadata": {},
}
```

`source` plugins resolve targets directly, while `processor` plugins add post-resolution capabilities (e.g. transcription routing, media processing policy).
