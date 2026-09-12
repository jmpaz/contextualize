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
- optional `list_targets(target: str, context: dict) -> dict`
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
available refs without reading all content. It should return an envelope with
items shaped like:

```python
{
  "targets": [
    {
      "target": "scheme://target/item",
      "label": "optional display label",
      "kind": "optional item kind",
      "traverse": False,
      "metadata": {},
    }
  ],
  "summary": {},
  "pagination": None,
  "metadata": {},
  "capabilities": {},
}
```

`contextualize cat --list --json TARGET...` preserves these envelopes in a
`listings` array, one entry per expanded input target. Each entry adds `source`
(the input target) and `provider` (the plugin name), alongside `targets`,
`summary`, `pagination`, `metadata`, and `capabilities`. Optional envelope fields
may be `null`. Item labels, kinds, traversal flags, and metadata remain intact;
results from different inputs are not flattened together.

The JSON object also contains `content`, the plain Markdown listing, and
`selectors`, any contextualize selector provenance. Without `--json`, `--list`
still prints deduplicated target references only. Use the structured envelope
when continuation cursors, scan bounds, or other provider metadata matter:
an empty `targets` array can still have more results available according to its
`pagination` and provider summary. Git listings use the same JSON structure
with `provider: "git"`.

Callers may pass `list_limit` and `list_offset` through plugin context. The
core plugin resolver applies that page window to the normalized envelope and
adds `offset`, `limit`, `returned`, `totalCount`, `hasMore`, and `nextOffset`
pagination fields where applicable.

Omit `traverse` or set it to `True` when embedded resolution may follow the listed
target from a collected item. Set `traverse` to `False` when the item should be
visible to listing and inspection, but should not be resolved as attached
context.

`materialize` lets a plugin expose a listed child target as one or more ordinary
files so the normal resolver stack can claim them. This is for embedded
targets such as an attachment that should be re-read as a zip, image, or audio
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
