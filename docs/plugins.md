# Plugins

`contextualize` automatically loads plugins from:

- `~/.config/contextualize/plugins`
- additional directories from `CONTEXTUALIZE_PLUGIN_DIRS` (path-separated)

Installed Python packages may also expose plugins through the
`contextualize.plugins` entry-point group.

Each plugin is a local repo that contains `plugin.yaml`:

```yaml
name: my-plugin
module: plugin
api_version: "1"
priority: 200
enabled: true
```

The module exports:

- `PLUGIN_API_VERSION = "1"`
- `PLUGIN_NAME = "my-plugin"`
- `PLUGIN_PRIORITY = 200`
- `can_resolve(target: str, context: dict) -> bool`
- `resolve(target: str, context: dict) -> list[dict]`
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
