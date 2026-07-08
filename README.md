# contextualize

`contextualize` is a utility for working with model context.


## Installation

Install the core package:

```bash
uv tool install contextualize
```

or install with the maintained plugin bundle:

```bash
uv tool install 'contextualize[plugins]'
```

The `plugins` extra installs provider plugins from [jmpaz/cx-plugins](https://github.com/jmpaz/cx-plugins).


## Commands

All of the following commands work with global flags `--prompt`, `--wrap`, `--copy`, `--staged-copy`, `--count`, and `--write-file`.

| command   | purpose |
|-----------|---------|
| `cat`     | gather file/target contents                                                                              |
| `map`     | survey file/folder structure(s)                                                                         |
| `shell`   | capture output from arbitrary shell commands                                                             |
| `payload` | compose text and file blocks from a YAML manifest                                                        |
| `plugins` | view installed plugins (see [`docs/plugins.md`](docs/plugins.md))|

`cat` is the default command. Refs given without a subcommand run `cat`, so `contextualize src/ README.md` is shorthand for `contextualize cat src/ README.md`.
A subcommand name wins over a same-named path; reach such a path via `cat name` or `./name`.


**Sample invocations (`cat`):**

```bash
# gather files and copy (individually wrapped + labelled, prefixed by '--prompt') to clipboard
contextualize src/ README.md --prompt "how does this work?" --copy

# fetch a single file from a remote repo (cached under ~/.local/share/contextualize/cache/git/)
contextualize github:jmpaz/contextualize:README.md

# gather multiple files/folder(s) from a repo
contextualize https://git.sr.ht/~cismonx/bookmarkfs:README.md,doc

# fetch a single hosted UTF-8 file
contextualize https://modelcontextprotocol.io/llms.txt
```

Details and more examples are available in [`docs/usage.md`](docs/usage.md).

## Plugins

`contextualize` loads source and processor plugins from installed Python packages via the
`contextualize.plugins` entry-point group.

See [`docs/plugins.md`](docs/plugins.md) for details.
