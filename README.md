# contextualize

`contextualize` is a CLI for assembling files and other text snippets for use with LLMs.

<img src="https://github.com/jmpaz/contextualize/assets/30947643/01dbcec2-69fc-405a-8d91-0a00626f8946" width=80%>


## Installation

Install the core CLI:

```bash
uv tool install contextualize
```

or install with the maintained plugin bundle:

```bash
uv tool install 'contextualize[plugins]'
```

The plugins extra installs provider plugins from [jmpaz/cx-plugins](https://github.com/jmpaz/cx-plugins).


## Commands

All of the following commands work with global flags `--prompt`, `--wrap`, `--copy`, `--staged-copy`, `--count`, and `--write-file`.

| command   | purpose |
|-----------|---------|
| `cat`     | gather file contents (e.g. for piping to [`llm`](https://github.com/simonw/llm), or pasting elsewhere) |
| `map`     | survey file/folder structure(s) with [aider](https://github.com/paul-gauthier/aider)                   |
| `shell`   | capture output from arbitrary shell commands                                                           |
| `payload` | compose text and file blocks from a YAML manifest                                                      |
| `paste`   | capture staged clipboard snippets                                                                      |

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
