# contextualize

`contextualize` is a CLI for assembling files and other text snippets for use with LLMs.

<img src="https://github.com/jmpaz/contextualize/assets/30947643/01dbcec2-69fc-405a-8d91-0a00626f8946" width=80%>


## Installation

```bash
pip install contextualize
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install contextualize
```


## Commands

| command   | purpose |
|-----------|---------|
| `cat`     | gather file contents (e.g. for piping to [`llm`](https://github.com/simonw/llm), or pasting elsewhere) |
| `map`     | survey file/folder structure(s) with [aider](https://github.com/paul-gauthier/aider)                   |
| `shell`   | capture output from arbitrary shell commands                                                           |
| `fetch`   | retrieve Linear issues (legacy)                                                                        |
| `payload` | compose text and file blocks from a YAML manifest                                                      |
| `paste`   | capture staged clipboard snippets                                                                      |

All commands work with the global flags `--prompt`, `--wrap`, `--copy`, `--count`, and `--write-file`.


**Sample invocations (`cat`):**

```bash
# gather files and copy (individually wrapped + labelled, prefixed by '--prompt') to clipboard
contextualize cat contextualize/ README.md --copy --prompt "how does this work?"

# fetch a single file from a remote repo (cached under ~/.local/share/contextualize/cache/git/)
contextualize cat gh:jmpaz/contextualize:README.md

# gather multiple files/folder(s) from a repo
contextualize cat https://git.sr.ht/~cismonx/bookmarkfs:README.md,doc

# fetch a single hosted UTF-8 file
contextualize cat https://modelcontextprotocol.io/llms.txt
```

Details and more examples are available in [`docs/usage.md`](docs/usage.md).
