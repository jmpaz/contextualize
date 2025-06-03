# contextualize

`contextualize` helps gather files and other text snippets for use with LLMs.

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
| `cat`     | gather file contents (e.g. for piping to [`llm`](https://github.com/simonw/llm)). often used with `--copy` + `--prompt` |
| `map`     | survey file/folder structure(s) with [aider](https://github.com/paul-gauthier/aider)                                                     |
| `shell`   | capture output from arbitrary shell commands                                                     |
| `fetch`   | retrieve Linear issues (legacy)                                                            |
| `payload` | compose text and file blocks from a YAML manifest                                          |

All commands work with the global flags `--prompt`, `--wrap`, `--copy`, and `--write-file`.

```bash
# example: gather a few files and copy to clipboard with a prefix
contextualize --copy --prompt "what does this do" cat contextualize/ README.md
```

```bash
# gather README from a remote repo (cached under ~/.local/share/contextualize/cache/git/)
contextualize cat gh:jmpaz/contextualize:README.md

# gather multiple files/folder(s) from a repo
contextualize cat https://git.sr.ht/~cismonx/bookmarkfs:README.md,doc
```

more examples and details are available in [`docs/usage.md`](docs/usage.md).
