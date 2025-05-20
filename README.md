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
| `payload` | compose text and file blocks from a YAML manifest                                          |
| `shell`   | capture output from arbitrary commands                                                     |
| `map`     | survey file/folder structure(s) with [aider](https://github.com/paul-gauthier/aider)                                                     |
| `fetch`   | retrieve Linear issues (legacy)                                                            |

All commands work with the global flags `--prompt`, `--wrap`, `--copy`, and `--write-file`. see [`docs/usage.md`](docs/usage.md) for details.

```bash
# example: gather a few files and copy to clipboard with a prefix
contextualize --copy --prompt "what does this do" cat contextualize/ README.md
```

more examples and details are available in [`docs/cat.md`](docs/cat.md) and [`docs/payload.md`](docs/payload.md).


## Shell scripting, etc
`contextualize` commands can <sup><sub>(should)</sub></sup> be aliased and composed/scripted/recalled freely in the shell.
