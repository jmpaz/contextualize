# Usage

### Global Options
`contextualize` provides a single CLI entry point with global options that work with all subcommands.

- `-p, --prompt` prepend and optionally append up to two strings
- `-w, --wrap` wrap output as `md` or `xml`; `-w` alone is a shorthand for `--wrap xml`
- `-c, --copy` copy to clipboard instead of printing; displays the token count
- `--count` dry run of `--copy`; prints a string containing the token count
- `--write-file PATH` write final output to a file
- `--token-target STR` choose the encoding/model for token counting (e.g. `cl100k_base`, `gpt-4o-mini`, `claude-3-5-sonnet-20241022`)
- `-a, --after` / `-b, --before` control placement in pipelines (default: after)

`--copy`, `--count`, and `--copy-segments` cannot be combined.

these flags can be combined with any command:

```bash
contextualize --copy -p "which file contains `fetch()`?" map src/
```

```bash
contextualize --copy -p "let's tidy up the `fetch()` fn" cat src/api.py
```

or piped into another program:

```bash
contextualize --prompt "what has changed in this patch?" shell "git diff --staged" | llm
```

## Commands

### `cat`

Collect file contents, with optional wrapping and labels.

```
contextualize cat PATH [PATH...] [--ignore PATH] [--format md|xml|shell] [--label relative|name|ext] [--tokens] [--git-pull] [--git-reclone]
```

`PATH` may also start with a git repo spec such as `gh:user/repo` or `https://host/repo.git:path`.
It can also be an `http(s)` URL pointing to a UTF-8 file.
Multiple paths can be separated with commas after the colon.
Brace expressions and glob patterns in those paths are expanded after cloning.
The `.git` suffix is optional and the repo will be cloned to `~/.local/share/contextualize/cache/git/` on first use.

| option | description |
|--------|-------------|
| `paths` | one or more files or directories |
| `--ignore` | glob pattern(s) to skip |
| `--format` | choose `md` (default), `xml`, or `shell` |
| `--label` | how to label each attachment: `relative` (default), `name`, or `ext` |
| `--tokens` | annotate each label with the file's token count |
| `--git-pull` | update cached git repos referenced in paths |
| `--git-reclone` | delete and re-clone cached git repos |


```
contextualize -p "review:" -c \  # prepend "summarize:" to the output of cat; copy the result
  cat -f xml \                   # wrap each file's content in '<paste>' tags
  pyproject.toml docs/           # extract content from pyproject.toml, docs/*
```


### `paste`

Capture clipboard text in stages so it can be combined with other `contextualize` steps.

```
contextualize paste [--count INT] [--format md|xml|shell|plain] [--tokens]
```

`paste` waits for you to copy text and press Enter for each requested chunk (defaults to one chunk).
`--count` lets you capture multiple clipboard entries in sequence, and every capture is labelled similar to `cat` output.
`--tokens` adds token count for each label.
Press `Esc` while a prompt is waiting to undo the most recent capture if you grabbed the wrong snippet.
The command works with all global flags, so you can wrap or prompt the captured text just like other sources.

```
# capture two snippets before piping them into another stage
contextualize paste --count 2 | contextualize -p "Please review both snippets:" -w --copy
```


### `map`


Generate repository maps summarizing file structure.

```
contextualize map PATH [PATH...] [--max-tokens INT] [--ignore PATH] [--format plain|shell] [--tokens] [--git-pull] [--git-reclone]
```

`PATH` accepts the same git repo specs as `cat`.

| option | description |
|--------|-------------|
| `paths` | directories or files to include |
| `--max-tokens` | limit map size for aider |
| `--ignore` | glob pattern(s) to skip |
| `--format` | choose `plain` (default) or `shell` |
| `--tokens` | annotate each file header with the full file token count |
| `--git-pull` | update cached git repos referenced in paths |
| `--git-reclone` | delete and re-clone cached git repos |


### `payload`

Compose arbitrary sets of text blocks + UTF-8 files into a single output via a YAML manifest.

```bash
contextualize payload MANIFEST.yaml  # or pipe into stdin
```

```yaml
# MANIFEST.yaml
config:
  root: ~/project  # optional base dir

components:
  - text: |
      some introductory text
  - name: core
    prefix: |
      here is the core logic:
    files:
      - contextualize/*.py
      - README.md
    suffix: |
      that was the core logic.
```

running the command yields the composed payload.

An upcoming release will add support for sections like `commands` and `maps` in addition to `files`.

### content injection

`--inject` replaces `{cx::...}` patterns inside target files with referenced content. targets can be HTTP URLs, git repos, or local paths. parameters can tweak labels and formats just like the `cat` command.

```bash
contextualize --inject cat index.md
```


## Pipelines

Commands can also be chained together with pipes, with each stage adding its own context:

```bash
# sequential context building - each prompt labels its section
contextualize -p "dependencies:" cat requirements.txt | \
  contextualize -p "recent changes:" shell "git log --oneline -5" | \
  contextualize -wp "current status:" shell "git status"
```

```bash
# control output positioning with -a (after, default) or -b (before)
contextualize -p "logs:" shell "tail app.log" | \
  contextualize -bp "system info:" shell "uname -a"
```

### Recipes

The following chains of commands will each yield a single formatted string that can be pasted into a chat UI (or piped into programs like [`llm`](https://github.com/simonw/llm) or [`claude`](https://claude.md)) to elicit certain results.

**In-style commit message**
```bash
contextualize -p "Given the following codebase:" \
  cat contextualize/ |

contextualize -wp "please write a commit message for the following changes:" \
  shell "git diff --staged -U0" |

contextualize -wp "while adhering to the following style:" \
  shell -f raw "git log --stat --oneline"
```

**Code review**
```bash
contextualize -p "Let's review the following pull request:" \
  shell "git show --stat HEAD" |

contextualize -wp "with these file changes:" \
  shell "git diff HEAD~1 --name-only | head -10 | xargs cat" |

contextualize -wp "in the context of recent commits:" \
  shell -f raw "git log --oneline -15"
```

---

**Code understanding**
```bash
contextualize -p "endpoints:" \
  shell "grep -r '@app.route' src/ | head -20" |

contextualize -p "models, config:" \
  cat src/models.py config.yaml .env.example |

contextualize -p "usage examples:" \
  cat examples/ tests/integration/ |

contextualize --before -wp "Please help me understand the following:"
```
in this chain, `--before` prepends its corresponding prompt string to the output of its incoming pipe; `-w` wraps the incoming text in a code fence.
