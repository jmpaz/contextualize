# Usage

### Global Options
`contextualize` provides a single CLI entry point with global options that work with all subcommands.

- `-p, --prompt` prepend and optionally append up to two strings
- `-w, --wrap` wrap output as `md` or `xml`; `-w` alone is a shorthand for `--wrap xml`
- `-c, --copy` copy to clipboard instead of printing; displays the token count
- `--write-file PATH` write final output to a file

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

## `cat`

Collect file contents, with optional wrapping and labels.

```
contextualize cat PATH [PATH...] [--ignore PATH] [--format md|xml|shell] [--label relative|name|ext]
```

| option | description |
|--------|-------------|
| `paths` | one or more files or directories |
| `--ignore` | glob pattern(s) to skip |
| `--format` | choose `md` (default), `xml`, or `shell` |
| `--label` | how to label each attachment: `relative` (default), `name`, or `ext` |


```bash
contextualize -p "review:" -c \  # prepend "summarize:" to the output of cat; copy the result
  cat -f xml \                   # wrap each file's content in '<paste>' tags
  pyproject.toml docs/           # extract content from pyproject.toml, docs/*
```


## `payload`

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
