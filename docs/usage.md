# Usage

### Global Options
`contextualize` provides a single CLI entry point with global options that work with all subcommands.

- `-p, --prompt` prepend and optionally append up to two strings
- `-w, --wrap` wrap output as `md` or `xml`; `-w` alone is a shorthand for `--wrap xml`
- `--verbose` enable provider diagnostics, Rich live progress on interactive stderr, and an end-of-run progress summary
- `--quiet` disable provider progress logs on stderr
- `-c, --copy` copy to clipboard instead of printing; displays the token count
- `-s, --staged-copy` with `--prompt` and a copy mode, copy preprompt, content, and postprompt as distinct stages
- `--count` dry run of `--copy`; prints a string containing the token count
- `--write-file PATH` write final output to a file
- `--token-target STR` choose the encoding/model for token counting (e.g. `cl100k_base`, `gpt-4o-mini`, `claude-3-5-sonnet-20241022`)
- `--spec-jobs N` set parallel file-spec resolution jobs; defaults to `CONTEXTUALIZE_PAYLOAD_SPEC_JOBS` or `8`
- `--media-jobs N` set parallel embedded/media processing jobs; defaults to `CONTEXTUALIZE_PAYLOAD_MEDIA_JOBS` or `4`
- `--transcription-jobs N` set concurrent transcription requests; defaults to `CONTEXTUALIZE_TRANSCRIPTION_JOBS` or `2`
- `--download-jobs N` set concurrent media downloads; defaults to `CONTEXTUALIZE_MEDIA_DOWNLOAD_JOBS` or `2`
- `-a, --after` / `-b, --before` control placement in pipelines (default: after)

`--count` cannot be combined with `--copy` or `--copy-segments`.
`--copy` and `--copy-segments` cannot be combined.
`--staged-copy` requires `--prompt` and either `--copy` or `--copy-segments`.

Hydration commands enable progress output by default. Use `contextualize hydrate --quiet ...` or `contextualize contexts hydrate --quiet ...` to suppress the live progress display and summary.

The copy modes (`--copy`, `--copy-segments`, `--staged-copy`) pick a clipboard route from the session:

- **Local:** the native clipboard — `pbcopy` on macOS, `wl-copy` on Wayland, otherwise pyperclip (X11, Windows, WSL). Every copy is read back, so `Copied … to clipboard` means the clipboard holds the output. If a native clipboard exists but the copy fails or reads back different content, the command fails; OSC52 is used only when no native clipboard is available.
- **SSH** (`SSH_TTY` or `SSH_CONNECTION` set): OSC52 to your terminal, since the remote host's own clipboard is not the one you paste from.
- **tmux**, local or remote: `tmux load-buffer -w -` first, then a tmux-wrapped OSC52 sequence.

Terminals do not acknowledge OSC52, so these sends report `Sent … to your terminal via OSC52` (or `via tmux`) instead of `Copied`, and keep a copy of the output. The terminal must allow clipboard writes and may cap how much it accepts; if tmux does not forward them, add `set -g set-clipboard on` to tmux config and ensure the terminal app allows OSC52 clipboard access.

If a copy fails, or a staged or segmented copy is interrupted, the command exits with status 1 and saves the full output. The same file keeps the output of unconfirmed sends: `$XDG_STATE_HOME/contextualize/clipboard/unconfirmed.txt` (default `~/.local/state/contextualize/clipboard/unconfirmed.txt`), readable only by you and replaced each time. Copy it again without repeating the extraction:

```bash
contextualize --copy < ~/.local/state/contextualize/clipboard/unconfirmed.txt
```

Set `CONTEXTUALIZE_CLIPBOARD=native` or `CONTEXTUALIZE_CLIPBOARD=osc52` to force a route, or leave it unset (`auto`); `pyperclip` is accepted as an older name for `native`. Forcing `native` gives verified copies inside a local tmux session, and over SSH it targets the remote host's clipboard.

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

### `auth`

Plugin-provided authentication helpers.

```
contextualize auth PROVIDER [provider options]
```

Run `contextualize auth` to list available handlers from loaded plugins.

### `cat`

Collect file contents, with optional wrapping and labels.

```
contextualize cat [PATH...] [--ignore PATH] [--format md|xml|shell|raw] [--label relative|name|ext] [--tokens] [--git-pull] [--git-reclone] [--list]
```

`PATH` may also start with a git repo spec such as `github:user/repo` or `https://host/repo.git:path`.

It can also be:
- an `http(s)` URL pointing to a UTF-8 file.
- a Bluesky `bsky.app` URL or an `at://` ATProto URI.
- a SoundCloud track/playlist/artist URL or URN (`soundcloud:tracks:*`, `soundcloud:playlists:*`, `soundcloud:users:*`).
- a WhatsApp exported chat archive zip.

Multiple paths can be separated with commas after the colon.
Brace expressions and glob patterns in those paths are expanded after cloning.
The `.git` suffix is optional and the repo will be cloned to `~/.local/share/contextualize/cache/git/` on first use.
If no paths are provided and stdin includes `http(s)` URLs, `cat` extracts them and treats them as refs.
Use `--list` to print Markdown bullets for refs exposed by git targets or plugins with a `list_targets` hook without reading full file contents.

Non-text files supported by [markitdown](https://github.com/microsoft/markitdown) are automatically converted to text.
For Are.na, ATProto, Discord, and WhatsApp media descriptions, media conversions are cached locally; use `--refresh-media` to force a re-fetch.

Images are described by the Codex app server when one answers, and by the OpenRouter/OpenAI chat endpoint otherwise; `CONTEXTUALIZE_MD_IMAGE_PROVIDER` pins that choice to `auto` (default), `app-server`, or `openrouter`. The app server's model comes from `CONTEXTUALIZE_CODEX_APP_SERVER_MODEL` and its command from `CODEX_APP_SERVER_COMMAND`; the OpenRouter model comes from `OPENAI_MODEL` against `OPENAI_BASE_URL`.

Every description that actually reaches a provider — a cached one reaches none — records a progress event naming the provider that ran, the model it ended up using after any reroute, and the tokens the call reported. `--verbose` prints them in the end-of-run summary, and `CONTEXTUALIZE_PROGRESS_JOURNAL=PATH` appends every event as JSONL. Tools that embed `contextualize` can read the same events back through `contextualize.progress.progress_events(context)`, scoping one call's events with `set_progress_context`.

| option | description |
|--------|-------------|
| `paths` | one or more files or directories |
| `--ignore` | glob pattern(s) to skip |
| `--format` | choose `md` (default), `xml`, `shell`, or `raw` |
| `--label` | how to label each attachment: `relative` (default), `name`, or `ext` |
| `--tokens` | annotate each label with the file's token count |
| `--git-pull` | update cached git repos referenced in paths |
| `--git-reclone` | delete and re-clone cached git repos |
| `--list` | list plugin refs without reading full content |
| `--trace` | print a breakdown of gathered inputs, sorted by token count |

Append `:Symbol` (or `:Sym1,Sym2`) to any path to extract only those definitions.


```
contextualize -p "review:" -c \  # prepend "summarize:" to the output of cat; copy the result
  cat -f xml \                   # wrap each file's content in '<paste>' tags
  pyproject.toml docs/           # extract content from pyproject.toml, docs/*
```


### `paste`

Capture clipboard text in stages so it can be combined with other `contextualize` steps.

```
contextualize paste [--count INT] [--format md|xml|shell|raw] [--tokens]
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
contextualize map PATH [PATH...] [--max-tokens INT] [--ignore PATH] [--format raw|shell|md|xml] [--tokens] [--git-pull] [--git-reclone]
```

`PATH` accepts the same git repo specs as `cat`.

| option | description |
|--------|-------------|
| `paths` | directories or files to include |
| `--max-tokens` | limit map size for aider |
| `--ignore` | glob pattern(s) to skip |
| `--format` | choose `raw` (default), `shell`, `md`, or `xml` |
| `--tokens` | annotate file headers + constituent symbols with token counts |
| `--git-pull` | update cached git repos referenced in paths |
| `--git-reclone` | delete and re-clone cached git repos |


### `payload`

Compose arbitrary sets of text blocks + files into a single output via a YAML manifest.

```bash
contextualize payload MANIFEST.yaml  # or pipe into stdin
```

Payload options:
- `-m, --map-compatible` render codemaps when possible; otherwise include file contents
- `--map NAME` render maps only for named components (repeat or comma-separate)
- `--exclude NAME` skip components by `name` (repeat or comma-separate)

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
      - path: src/contextualize/**/*.py
        comment: "implementation details"
      - README.md
    suffix: |
      that was the core logic.
```

running the command yields the composed payload.

### `contexts`

Registered contexts hydrate named manifests into target directories. Static entries are read from `~/.config/contextualize/contexts.json`. Optional subscriptions in `~/.config/contextualize/config.yaml` can discover zk notes by tag and add them to the registry at runtime.

```yaml
contexts:
  subscriptions:
    - source: zk
      root: ~/notes
      tag: ctx/ref
      targetRoot: ~/ref
      contextDir: "."
      replace: guarded
```

Subscribed notes must contain a contextualize manifest. The context name comes from frontmatter `cx.context` when present, otherwise from a slugged manifest `config.name`. Static registry entries stay authoritative when names or manifest sources overlap.

`contextDir` is optional for both static entries and subscriptions. Relative values are resolved from `targetDir`; `contextDir: "."` makes the target itself the generated context root. This direct form is intended for dedicated, disposable output directories. It is never inferred from Git repository presence. Command-line `--dir` overrides registry placement, which overrides the manifest's `config.context.dir`.

```bash
contextualize contexts list  # shows source, origin, and target
contextualize contexts hydrate my-context
```

Authored component text normally hydrates under the component's `notes/`
directory. Set `text-file` to a top-level filename when the text should instead
become a named root document, such as a context's orientation README:

```yaml
components:
  - name: orientation
    text-file: README.md
    text: |
      # Orientation

      Read this before using the context.
```

Static registry entries default to origin `registry`. Nix-generated registries set origin `nix`; subscribed contexts use `tag:<tag>`.

### shell completion

`contextualize` ships dynamic shell completions for Bash, Zsh, and Fish. The Nix/Home Manager package installs those completion files automatically. Non-Nix installs can source Click's generated completion script directly:

```bash
_CONTEXTUALIZE_COMPLETE=bash_source contextualize > ~/.local/share/bash-completion/completions/contextualize
_CONTEXTUALIZE_COMPLETE=zsh_source contextualize > ~/.zfunc/_contextualize
_CONTEXTUALIZE_COMPLETE=fish_source contextualize > ~/.config/fish/completions/contextualize.fish
```

Target arguments to `cat` (including the implicit `contextualize <path>` form), `map`, and `hydrate` complete file paths. Context registry commands complete live context names, including tag-discovered contexts, for `contextualize contexts hydrate <TAB>`.

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
  cat src/contextualize/ |

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
