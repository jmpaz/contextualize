# contextualize

`contextualize` is a package to quickly retrieve and format file contents for use with LLMs.

<img src="https://github.com/jmpaz/contextualize/assets/30947643/01dbcec2-69fc-405a-8d91-0a00626f8946" width=80%>


## Installation

You can install the package using pip:
```python
pip install contextualize
```

or with [uv](https://docs.astral.sh/uv/) to use the CLI globally:
```python
uv tool install contextualize
```


## Usage (`reference.py`)

Define `FileReference` objects for specified file paths and optional ranges.
- set `range` to a tuple of line numbers to include only a portion of the file, e.g. `range=(1, 10)`
- set `format` to "md" (default) or "xml" to wrap file contents in Markdown code blocks or `<file>` tags
- set `label` to "relative" (default), "name", or "ext" to determine what label is affixed to the enclosing Markdown/XML string
    - "relative" will use the relative path from the current working directory
    - "name" will use the file name only
    - "ext" will use the file extension only

Retrieve wrapped contents from the `output` attribute.


### CLI

A CLI (`cli.py`) is provided to print file contents to the console from the command line.

- **`cat`**: Prepare and concatenate file references
  - **`paths`**: Positional arguments for target file(s) or directories
  - **`--ignore`**: File(s) to ignore (optional)
  - **`--format`**: Output format (`md`, `xml`, or `shell`; default is `md`):
    - **`shell`** mimics `cat` output in a live shell prompt
    - **`xml`** encloses file contents in `<file>` tags
    - **`md`** encloses file contents in triple backticks
  - **`--label`**: Label style (`relative` for relative file path, `name` for file name only, `ext` for file extension only; default is `relative`)
  - **`--output`**: Output target (`console` (default), `clipboard`)
  - **`--output-file`**: Output file path (optional, compatible with `--output clipboard`)

- **`ls`**: List file token counts
  - **`paths`**: Positional arguments for target file(s) or directories to process
  - **`--openai-encoding`**: OpenAI encoding to use for tokenization, e.g., `cl100k_base` (default), `p50k_base`, `r50k_base`
  - **`--openai-model`**: OpenAI model (e.g., `gpt-3.5-turbo`/`gpt-4` (default), `text-davinci-003`, `code-davinci-002`) to determine which encoding to use for tokenization.
  - **`--anthropic-model`**: Anthropic model to use for token counting (e.g., `claude-3-5-sonnet-latest`)

- **`map`**: Generate file/repo maps with [aider](https://aider.chat/2023/10/22/repomap.html)
  - **`paths`**: Positional arguments for file(s) or folder(s) to include in the repo map
  - **`-t, --max-tokens`**: Maximum tokens for the repo map (default: 10000)
  - **`--output`**: Output target (options: `console` (default), `clipboard`)
  - **`--output-file`**: Optional output file path

#### Examples

- **`cat`**:
  - `contextualize cat README.md` will print the wrapped contents of `README.md` to the console with default settings (Markdown format, relative path label).
  - `contextualize cat README.md --format xml` will print the wrapped content of `README.md` to the console with XML format.
  - `contextualize cat README.md --format shell` will print the content as if a user is running `cat README.md` in a live shell prompt.
  - `contextualize cat contextualize/ dev/ README.md --format xml` will prepare file references for files in the `contextualize/` and `dev/` directories and `README.md`, and print each file’s contents (wrapped in corresponding XML tags) to the console.

- **`ls`**:
  - `contextualize ls README.md` will count and print the number of tokens in `README.md` using the default `cl100k_base` encoding, unless `ANTHROPIC_API_KEY` is set, in which case the Anthropic [token counting API](https://docs.anthropic.com/en/docs/build-with-claude/token-counting) will be used.
  - `contextualize ls contextualize/ --openai-model text-davinci-003` will count and print the number of tokens in each file in the `contextualize/` directory using the `p50k_base` encoding associated with the `text-davinci-003` model, then print the total tokens for all processed files.

- **`map`**:
  - `contextualize map .` will generate and print a repository map for the current directory.
  - `contextualize map src/ tests/ -t 15000` will generate a repository map for files in the `src/` and `tests/` directories with a maximum of 15000 tokens.
