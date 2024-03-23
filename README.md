# contextualize

`contextualize` is a package to quickly retrieve and format file contents for use with LLMs.


## Installation

You can install the package using pip:
```python
pip install contextualize
```

or pipx for using the CLI globally:
```python
pipx install contextualize
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
- `cat`: Prepare and concatenate file references
    - `paths`: Positional arguments for target file(s) or directories
    - `--ignore`: File(s) to ignore (optional)
    - `--format`: Output format (`md` or `xml`, default is `md`)
    - `--label`: Label style (`relative` for relative file path, `name` for file name only, `ext` for file extension only; default is `relative`)
    - `--output`: Output target (`console` (default), `clipboard`)
    - `--output-file`: Output file path (optional, compatible with `--output clipboard`)
- `ls`: List token counts
    - `paths`: Positional arguments for target file(s) or directories  
    - `--encoding`: Encoding to use for tokenization, e.g., `cl100k_base` (default), `p50k_base`, `r50k_base`
    - `--model`: Model (e.g., `gpt-3.5-turbo`/`gpt-4` (default), `text-davinci-003`, `code-davinci-002`) to determine which encoding to use for tokenization. Not used if `encoding` is provided.

#### Examples
- `cat`:
    - `contextualize cat README.md` will print the wrapped contents of `README.md` to the console with default settings (Markdown format, relative path label).
    - `contextualize cat README.md --format xml` will print the wrapped contents of `README.md` to the console with XML format.
    - `contextualize cat contextualize/ dev/ README.md --format xml` will prepare file references for files in the `contextualize/` and `dev/` directories and `README.md`, and print each file's contents (wrapped in corresponding XML tags) to the console.
- `ls`:
    - `contextualize ls README.md` will count and print the number of tokens in `README.md` using the default `cl100k_base` encoding.
    - `contextualize ls contextualize/ --model text-davinci-003` will count and print the number of tokens in each file in the `contextualize/` directory using the `p50k_base` encoding associated with the `text-davinci-003` model, then print the total tokens for all processed files.

## Related projects

- [lumpenspace/jamall](https://github.com/lumpenspace/jamall)
