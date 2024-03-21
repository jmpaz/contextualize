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
- set `format` to "md" or "xml" to wrap file contents in Markdown code blocks or `<file>` tags
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
- Example usage:
    - `contextualize cat README.md` will print the wrapped contents of `README.md` to the console with default settings (Markdown format, relative path label).
    - `contextualize cat README.md --format xml` will print the wrapped contents of `README.md` to the console with XML format.
    - `contextualize cat contextualize/ dev/ README.md --format xml` will prepare file references for files in the `contextualize/` and `dev/` directories and `README.md`, and print each file's contents (wrapped in corresponding XML tags) to the console.


## Related projects

- [lumpenspace/jamall](https://github.com/lumpenspace/jamall)
