import os

from contextualize.tokenize import count_tokens


def generate_repo_map_data(paths, max_tokens, fmt):
    """
    Generate a repository map and return a dict containing:
      - repo_map: The generated repository map as a string.
      - summary: A summary string with file/token info.
      - messages: Any warnings/errors collected.
      - error: Present if no map could be generated.
    """
    from aider.repomap import RepoMap, find_src_files

    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(find_src_files(path))
        else:
            files.append(path)

    class CollectorIO:
        def __init__(self):
            self.messages = []

        def tool_output(self, msg):
            self.messages.append(msg)

        def tool_error(self, msg):
            self.messages.append(f"ERROR: {msg}")

        def tool_warning(self, msg):
            self.messages.append(f"WARNING: {msg}")

        def read_text(self, fname):
            try:
                with open(fname, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                self.tool_warning(f"Error reading file {fname}: {str(e)}")
                return ""

    class TokenCounter:
        def token_count(self, text):
            result = count_tokens(text, target="cl100k_base")
            return result["count"]

    io = CollectorIO()
    token_counter = TokenCounter()

    rm = RepoMap(map_tokens=max_tokens, main_model=token_counter, io=io)
    repo_map = rm.get_repo_map(chat_files=[], other_files=files)

    if not repo_map:
        error_message = "\n".join(io.messages) or "No repository map was generated."
        return {"error": error_message}

    if fmt == "shell":
        repo_map = f"❯ repo-map {' '.join(paths)}\n{repo_map}"

    token_info = count_tokens(repo_map, target="cl100k_base")
    num_files = len(files)
    summary_str = f"Map of {num_files} files ({token_info['count']} tokens)"

    return {
        "repo_map": repo_map,
        "summary": summary_str,
        "messages": io.messages,
    }
