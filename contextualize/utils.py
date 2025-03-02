import os
import re


def get_config_path(custom_path=None):
    if custom_path:
        return custom_path
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return os.path.join(xdg_config_home, "contextualize", "config.yaml")


def read_config(custom_path=None):
    import yaml

    config_path = get_config_path(custom_path)
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}


def wrap_text(content: str, wrap_mode: str) -> str:
    """
    Wrap the given content according to wrap_mode ('xml' or 'md').
    If wrap_mode is None or empty string, return content unmodified.
    """
    if not wrap_mode:
        return content

    if wrap_mode == "xml":
        return f"<paste>\n{content}\n</paste>"

    if wrap_mode in ("md", "markdown"):
        backtick_runs = re.findall(r"`+", content)
        longest = max(len(run) for run in backtick_runs) if backtick_runs else 0

        fence_len = longest + 2 if longest >= 3 else 3
        fence = "`" * fence_len

        return f"{fence}\n{content}\n{fence}"

    return content
