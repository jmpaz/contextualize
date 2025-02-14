import os


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
