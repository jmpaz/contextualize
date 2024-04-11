import yaml

def read_config(filename="config.yaml"):
    try:
        with open(filename, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}

