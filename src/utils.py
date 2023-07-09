import yaml


def load_yaml_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            raise exc
