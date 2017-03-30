from ruamel import yaml


def load_yaml_config(yaml_file):
    with open(yaml_file) as fp:
        return yaml.round_trip_load(fp)
