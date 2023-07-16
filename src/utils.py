import yaml
from pathlib import Path


def load_yaml_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            raise exc


def create_label_map(data_dir):
    import json
    label_map = {dir_.name: i for i, dir_ in enumerate(Path(data_dir).glob('*'))}
    with open(Path(data_dir).parents[1] / 'meta/label_map.json', 'w') as f:
        json.dump(label_map, f)
    return label_map


def label_to_idx(label, label_map=None, label_map_path=None):
    import json
    if label_map is None:
        assert label_map_path is not None, 'Either label_map or label_map_path must be provided.'
        with open(Path(label_map_path), 'r') as f:
            label_map = json.load(f)
    return label_map[label]


def idx_to_label(idx, label_map=None, label_map_path=None):
    import json
    if label_map is None:
        with open(Path(label_map_path), 'r') as f:
            label_map = json.load(f)
    return list(label_map)[idx]




#label_map = create_label_map('data/raw/train')

#label_to_idx('waffles', label_map_path='data/meta/label_map.json')

#idx_to_label(100, label_map_path='data/meta/label_map.json')

#('apple_pie', label_map=label_map)