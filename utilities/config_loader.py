from pathlib import Path
from typing import Any

import yaml


def _coerce_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Convert *path keys from strings to ``Path`` objects."""
    for key, value in config.items():
        if isinstance(value, str) and key.endswith("path"):
            config[key] = Path(value)
    return config


def load_config(config: str) -> dict:
    """
    Load a YAML config file and return it as a dict.

    If a parameter ends with 'path' (i.e model_path), it will be converted to a Path object.

    :param config (str): Path to YAML config file

    :return (dict): dict containing config
    """
    config_path = Path(config)

    # Ensure file exists and is YAML file
    if not config_path.exists():
        raise FileNotFoundError(f'File {config_path} does not exist')
    if config_path.suffix != '.yaml':
        raise ValueError(f'File {config_path} is not a YAML file, ensure extension is .yaml')

    # Load yaml and create dict
    with open(config_path, 'r', encoding='utf-8') as f:
        config_from_yaml = yaml.safe_load(f)

    if config_from_yaml is None:
        raise ValueError(f'File {config_path} is empty')

    # Backwards compatibility for legacy config key naming.
    if (
        'augment_maximum_drop_point_probability' not in config_from_yaml
        and 'augment_drop_drop_probability' in config_from_yaml
    ):
        config_from_yaml['augment_maximum_drop_point_probability'] = config_from_yaml['augment_drop_drop_probability']

    return _coerce_paths(config_from_yaml)
