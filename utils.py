# Standard library imports
from typing import Dict
import yaml


def load_config(config_file: str) -> Dict:
    """
    Load and parse a YAML configuration file.

    This function opens the specified YAML file and loads its contents
    into a Python dictionary using PyYAML's safe_load function.

    Args:
        config_file (str): The path to the YAML configuration file.

    Returns:
        Dict: A dictionary containing the parsed configuration data.

    Raises:
        yaml.YAMLError: If there's an error parsing the YAML file.
        FileNotFoundError: If the specified file doesn't exist.
        PermissionError: If the user doesn't have permission to read the file.
    """

    with open(config_file, 'r') as f:
        return yaml.safe_load(f)
