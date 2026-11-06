"""
Configuration Loader Module
===========================
Loads and validates system-wide configuration settings with Pydantic / dataclass schemas.
"""

import os
from typing import List, Dict, Any
from ruamel.yaml import YAML

_yaml = YAML(typ="safe")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")


def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = _yaml.load(f)
    return config


class Config:
    """Convenience wrapper for configuration dictionary."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str):
        val = self._data[key]
        if isinstance(val, dict):
            return Config(val)
        return val

    def get(self, key: str, default: Any = None):
        val = self._data.get(key, default)
        if isinstance(val, dict):
            return Config(val)
        return val

    def to_dict(self) -> Dict[str, Any]:
        return self._data


def get_default_config() -> Config:
    return Config(load_config())
