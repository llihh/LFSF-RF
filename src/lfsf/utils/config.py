"""Configuration utilities."""

import importlib
from pathlib import Path
from typing import Any, Dict

import yaml


def get_obj_from_str(string: str, reload: bool = False):
    """Dynamically import and return a class/function from a dotted path string."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Dict[str, Any]):
    """Instantiate a Python object from a config dict with 'target' and 'params' keys."""
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))


def load_config(config_path: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load a YAML config file and optionally apply CLI overrides.

    Args:
        config_path: Path to the YAML config file.
        overrides: Optional dict of parameter overrides (merged recursively).

    Returns:
        Merged config dict.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if overrides:
        _deep_update(config, overrides)
    return config


def _deep_update(base: dict, update: dict) -> dict:
    """Recursively update a nested dict."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base
