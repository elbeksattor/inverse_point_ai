#!/usr/bin/env python3
"""
Configuration parser for IP AI Analytics
Loads and validates YAML configuration files
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigParser:
    """YAML configuration file parser"""

    def __init__(self, config_path: str):
        """
        Initialize configuration parser

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                return config if config is not None else {}
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML config: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path to config value (e.g., "deepstream.pgie.batch_size")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            config.get("analytics.reid.similarity_threshold", 0.75)
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """
        Get configuration section

        Args:
            key: Top-level configuration key

        Returns:
            Configuration section
        """
        return self.config.get(key, {})

    def __repr__(self) -> str:
        return f"ConfigParser(config_path='{self.config_path}')"


def load_config(config_path: str) -> ConfigParser:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ConfigParser instance
    """
    return ConfigParser(config_path)
