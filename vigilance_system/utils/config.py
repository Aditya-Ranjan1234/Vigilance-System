"""
Configuration utility for the vigilance system.

This module provides functions to load and access configuration settings
from the config.yaml file or default_config.json if the yaml file is not found.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional


class Config:
    """
    Configuration manager for the vigilance system.

    Loads configuration from a YAML file and provides access to the settings.
    """

    _instance = None

    def __new__(cls, config_path: Optional[str] = None):
        """
        Singleton pattern implementation to ensure only one config instance exists.

        Args:
            config_path: Path to the configuration file. If None, uses default path.

        Returns:
            Config: The singleton Config instance
        """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        if self._initialized:
            return

        self._config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'config.yaml'
        )
        self._config = self._load_config()
        self._initialized = True

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from the YAML file or default JSON config if YAML not found.

        Returns:
            Dict[str, Any]: The configuration dictionary

        Raises:
            yaml.YAMLError: If the configuration file is not valid YAML
        """
        try:
            with open(self._config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Try to load the default config from JSON
            default_config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config', 'default_config.json'
            )
            try:
                with open(default_config_path, 'r') as f:
                    print(f"Using default configuration from {default_config_path}")
                    return json.load(f)
            except FileNotFoundError:
                print(f"Default configuration not found: {default_config_path}")
                return {}  # Return empty config instead of raising an error
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: The configuration key (can use dot notation for nested keys)
            default: Default value to return if key is not found

        Returns:
            Any: The configuration value or default if not found
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_camera_config(self, camera_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific camera.

        Args:
            camera_name: The name of the camera

        Returns:
            Dict[str, Any]: The camera configuration or empty dict if not found
        """
        cameras = self.get('cameras', [])
        for camera in cameras:
            if camera.get('name') == camera_name:
                return camera
        return {}

    def reload(self) -> None:
        """Reload the configuration from the file."""
        self._config = self._load_config()

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.

        Args:
            key: The configuration key (can use dot notation for nested keys)
            value: The value to set
        """
        keys = key.split('.')
        config_dict = self._config

        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            elif not isinstance(config_dict[k], dict):
                config_dict[k] = {}
            config_dict = config_dict[k]

        # Set the value
        config_dict[keys[-1]] = value

    def save(self) -> None:
        """Save the current configuration to the file."""
        try:
            with open(self._config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
        except Exception as e:
            raise IOError(f"Error saving configuration file: {e}")


# Create a default instance
config = Config()
