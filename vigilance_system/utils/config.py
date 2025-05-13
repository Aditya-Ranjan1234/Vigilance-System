"""
Configuration utility for the vigilance system.

This module provides functions to load and access configuration settings
from the config/config.yaml file.
"""

import os
import yaml
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
            'config', 'config.yaml'
        )
        self._config = self._load_config()
        self._initialized = True

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from the YAML file.

        Returns:
            Dict[str, Any]: The configuration dictionary

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the configuration file is not valid YAML
        """
        try:
            with open(self._config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self._config_path}")
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
        try:
            self._config = self._load_config()
            print(f"Configuration reloaded from {self._config_path}")
        except Exception as e:
            print(f"Error reloading configuration: {e}")
            # If there's an error, create an empty config
            self._config = {}

    def set(self, key: str, value: Any, save: bool = False) -> None:
        """
        Set a configuration value by key.

        Args:
            key: The configuration key (can use dot notation for nested keys)
            value: The value to set
            save: Whether to save the changes to the config file
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

        # Save to file if requested
        if save:
            self.save()

    def save(self) -> None:
        """
        Save the current configuration to the config file.
        """
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)

            # Save the configuration
            with open(self._config_path, 'w') as f:
                yaml.safe_dump(self._config, f, default_flow_style=False)

            # Log the save operation
            print(f"Configuration saved to {self._config_path}")

            # Force reload the configuration to ensure changes are applied
            self._config = self._load_config()
        except Exception as e:
            raise IOError(f"Error saving configuration file: {e}")


# Create a default instance
config = Config()
