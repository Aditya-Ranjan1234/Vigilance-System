"""
Tests for the configuration module.
"""

import os
import pytest
import tempfile
import yaml

from vigilance_system.utils.config import Config


def test_config_singleton():
    """Test that Config is a singleton."""
    config1 = Config()
    config2 = Config()
    assert config1 is config2


def test_config_loading():
    """Test loading configuration from a file."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'test': {
                'value1': 'test_value',
                'value2': 42,
                'nested': {
                    'value3': True
                }
            }
        }, f)
    
    try:
        # Initialize config with the temporary file
        config = Config(f.name)
        
        # Test getting values
        assert config.get('test.value1') == 'test_value'
        assert config.get('test.value2') == 42
        assert config.get('test.nested.value3') is True
        
        # Test getting non-existent values
        assert config.get('nonexistent') is None
        assert config.get('nonexistent', 'default') == 'default'
        assert config.get('test.nonexistent') is None
        assert config.get('test.nonexistent', 123) == 123
        
    finally:
        # Clean up
        os.unlink(f.name)


def test_camera_config():
    """Test getting camera configuration."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'cameras': [
                {
                    'name': 'front_door',
                    'url': 'rtsp://example.com/stream1',
                    'type': 'rtsp'
                },
                {
                    'name': 'back_yard',
                    'url': 'http://example.com/stream2',
                    'type': 'http'
                }
            ]
        }, f)
    
    try:
        # Initialize config with the temporary file
        config = Config(f.name)
        
        # Test getting camera config
        front_door = config.get_camera_config('front_door')
        assert front_door['name'] == 'front_door'
        assert front_door['url'] == 'rtsp://example.com/stream1'
        assert front_door['type'] == 'rtsp'
        
        back_yard = config.get_camera_config('back_yard')
        assert back_yard['name'] == 'back_yard'
        assert back_yard['url'] == 'http://example.com/stream2'
        assert back_yard['type'] == 'http'
        
        # Test getting non-existent camera
        assert config.get_camera_config('nonexistent') == {}
        
    finally:
        # Clean up
        os.unlink(f.name)
