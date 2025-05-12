"""
Stream manager module for handling multiple camera streams.

This module provides a centralized manager for all camera streams in the system.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from threading import Lock

from vigilance_system.utils.logger import get_logger
from vigilance_system.utils.config import config
from vigilance_system.video_acquisition.camera import Camera, create_camera, find_video_files

# Initialize logger
logger = get_logger(__name__)


class StreamManager:
    """
    Manager for multiple camera streams.

    Handles the lifecycle of all cameras in the system, providing a unified
    interface for accessing frames from any camera.
    """

    _instance = None

    def __new__(cls):
        """
        Singleton pattern implementation to ensure only one stream manager exists.

        Returns:
            StreamManager: The singleton StreamManager instance
        """
        if cls._instance is None:
            cls._instance = super(StreamManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the stream manager."""
        if self._initialized:
            return

        self.cameras: Dict[str, Camera] = {}
        self.camera_lock = Lock()
        self._initialized = True
        logger.info("Stream manager initialized")

    def initialize_cameras(self) -> None:
        """
        Initialize all cameras from the configuration.

        Loads camera configurations from the config file and creates camera instances.
        If no cameras are configured, falls back to video files in the videos directory.
        """
        camera_configs = config.get('cameras', [])

        # If no cameras are configured, look for video files
        if not camera_configs:
            logger.warning("No cameras configured in config file, looking for video files...")
            camera_configs = find_video_files()

            # If still no cameras, try to use the default webcam
            if not camera_configs:
                logger.warning("No video files found, trying to use default webcam...")
                camera_configs = [{
                    'name': 'default_webcam',
                    'url': '0',
                    'type': 'webcam',
                    'fps': 30
                }]

        # Create cameras from configurations
        for camera_config in camera_configs:
            camera = create_camera(camera_config)
            if camera:
                self.add_camera(camera)

        # If no cameras were added, log an error
        if not self.cameras:
            logger.error("No cameras could be initialized. Please check your configuration or add video files to the 'videos' directory.")

    def add_camera(self, camera: Camera) -> bool:
        """
        Add a camera to the manager.

        Args:
            camera: Camera instance to add

        Returns:
            bool: True if camera was added successfully, False otherwise
        """
        with self.camera_lock:
            if camera.name in self.cameras:
                logger.warning(f"Camera with name '{camera.name}' already exists")
                return False

            self.cameras[camera.name] = camera
            logger.info(f"Added camera '{camera.name}' to stream manager")
            return True

    def remove_camera(self, camera_name: str) -> bool:
        """
        Remove a camera from the manager.

        Args:
            camera_name: Name of the camera to remove

        Returns:
            bool: True if camera was removed successfully, False otherwise
        """
        with self.camera_lock:
            if camera_name not in self.cameras:
                logger.warning(f"Camera '{camera_name}' not found")
                return False

            camera = self.cameras[camera_name]
            camera.stop()
            del self.cameras[camera_name]
            logger.info(f"Removed camera '{camera_name}' from stream manager")
            return True

    def get_camera(self, camera_name: str) -> Optional[Camera]:
        """
        Get a camera by name.

        Args:
            camera_name: Name of the camera to get

        Returns:
            Optional[Camera]: Camera instance or None if not found
        """
        with self.camera_lock:
            return self.cameras.get(camera_name)

    def get_all_cameras(self) -> List[Camera]:
        """
        Get all cameras.

        Returns:
            List[Camera]: List of all camera instances
        """
        with self.camera_lock:
            return list(self.cameras.values())

    def get_camera_names(self) -> List[str]:
        """
        Get names of all cameras.

        Returns:
            List[str]: List of camera names
        """
        with self.camera_lock:
            return list(self.cameras.keys())

    def start_all_cameras(self) -> None:
        """Start all cameras."""
        with self.camera_lock:
            for camera in self.cameras.values():
                camera.start()
            logger.info(f"Started all cameras ({len(self.cameras)})")

    def stop_all_cameras(self) -> None:
        """Stop all cameras."""
        with self.camera_lock:
            for camera in self.cameras.values():
                camera.stop()
            logger.info(f"Stopped all cameras ({len(self.cameras)})")

    def get_latest_frame(self, camera_name: str) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the latest frame from a specific camera.

        Args:
            camera_name: Name of the camera

        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and frame if available
        """
        camera = self.get_camera(camera_name)
        if not camera:
            return False, None

        return camera.get_latest_frame()

    def get_all_latest_frames(self) -> Dict[str, np.ndarray]:
        """
        Get the latest frames from all cameras.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping camera names to frames
        """
        # First check if we have real network nodes available
        from vigilance_system.network.node_client import node_client
        stats = node_client.get_stats()
        real_nodes = stats.get('real_nodes', 0)

        # If no real nodes are available, log a warning and return empty frames
        if real_nodes == 0:
            logger.warning("No real network nodes available - cannot process video frames")
            return {}

        frames = {}
        with self.camera_lock:
            for name, camera in self.cameras.items():
                success, frame = camera.get_latest_frame()
                if success:
                    frames[name] = frame
        return frames

    def on_network_disconnect(self) -> None:
        """
        Handle network disconnection event.

        This method is called when the network nodes are disconnected.
        It pauses all camera streams to ensure video processing stops.
        """
        logger.warning("Network disconnected - pausing all camera streams")
        with self.camera_lock:
            for camera in self.cameras.values():
                # Don't fully stop the cameras, just pause them
                camera.pause()
            logger.info(f"Paused all cameras ({len(self.cameras)}) due to network disconnect")

    def resume_cameras(self) -> None:
        """
        Resume all paused cameras.

        This method can be called when network connectivity is restored.
        """
        logger.info("Resuming all paused cameras")
        with self.camera_lock:
            for camera in self.cameras.values():
                if not camera.is_running and camera.is_connected:
                    camera.start()
            logger.info(f"Resumed cameras ({len(self.cameras)})")


# Create a default instance
stream_manager = StreamManager()
