"""
Camera module for handling different types of camera streams.

This module provides classes for connecting to and reading from various camera types
(RTSP, HTTP, local webcams).
"""

import time
import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
from threading import Thread, Lock
from pathlib import Path

from vigilance_system.utils.logger import get_logger
from vigilance_system.utils.config import config

# Initialize logger
logger = get_logger(__name__)


class Camera(ABC):
    """
    Abstract base class for all camera types.

    Defines the interface that all camera implementations must follow.
    """

    def __init__(self, name: str, url: str, fps: int = 10, resolution: Optional[Tuple[int, int]] = None):
        """
        Initialize the camera.

        Args:
            name: Unique identifier for the camera
            url: URL or device ID for the camera
            fps: Target frames per second to process
            resolution: Desired resolution as (width, height)
        """
        self.name = name
        self.url = url
        self.target_fps = fps
        self.resolution = resolution
        self.is_connected = False
        self.is_running = False
        self.last_frame = None
        self.last_frame_time = 0
        self.frame_lock = Lock()
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.reconnect_delay = 5  # seconds

        logger.info(f"Initialized camera '{name}' with URL: {url}")

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the camera.

        Returns:
            bool: True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Release resources and disconnect from the camera."""
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the camera.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and frame if successful
        """
        pass

    def start(self) -> None:
        """Start the frame acquisition thread."""
        if self.is_running:
            logger.warning(f"Camera '{self.name}' is already running")
            return

        if not self.is_connected and not self.connect():
            logger.error(f"Failed to start camera '{self.name}': Connection failed")
            return

        self.is_running = True
        self.acquisition_thread = Thread(target=self._acquisition_loop, daemon=True)
        self.acquisition_thread.start()
        logger.info(f"Started acquisition thread for camera '{self.name}'")

    def stop(self) -> None:
        """Stop the frame acquisition thread."""
        self.is_running = False
        if hasattr(self, 'acquisition_thread') and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=1.0)
        self.disconnect()
        logger.info(f"Stopped camera '{self.name}'")

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the most recent frame from the camera.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and frame if available
        """
        with self.frame_lock:
            if self.last_frame is None:
                return False, None
            return True, self.last_frame.copy()

    def _acquisition_loop(self) -> None:
        """
        Main loop for frame acquisition.

        Runs in a separate thread to continuously fetch frames from the camera
        at the target frame rate.
        """
        while self.is_running:
            # Calculate time to wait to achieve target FPS
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            sleep_time = max(0, 1.0/self.target_fps - elapsed)

            if sleep_time > 0:
                time.sleep(sleep_time)

            # Read frame
            success, frame = self.read_frame()

            if success:
                self.connection_attempts = 0
                with self.frame_lock:
                    self.last_frame = frame
                    self.last_frame_time = time.time()
            else:
                self.connection_attempts += 1
                logger.warning(f"Failed to read frame from camera '{self.name}' "
                              f"(attempt {self.connection_attempts}/{self.max_connection_attempts})")

                if self.connection_attempts >= self.max_connection_attempts:
                    logger.error(f"Too many failed attempts for camera '{self.name}', reconnecting...")
                    self.disconnect()
                    time.sleep(self.reconnect_delay)
                    if not self.connect():
                        logger.error(f"Reconnection failed for camera '{self.name}'")
                        self.is_running = False
                    else:
                        self.connection_attempts = 0


class RTSPCamera(Camera):
    """Camera implementation for RTSP streams."""

    def __init__(self, name: str, url: str, fps: int = 10, resolution: Optional[Tuple[int, int]] = None):
        """
        Initialize an RTSP camera.

        Args:
            name: Unique identifier for the camera
            url: RTSP URL for the camera
            fps: Target frames per second to process
            resolution: Desired resolution as (width, height)
        """
        super().__init__(name, url, fps, resolution)
        self.cap = None

    def connect(self) -> bool:
        """
        Connect to the RTSP stream.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Configure OpenCV to use FFMPEG backend for RTSP
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

            # Set buffer size to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set resolution if specified
            if self.resolution:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Check if connection is successful
            if not self.cap.isOpened():
                logger.error(f"Failed to open RTSP stream: {self.url}")
                return False

            self.is_connected = True
            logger.info(f"Successfully connected to RTSP camera '{self.name}'")
            return True

        except Exception as e:
            logger.error(f"Error connecting to RTSP camera '{self.name}': {str(e)}")
            return False

    def disconnect(self) -> None:
        """Release the OpenCV VideoCapture resources."""
        if self.cap and self.is_connected:
            self.cap.release()
            self.is_connected = False
            logger.info(f"Disconnected from RTSP camera '{self.name}'")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the RTSP stream.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and frame if successful
        """
        if not self.is_connected or not self.cap:
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning(f"Failed to read frame from RTSP camera '{self.name}'")
                return False, None

            return True, frame

        except Exception as e:
            logger.error(f"Error reading frame from RTSP camera '{self.name}': {str(e)}")
            return False, None


class HTTPCamera(Camera):
    """Camera implementation for HTTP streams (MJPEG, etc.)."""

    def __init__(self, name: str, url: str, fps: int = 10, resolution: Optional[Tuple[int, int]] = None):
        """
        Initialize an HTTP camera.

        Args:
            name: Unique identifier for the camera
            url: HTTP URL for the camera stream
            fps: Target frames per second to process
            resolution: Desired resolution as (width, height)
        """
        super().__init__(name, url, fps, resolution)
        self.cap = None

    def connect(self) -> bool:
        """
        Connect to the HTTP stream.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.url)

            # Set buffer size to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set resolution if specified
            if self.resolution:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Check if connection is successful
            if not self.cap.isOpened():
                logger.error(f"Failed to open HTTP stream: {self.url}")
                return False

            self.is_connected = True
            logger.info(f"Successfully connected to HTTP camera '{self.name}'")
            return True

        except Exception as e:
            logger.error(f"Error connecting to HTTP camera '{self.name}': {str(e)}")
            return False

    def disconnect(self) -> None:
        """Release the OpenCV VideoCapture resources."""
        if self.cap and self.is_connected:
            self.cap.release()
            self.is_connected = False
            logger.info(f"Disconnected from HTTP camera '{self.name}'")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the HTTP stream.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and frame if successful
        """
        if not self.is_connected or not self.cap:
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning(f"Failed to read frame from HTTP camera '{self.name}'")
                return False, None

            return True, frame

        except Exception as e:
            logger.error(f"Error reading frame from HTTP camera '{self.name}': {str(e)}")
            return False, None


class WebcamCamera(Camera):
    """Camera implementation for local webcams."""

    def __init__(self, name: str, device_id: int = 0, fps: int = 30, resolution: Optional[Tuple[int, int]] = None):
        """
        Initialize a webcam camera.

        Args:
            name: Unique identifier for the camera
            device_id: Device ID for the webcam (usually 0 for the default camera)
            fps: Target frames per second to process
            resolution: Desired resolution as (width, height)
        """
        super().__init__(name, str(device_id), fps, resolution)
        self.device_id = device_id
        self.cap = None


class VideoFileCamera(Camera):
    """Camera implementation for video files."""

    def __init__(self, name: str, file_path: str, fps: Optional[int] = None, resolution: Optional[Tuple[int, int]] = None, loop: bool = True):
        """
        Initialize a video file camera.

        Args:
            name: Unique identifier for the camera
            file_path: Path to the video file
            fps: Target frames per second to process (if None, uses video's native FPS)
            resolution: Desired resolution as (width, height)
            loop: Whether to loop the video when it reaches the end
        """
        self.file_path = file_path
        self.loop = loop
        self.cap = None
        self.video_fps = 0
        self.frame_count = 0
        self.current_frame = 0

        # Open the video file to get its properties
        try:
            temp_cap = cv2.VideoCapture(file_path)
            if temp_cap.isOpened():
                self.video_fps = temp_cap.get(cv2.CAP_PROP_FPS)
                self.frame_count = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                temp_cap.release()
        except Exception as e:
            logger.error(f"Error getting video properties: {str(e)}")

        # If fps is not specified, use the video's native FPS
        target_fps = fps if fps is not None else self.video_fps

        super().__init__(name, file_path, target_fps, resolution)

    def connect(self) -> bool:
        """
        Connect to the webcam.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.device_id)

            # Set resolution if specified
            if self.resolution:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Check if connection is successful
            if not self.cap.isOpened():
                logger.error(f"Failed to open webcam with device ID: {self.device_id}")
                return False

            self.is_connected = True
            logger.info(f"Successfully connected to webcam '{self.name}'")
            return True

        except Exception as e:
            logger.error(f"Error connecting to webcam '{self.name}': {str(e)}")
            return False

    def disconnect(self) -> None:
        """Release the OpenCV VideoCapture resources."""
        if self.cap and self.is_connected:
            self.cap.release()
            self.is_connected = False
            logger.info(f"Disconnected from webcam '{self.name}'")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the webcam.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and frame if successful
        """
        if not self.is_connected or not self.cap:
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning(f"Failed to read frame from webcam '{self.name}'")
                return False, None

            return True, frame

        except Exception as e:
            logger.error(f"Error reading frame from webcam '{self.name}': {str(e)}")
            return False, None


class VideoFileCamera(Camera):
    """Camera implementation for video files."""

    def __init__(self, name: str, file_path: str, fps: Optional[int] = None, resolution: Optional[Tuple[int, int]] = None, loop: bool = True):
        """
        Initialize a video file camera.

        Args:
            name: Unique identifier for the camera
            file_path: Path to the video file
            fps: Target frames per second to process (if None, uses video's native FPS)
            resolution: Desired resolution as (width, height)
            loop: Whether to loop the video when it reaches the end
        """
        self.file_path = file_path
        self.loop = loop
        self.cap = None
        self.video_fps = 0
        self.frame_count = 0
        self.current_frame = 0

        # Open the video file to get its properties
        try:
            temp_cap = cv2.VideoCapture(file_path)
            if temp_cap.isOpened():
                self.video_fps = temp_cap.get(cv2.CAP_PROP_FPS)
                self.frame_count = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                temp_cap.release()
        except Exception as e:
            logger.error(f"Error getting video properties: {str(e)}")

        # If fps is not specified, use the video's native FPS
        target_fps = fps if fps is not None else self.video_fps

        super().__init__(name, file_path, target_fps, resolution)

    def connect(self) -> bool:
        """
        Connect to the video file.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.file_path)

            # Set resolution if specified
            if self.resolution:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Check if connection is successful
            if not self.cap.isOpened():
                logger.error(f"Failed to open video file: {self.file_path}")
                return False

            self.is_connected = True
            self.current_frame = 0
            logger.info(f"Successfully opened video file '{self.name}' with {self.frame_count} frames at {self.video_fps} FPS")
            return True

        except Exception as e:
            logger.error(f"Error opening video file '{self.name}': {str(e)}")
            return False

    def disconnect(self) -> None:
        """Release the OpenCV VideoCapture resources."""
        if self.cap and self.is_connected:
            self.cap.release()
            self.is_connected = False
            logger.info(f"Closed video file '{self.name}'")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video file.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and frame if successful
        """
        if not self.is_connected or not self.cap:
            return False, None

        try:
            ret, frame = self.cap.read()
            self.current_frame += 1

            # If we've reached the end of the video and looping is enabled, reset to the beginning
            if not ret and self.loop and self.frame_count > 0:
                logger.info(f"Reached end of video '{self.name}', looping back to start")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                ret, frame = self.cap.read()

            if not ret or frame is None:
                if self.loop:
                    logger.warning(f"Failed to read frame from video file '{self.name}'")
                else:
                    logger.info(f"Reached end of video file '{self.name}'")
                return False, None

            return True, frame

        except Exception as e:
            logger.error(f"Error reading frame from video file '{self.name}': {str(e)}")
            return False, None


class OnlineVideoCamera(VideoFileCamera):
    """Camera implementation for online videos that are downloaded and cached locally."""

    def __init__(self, name: str, url: str, fps: Optional[int] = None, resolution: Optional[Tuple[int, int]] = None, loop: bool = True):
        """
        Initialize an online video camera.

        Args:
            name: Unique identifier for the camera
            url: URL to the online video
            fps: Target frames per second to process (if None, uses video's native FPS)
            resolution: Desired resolution as (width, height)
            loop: Whether to loop the video when it reaches the end
        """
        # Create cache directory if it doesn't exist
        cache_dir = Path("cache/videos")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate a filename for the cached video
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = cache_dir / f"{name}_{url_hash}.mp4"

        self.url = url
        self.cache_path = str(cache_path)
        self.is_downloaded = False

        # Initialize with cache path
        super().__init__(name, self.cache_path, fps, resolution, loop)

    def connect(self) -> bool:
        """
        Connect to the online video, downloading it first if necessary.

        Returns:
            bool: True if connection successful, False otherwise
        """
        # Check if the video is already downloaded
        if not os.path.exists(self.cache_path):
            logger.info(f"Downloading online video '{self.name}' from {self.url}")
            try:
                import requests
                from tqdm import tqdm

                # Download the video
                response = requests.get(self.url, stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte

                with open(self.cache_path, 'wb') as file, tqdm(
                    desc=f"Downloading {self.name}",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(block_size):
                        size = file.write(data)
                        bar.update(size)

                self.is_downloaded = True
                logger.info(f"Successfully downloaded video '{self.name}' to {self.cache_path}")
            except Exception as e:
                logger.error(f"Error downloading video '{self.name}': {str(e)}")
                return False
        else:
            self.is_downloaded = True
            logger.info(f"Using cached video '{self.name}' from {self.cache_path}")

        # Now connect to the downloaded video file
        return super().connect()


def create_camera(camera_config: Dict[str, Any]) -> Optional[Camera]:
    """
    Factory function to create a camera instance based on configuration.

    Args:
        camera_config: Dictionary containing camera configuration

    Returns:
        Optional[Camera]: Camera instance or None if creation failed
    """
    try:
        name = camera_config.get('name')
        url = camera_config.get('url')
        camera_type = camera_config.get('type', 'rtsp').lower()
        fps = camera_config.get('fps', 10)
        resolution = camera_config.get('resolution')
        loop = camera_config.get('loop', True)

        if not name or not url:
            logger.error("Camera configuration missing required fields (name, url)")
            return None

        if camera_type == 'rtsp':
            return RTSPCamera(name, url, fps, resolution)
        elif camera_type == 'http':
            return HTTPCamera(name, url, fps, resolution)
        elif camera_type == 'webcam':
            try:
                device_id = int(url)
            except ValueError:
                logger.error(f"Invalid device ID for webcam: {url}")
                return None
            return WebcamCamera(name, device_id, fps, resolution)
        elif camera_type == 'video':
            return VideoFileCamera(name, url, fps, resolution, loop)
        elif camera_type == 'online_video':
            return OnlineVideoCamera(name, url, fps, resolution, loop)
        else:
            logger.error(f"Unsupported camera type: {camera_type}")
            return None

    except Exception as e:
        logger.error(f"Error creating camera: {str(e)}")
        return None


def find_video_files(directory: str = 'videos') -> List[Dict[str, Any]]:
    """
    Find video files in the specified directory and its subdirectories.

    Args:
        directory: Directory to search for video files

    Returns:
        List[Dict[str, Any]]: List of camera configurations for found video files
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_configs = []

    # Get the absolute path to the videos directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    videos_dir = os.path.join(base_dir, directory)

    try:
        if not os.path.exists(videos_dir):
            logger.warning(f"Videos directory not found: {videos_dir}")
            # Create the videos directory if it doesn't exist
            os.makedirs(videos_dir, exist_ok=True)
            logger.info(f"Created videos directory: {videos_dir}")
            return []

        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(videos_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    file_path = os.path.join(root, file)

                    # Create a camera name based on the file name
                    # Remove extension and clean up the name
                    name = os.path.splitext(file)[0]
                    # Replace underscores with spaces and capitalize words
                    name = ' '.join(word.capitalize() for word in name.replace('_', ' ').split())

                    # If the file is in a subdirectory, add the subdirectory name to the camera name
                    rel_dir = os.path.relpath(root, videos_dir)
                    if rel_dir != '.':
                        # Format the subdirectory path as part of the name
                        subdir_name = rel_dir.replace(os.path.sep, ' - ').replace('_', ' ')
                        name = f"{subdir_name} - {name}"

                    # Create a camera configuration for this video file
                    video_configs.append({
                        'name': name,
                        'url': file_path,
                        'type': 'video',
                        'fps': None,  # Use video's native FPS
                        'loop': True
                    })

        logger.info(f"Found {len(video_configs)} video files in {videos_dir} and its subdirectories")

        # If no local videos found, add some online sample videos
        if not video_configs:
            logger.info("No local videos found, adding online sample videos")

            # Sample videos from public sources
            online_videos = [
                {
                    'name': 'Street Traffic',
                    'url': 'https://cdn.pixabay.com/vimeo/328726494/street-23849.mp4?width=640&hash=c4a1e7c9b5a7b5e5e1f2e2e2e2e2e2e2e2e2e2e2',
                    'type': 'online_video',
                    'fps': 25,
                    'loop': True
                },
                {
                    'name': 'Pedestrians',
                    'url': 'https://cdn.pixabay.com/vimeo/414678210/people-27664.mp4?width=640&hash=c4a1e7c9b5a7b5e5e1f2e2e2e2e2e2e2e2e2e2e2',
                    'type': 'online_video',
                    'fps': 25,
                    'loop': True
                },
                {
                    'name': 'Highway Traffic',
                    'url': 'https://cdn.pixabay.com/vimeo/330303968/highway-24021.mp4?width=640&hash=c4a1e7c9b5a7b5e5e1f2e2e2e2e2e2e2e2e2e2e2',
                    'type': 'online_video',
                    'fps': 25,
                    'loop': True
                }
            ]

            video_configs.extend(online_videos)
            logger.info(f"Added {len(online_videos)} online sample videos")

        return video_configs

    except Exception as e:
        logger.error(f"Error finding video files: {str(e)}")
        return []
