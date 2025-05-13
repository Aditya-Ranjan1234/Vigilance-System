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

    def pause(self) -> None:
        """
        Pause the frame acquisition thread without disconnecting.

        This allows the camera to be resumed later without reconnecting.
        """
        self.is_running = False
        if hasattr(self, 'acquisition_thread') and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=1.0)
        logger.info(f"Paused camera '{self.name}'")

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
            # Handle special 'blank' URL
            if self.file_path == 'blank':
                logger.info(f"Creating a blank video for camera '{self.name}'")
                # Create a blank frame with informative text
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "No video available", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(blank_frame, "This is a fallback blank video", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(blank_frame, "Please add video files to the videos directory", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(blank_frame, "or check your camera configuration", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Add a moving element to make it look like a video
                current_time = time.time()
                position = int(((current_time % 5) / 5) * 540)  # Move across the screen every 5 seconds
                cv2.circle(blank_frame, (50 + position, 300), 20, (0, 165, 255), -1)

                # Store the blank frame
                self.blank_frame = blank_frame
                self.is_connected = True
                self.current_frame = 0
                self.video_fps = 25
                self.frame_count = 1000  # Simulate a long video
                return True

            # Handle image files (jpg, png, etc.) as static videos
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            if any(self.file_path.lower().endswith(ext) for ext in image_extensions):
                logger.info(f"Loading image as static video: {self.file_path}")
                try:
                    # Read the image
                    image = cv2.imread(self.file_path)
                    if image is not None:
                        # Store the image as a static frame
                        self.blank_frame = image
                        self.is_connected = True
                        self.current_frame = 0
                        self.video_fps = 25
                        self.frame_count = 1000  # Simulate a long video
                        logger.info(f"Successfully loaded image as static video: {self.file_path}")
                        return True
                    else:
                        logger.error(f"Failed to load image: {self.file_path}")
                except Exception as img_e:
                    logger.error(f"Error loading image: {str(img_e)}")

            # Check if the file exists
            if not os.path.exists(self.file_path):
                logger.error(f"Video file does not exist: {self.file_path}")

                # Check if this is a WhatsApp video and try to find it in the surveillance directory
                found_whatsapp_video = False
                if "WhatsApp Video" in self.file_path:
                    surveillance_dir = 'D:\\Main EL\\videos\\surveillance'
                    if os.path.exists(surveillance_dir):
                        logger.info(f"Looking for WhatsApp videos in {surveillance_dir}")
                        try:
                            files = os.listdir(surveillance_dir)
                            whatsapp_videos = [f for f in files if f.startswith("WhatsApp Video")]
                            if whatsapp_videos:
                                # Use the first WhatsApp video found
                                whatsapp_video = os.path.join(surveillance_dir, whatsapp_videos[0])
                                logger.info(f"Found alternative WhatsApp video: {whatsapp_video}")
                                self.file_path = whatsapp_video
                                # Skip to the video opening code
                                if os.path.exists(self.file_path):
                                    logger.info(f"Using WhatsApp video: {self.file_path}")
                                    found_whatsapp_video = True
                        except Exception as e:
                            logger.error(f"Error looking for WhatsApp videos: {str(e)}")

                # If not a WhatsApp video or no WhatsApp videos found, try sample videos
                if not os.path.exists(self.file_path) and not found_whatsapp_video:
                    # Try to find a video file in the surveillance directory
                    surveillance_dir = os.path.join("D:", "Main EL", "videos", "surveillance")
                    sample_videos = []

                    # First check if surveillance directory exists
                    if os.path.exists(surveillance_dir):
                        logger.info(f"Checking surveillance directory: {surveillance_dir}")
                        try:
                            # List all files in the surveillance directory
                            files = os.listdir(surveillance_dir)
                            logger.info(f"Files in surveillance directory: {files}")

                            # Add all video files from the surveillance directory
                            for file in files:
                                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                                    video_path = os.path.join(surveillance_dir, file)
                                    sample_videos.append(video_path)
                                    logger.info(f"Added video from surveillance dir: {video_path}")
                        except Exception as e:
                            logger.error(f"Error listing surveillance directory: {str(e)}")
                    else:
                        logger.warning(f"Surveillance directory not found: {surveillance_dir}")

                    # If no videos found in surveillance directory, add fallback options
                    if not sample_videos:
                        logger.warning("No videos found in surveillance directory, using fallbacks")
                        sample_videos = [
                            # First check surveillance directory
                            os.path.join("D:", "Main EL", "videos", "surveillance", "WhatsApp Video 2025-04-22 at 20.45.45_529ae150.mp4"),
                            os.path.join("D:", "Main EL", "videos", "surveillance", "WhatsApp Video 2025-04-22 at 20.47.50_dd1a37f6.mp4"),
                            os.path.join("D:", "Main EL", "videos", "surveillance", "WhatsApp Video 2025-04-22 at 20.48.20_3c3ca36c.mp4"),
                            # Then check other possible locations
                            os.path.join("videos", "surveillance", "WhatsApp Video 2025-04-22 at 20.45.45_529ae150.mp4"),
                            os.path.join("videos", "surveillance", "WhatsApp Video 2025-04-22 at 20.47.50_dd1a37f6.mp4"),
                            os.path.join("videos", "surveillance", "WhatsApp Video 2025-04-22 at 20.48.20_3c3ca36c.mp4"),
                            os.path.join("videos", "samples", "sample1.mp4"),
                            os.path.join("videos", "samples", "sample2.mp4"),
                            os.path.join("videos", "samples", "sample3.mp4"),
                            os.path.join("vigilance_system", "videos", "sample1.mp4"),
                            os.path.join("vigilance_system", "videos", "sample2.mp4"),
                            os.path.join("vigilance_system", "videos", "sample3.mp4"),
                        ]

                for sample_path in sample_videos:
                    if os.path.exists(sample_path):
                        logger.info(f"Using sample video instead: {sample_path}")
                        self.file_path = sample_path
                        break
                else:
                    # If no sample videos found, create a blank video
                    logger.warning("No sample videos found, creating a blank video frame")
                    # Create a blank frame with informative text
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "No video available", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Truncate filename if too long
                    filename = os.path.basename(self.file_path)
                    if len(filename) > 30:
                        filename = filename[:27] + "..."

                    cv2.putText(blank_frame, f"File not found: {filename}", (50, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Add specific instructions for the correct path
                    cv2.putText(blank_frame, "Please place videos in:", (50, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(blank_frame, "D:\\Main EL\\videos\\surveillance", (50, 230),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Add a moving element to make it look like a video
                    current_time = time.time()
                    position = int(((current_time % 5) / 5) * 540)  # Move across the screen every 5 seconds
                    cv2.circle(blank_frame, (50 + position, 300), 20, (0, 165, 255), -1)

                    # Store the blank frame
                    self.blank_frame = blank_frame
                    self.is_connected = True
                    self.current_frame = 0
                    self.video_fps = 25
                    self.frame_count = 1000  # Simulate a long video
                    return True

            # Log the file path for debugging
            logger.info(f"Attempting to open video file: {self.file_path}")

            # Check if the file path exists directly - don't override with a different video
            if os.path.exists(self.file_path):
                logger.info(f"File exists at path: {self.file_path}")
                self.cap = cv2.VideoCapture(self.file_path)
            else:
                # If no WhatsApp videos found, try the original path
                # Use absolute path to ensure the file is found
                abs_path = os.path.abspath(self.file_path)
                logger.info(f"Using absolute path: {abs_path}")

                # Double check that the file exists
                if not os.path.exists(abs_path):
                    logger.error(f"Absolute path does not exist: {abs_path}")
                    # Try using the raw path string directly
                    logger.info(f"Trying raw file path: {self.file_path}")
                    self.cap = cv2.VideoCapture(self.file_path)
                else:
                    logger.info(f"File exists at absolute path: {abs_path}")
                    self.cap = cv2.VideoCapture(abs_path)

            # Set resolution if specified
            if self.resolution:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Check if connection is successful
            if not self.cap.isOpened():
                logger.error(f"Failed to open video file: {abs_path}")
                # Create a blank frame as fallback
                logger.warning("Creating a blank video frame as fallback")
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Video loading error", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(blank_frame, f"Failed to open: {os.path.basename(self.file_path)}", (50, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(blank_frame, "Please check your video format", (50, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Add a moving element to make it look like a video
                current_time = time.time()
                position = int(((current_time % 5) / 5) * 540)  # Move across the screen every 5 seconds
                cv2.circle(blank_frame, (50 + position, 300), 20, (0, 165, 255), -1)

                self.blank_frame = blank_frame
                self.is_connected = True
                self.current_frame = 0
                self.video_fps = 25
                self.frame_count = 1000  # Simulate a long video
                return True

            # Get video properties
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.video_fps <= 0:
                self.video_fps = 25  # Default to 25 fps if invalid

            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.frame_count <= 0:
                self.frame_count = 1000  # Default to 1000 frames if invalid

            self.is_connected = True
            self.current_frame = 0
            logger.info(f"Successfully opened video file '{self.name}' with {self.frame_count} frames at {self.video_fps} FPS")
            return True

        except Exception as e:
            logger.error(f"Error opening video file '{self.name}': {str(e)}")
            # Create a blank frame as fallback
            logger.warning("Creating a blank video frame due to exception")
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Error loading video", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(blank_frame, f"Error: {str(e)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(blank_frame, "Please check your video configuration", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add a moving element to make it look like a video
            current_time = time.time()
            position = int(((current_time % 5) / 5) * 540)  # Move across the screen every 5 seconds
            cv2.circle(blank_frame, (50 + position, 300), 20, (0, 165, 255), -1)

            self.blank_frame = blank_frame
            self.is_connected = True
            self.current_frame = 0
            self.video_fps = 25
            self.frame_count = 1000  # Simulate a long video
            return True

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
        # If we have a blank frame (fallback), return it with some animation
        if hasattr(self, 'blank_frame'):
            # Create a copy to avoid modifying the original
            frame = self.blank_frame.copy()

            # Add a moving element to make it look like a video
            current_time = time.time()
            position = int(((current_time % 5) / 5) * 540)  # Move across the screen every 5 seconds

            # Draw a moving circle
            cv2.circle(frame, (50 + position, 300), 20, (0, 165, 255), -1)

            # Add a timestamp to show it's updating
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            cv2.putText(frame, f"Time: {timestamp}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            return True, frame

        if not self.is_connected:
            logger.warning(f"Cannot read frame: camera '{self.name}' is not connected")
            return False, None

        if not self.cap:
            logger.warning(f"Cannot read frame: no video capture for camera '{self.name}'")
            # Create a blank frame as fallback
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "No video capture", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(blank_frame, f"Camera: {self.name}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(blank_frame, "Please check your camera configuration", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.blank_frame = blank_frame
            return True, self.blank_frame.copy()

        try:
            ret, frame = self.cap.read()

            # If we've reached the end of the video and looping is enabled, reset to the beginning
            if not ret and self.loop and self.frame_count > 0:
                logger.info(f"Reached end of video '{self.name}', looping back to start")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                ret, frame = self.cap.read()

                # If still can't read after resetting, try reopening the file
                if not ret:
                    logger.warning(f"Failed to loop video '{self.name}', trying to reopen")
                    self.disconnect()
                    if self.connect():
                        ret, frame = self.cap.read()

            if not ret or frame is None:
                logger.warning(f"Failed to read frame from video file '{self.name}'")
                # Create a blank frame as fallback if we don't have one yet
                if not hasattr(self, 'blank_frame'):
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "End of video", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(blank_frame, f"Video: {os.path.basename(self.file_path)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(blank_frame, "Video has ended or is corrupted", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    self.blank_frame = blank_frame

                # Create a copy to avoid modifying the original
                frame = self.blank_frame.copy()

                # Add a moving element to make it look like a video
                current_time = time.time()
                position = int(((current_time % 5) / 5) * 540)  # Move across the screen every 5 seconds
                cv2.circle(frame, (50 + position, 300), 20, (0, 165, 255), -1)

                # Add a timestamp to show it's updating
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                cv2.putText(frame, f"Time: {timestamp}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                return True, frame

            self.current_frame += 1
            return True, frame

        except Exception as e:
            logger.error(f"Error reading frame from video file '{self.name}': {str(e)}")
            # Create a blank frame as fallback if we don't have one yet
            if not hasattr(self, 'blank_frame'):
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Error reading video", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(blank_frame, f"Error: {str(e)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(blank_frame, "Please check your video file", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                self.blank_frame = blank_frame

            # Create a copy to avoid modifying the original
            frame = self.blank_frame.copy()

            # Add a moving element to make it look like a video
            current_time = time.time()
            position = int(((current_time % 5) / 5) * 540)  # Move across the screen every 5 seconds
            cv2.circle(frame, (50 + position, 300), 20, (0, 165, 255), -1)

            # Add a timestamp to show it's updating
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            cv2.putText(frame, f"Time: {timestamp}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # Try to reconnect
            self.disconnect()
            self.connect()
            return True, frame


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
    Prioritizes the D:\\Main EL\\videos\\surveillance directory as specified by the user.

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

    # Primary surveillance directory - this is where videos should be
    surveillance_dir = 'D:\\Main EL\\videos\\surveillance'

    # ALWAYS use the surveillance directory first
    if os.path.exists(surveillance_dir):
        logger.info(f"Found surveillance directory at {surveillance_dir}")
        try:
            files = os.listdir(surveillance_dir)
            logger.info(f"Files in surveillance directory: {files}")

            # Check for video files
            video_files = [f for f in files if any(f.lower().endswith(ext) for ext in video_extensions)]
            if video_files:
                logger.info(f"Found {len(video_files)} video files in surveillance directory: {video_files}")

                # Create camera configs for each video file in the surveillance directory
                for file in video_files:
                    file_path = os.path.join(surveillance_dir, file)
                    # Create a camera name based on the file name
                    name = os.path.splitext(file)[0]
                    name = ' '.join(word.capitalize() for word in name.replace('_', ' ').split())

                    # Create a camera configuration for this video file
                    camera_config = {
                        'name': name,
                        'url': file_path,
                        'type': 'video',
                        'fps': None,  # Use video's native FPS
                        'loop': True
                    }

                    # Add the camera configuration
                    video_configs.append(camera_config)
                    logger.info(f"Added camera config for surveillance video: {camera_config}")

                # If we found videos in the surveillance directory, return them immediately
                if video_configs:
                    logger.info(f"Using {len(video_configs)} videos from surveillance directory")
                    return video_configs
            else:
                logger.warning(f"No video files found in surveillance directory")
        except Exception as e:
            logger.error(f"Error listing files in surveillance directory: {str(e)}")
    else:
        logger.warning(f"Surveillance directory not found at {surveillance_dir}, creating it...")
        try:
            os.makedirs(surveillance_dir, exist_ok=True)
            logger.info(f"Created surveillance directory at {surveillance_dir}")
        except Exception as e:
            logger.error(f"Failed to create surveillance directory: {str(e)}")

    # Only check these other directories if no videos found in surveillance_dir
    additional_dirs = [
        os.path.join(base_dir, 'videos', 'surveillance'),
        videos_dir,
        'D:\\Main EL\\videos',
        'D:\\Main EL\\vigilance_system\\videos',
        os.path.join(base_dir, 'vigilance_system', 'videos'),
        os.path.join(base_dir, 'videos'),
        os.path.join(os.getcwd(), 'videos', 'surveillance'),
        os.path.join(os.getcwd(), 'videos'),
        os.path.join(os.getcwd(), 'vigilance_system', 'videos')
    ]

    # Log the existence of the surveillance directory and its contents
    if os.path.exists(surveillance_dir):
        logger.info(f"Surveillance directory exists at: {surveillance_dir}")
        try:
            files = os.listdir(surveillance_dir)
            logger.info(f"Files in surveillance directory: {files}")

            # Check for specific WhatsApp videos
            for file in files:
                if file.startswith("WhatsApp Video"):
                    logger.info(f"Found WhatsApp video: {file}")
        except Exception as e:
            logger.error(f"Error listing files in surveillance directory: {str(e)}")

    logger.info(f"Checking the following directories for videos: {additional_dirs}")

    try:
        # Create the default videos directory if it doesn't exist
        if not os.path.exists(videos_dir):
            logger.warning(f"Default videos directory not found: {videos_dir}")
            os.makedirs(videos_dir, exist_ok=True)
            logger.info(f"Created default videos directory: {videos_dir}")

        # Check all directories for videos
        for dir_path in additional_dirs:
            if os.path.exists(dir_path):
                logger.info(f"Checking for videos in: {dir_path}")

                # Walk through the directory and its subdirectories
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in video_extensions):
                            file_path = os.path.join(root, file)
                            logger.info(f"Found video file: {file_path}")

                            # Create a camera name based on the file name
                            name = os.path.splitext(file)[0]
                            name = ' '.join(word.capitalize() for word in name.replace('_', ' ').split())

                            # If the file is in a subdirectory, add the subdirectory name to the camera name
                            rel_dir = os.path.relpath(root, dir_path)
                            if rel_dir != '.':
                                subdir_name = rel_dir.replace(os.path.sep, ' - ').replace('_', ' ')
                                name = f"{subdir_name} - {name}"

                            # Create a camera configuration for this video file
                            camera_config = {
                                'name': name,
                                'url': file_path,
                                'type': 'video',
                                'fps': None,  # Use video's native FPS
                                'loop': True
                            }

                            # Add the camera configuration
                            video_configs.append(camera_config)
                            logger.info(f"Added camera config: {camera_config}")

                logger.info(f"Found {len(video_configs)} video files in {dir_path} and its subdirectories")

                # If we found videos, no need to check other directories
                if video_configs:
                    logger.info(f"Using videos from {dir_path}")
                    break
            else:
                logger.warning(f"Directory not found: {dir_path}")

        # If no videos found in any directory, log a warning
        if not video_configs:
            logger.warning("No video files found in any of the checked directories")

            # Try to directly use all WhatsApp videos in the surveillance directory
            surveillance_dir = 'D:\\Main EL\\videos\\surveillance'
            if os.path.exists(surveillance_dir):
                logger.info(f"Looking for WhatsApp videos in {surveillance_dir}")
                try:
                    files = os.listdir(surveillance_dir)
                    whatsapp_videos = [f for f in files if f.startswith("WhatsApp Video")]

                    if whatsapp_videos:
                        logger.info(f"Found {len(whatsapp_videos)} WhatsApp videos in surveillance directory")

                        # Add each WhatsApp video to the configs
                        for i, video_file in enumerate(whatsapp_videos):
                            video_path = os.path.join(surveillance_dir, video_file)
                            logger.info(f"Adding WhatsApp video {i+1}: {video_path}")

                            # Create a camera name based on the file name
                            name = f"WhatsApp Video {i+1}"

                            video_configs.append({
                                'name': name,
                                'url': video_path,
                                'type': 'video',
                                'fps': None,  # Use video's native FPS
                                'loop': True
                            })
                            logger.info(f"Added WhatsApp video config: {name} at {video_path}")
                    else:
                        logger.warning("No WhatsApp videos found in surveillance directory")
                except Exception as e:
                    logger.error(f"Error listing files in surveillance directory: {str(e)}")

        logger.info(f"Found {len(video_configs)} video files in total")

        # If no local videos found, create some test videos with colored patterns
        if not video_configs:
            logger.info("No local videos found, creating test pattern videos")

            # Create a directory for test videos if it doesn't exist
            test_videos_dir = os.path.join(videos_dir, 'test')
            os.makedirs(test_videos_dir, exist_ok=True)

            # Create test pattern videos
            test_videos = []

            # Create a test video with a moving pattern
            for pattern_name, color in [
                ('Red Pattern', (0, 0, 255)),
                ('Green Pattern', (0, 255, 0)),
                ('Blue Pattern', (255, 0, 0))
            ]:
                # We'll use an image instead of a video file for simplicity
                # No need to create a video file path

                # Create a test video file with a moving pattern
                try:
                    # Create a blank frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

                    # Add text to the frame
                    cv2.putText(frame, f"Test Video: {pattern_name}", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, "No real video files found", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "Using generated test pattern", (50, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Draw a colored rectangle
                    cv2.rectangle(frame, (50, 200), (590, 400), color, -1)

                    # Save the frame as an image
                    test_image_path = os.path.join(test_videos_dir, f"{pattern_name.lower().replace(' ', '_')}.jpg")
                    cv2.imwrite(test_image_path, frame)

                    # Add the test video to the list
                    test_videos.append({
                        'name': pattern_name,
                        'url': test_image_path,  # Use the image as a static video
                        'type': 'video',
                        'fps': 25,
                        'loop': True
                    })

                    logger.info(f"Created test pattern image: {test_image_path}")
                except Exception as e:
                    logger.error(f"Error creating test pattern video: {str(e)}")

            # Add the test videos to the list
            video_configs.extend(test_videos)
            logger.info(f"Added {len(test_videos)} test pattern videos")

        # If still no videos, add some online sample videos as a last resort
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

        # If we still have no videos, create a blank video as a last resort
        if not video_configs:
            logger.warning("No videos found or created, adding a blank video as last resort")

            # Create a blank video
            blank_video = {
                'name': 'Blank Video',
                'url': 'blank',  # Special URL that will be handled in VideoFileCamera.connect()
                'type': 'video',
                'fps': 25,
                'loop': True
            }

            video_configs.append(blank_video)
            logger.info("Added a blank video as last resort")

        return video_configs

    except Exception as e:
        logger.error(f"Error finding video files: {str(e)}")
        # Return a fallback blank video in case of error
        return [{
            'name': 'Fallback Blank Video',
            'url': 'blank',  # Special URL that will be handled in VideoFileCamera.connect()
            'type': 'video',
            'fps': 25,
            'loop': True
        }]
