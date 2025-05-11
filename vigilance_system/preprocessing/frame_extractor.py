"""
Frame extractor module for processing video streams.

This module provides functionality to extract frames from video streams
at a specified rate and with optional preprocessing.
"""

import time
import cv2
import numpy as np
from typing import Optional, Tuple, List, Callable

from vigilance_system.utils.logger import get_logger
from vigilance_system.utils.config import config

# Initialize logger
logger = get_logger(__name__)


class FrameExtractor:
    """
    Extracts and preprocesses frames from video streams.
    
    Handles frame rate control, buffering, and basic preprocessing operations.
    """
    
    def __init__(self, target_fps: int = 10, buffer_size: int = 5):
        """
        Initialize the frame extractor.
        
        Args:
            target_fps: Target frames per second to extract
            buffer_size: Number of frames to keep in the buffer
        """
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.frame_buffer: List[np.ndarray] = []
        self.last_extraction_time = 0
        self.preprocessing_pipeline: List[Callable[[np.ndarray], np.ndarray]] = []
        
        logger.info(f"Initialized frame extractor with target FPS: {target_fps}")
    
    def add_preprocessing_step(self, step: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Add a preprocessing step to the pipeline.
        
        Args:
            step: Function that takes a frame and returns a processed frame
        """
        self.preprocessing_pipeline.append(step)
        logger.info(f"Added preprocessing step: {step.__name__}")
    
    def clear_preprocessing_pipeline(self) -> None:
        """Clear all preprocessing steps."""
        self.preprocessing_pipeline = []
        logger.info("Cleared preprocessing pipeline")
    
    def should_extract_frame(self) -> bool:
        """
        Check if enough time has passed to extract a new frame.
        
        Returns:
            bool: True if a new frame should be extracted, False otherwise
        """
        current_time = time.time()
        elapsed = current_time - self.last_extraction_time
        return elapsed >= 1.0 / self.target_fps
    
    def extract_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract and preprocess a frame.
        
        Args:
            frame: Input frame to process
        
        Returns:
            np.ndarray: Processed frame
        """
        self.last_extraction_time = time.time()
        
        # Apply preprocessing steps
        processed_frame = frame.copy()
        for step in self.preprocessing_pipeline:
            processed_frame = step(processed_frame)
        
        # Add to buffer
        self.frame_buffer.append(processed_frame)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        return processed_frame
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame from the buffer.
        
        Returns:
            Optional[np.ndarray]: Latest frame or None if buffer is empty
        """
        if not self.frame_buffer:
            return None
        return self.frame_buffer[-1]
    
    def get_frame_buffer(self) -> List[np.ndarray]:
        """
        Get all frames in the buffer.
        
        Returns:
            List[np.ndarray]: List of frames in the buffer
        """
        return self.frame_buffer.copy()


# Common preprocessing functions

def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize a frame to the specified dimensions.
    
    Args:
        frame: Input frame
        width: Target width
        height: Target height
    
    Returns:
        np.ndarray: Resized frame
    """
    return cv2.resize(frame, (width, height))


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] range.
    
    Args:
        frame: Input frame
    
    Returns:
        np.ndarray: Normalized frame
    """
    return frame.astype(np.float32) / 255.0


def denoise_frame(frame: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Apply denoising to a frame.
    
    Args:
        frame: Input frame
        strength: Denoising strength (higher = more denoising)
    
    Returns:
        np.ndarray: Denoised frame
    """
    return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)


def adjust_brightness_contrast(frame: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    """
    Adjust brightness and contrast of a frame.
    
    Args:
        frame: Input frame
        alpha: Contrast control (1.0 means no change)
        beta: Brightness control (0 means no change)
    
    Returns:
        np.ndarray: Adjusted frame
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def create_frame_extractor_from_config() -> FrameExtractor:
    """
    Create a frame extractor with settings from the configuration.
    
    Returns:
        FrameExtractor: Configured frame extractor
    """
    # Get preprocessing configuration
    preprocessing_config = config.get('preprocessing', {})
    target_fps = config.get('preprocessing.target_fps', 10)
    buffer_size = config.get('preprocessing.buffer_size', 5)
    
    # Create frame extractor
    extractor = FrameExtractor(target_fps, buffer_size)
    
    # Add denoising if enabled
    if preprocessing_config.get('denoising', {}).get('enabled', False):
        strength = preprocessing_config.get('denoising', {}).get('strength', 10)
        extractor.add_preprocessing_step(lambda frame: denoise_frame(frame, strength))
    
    return extractor
