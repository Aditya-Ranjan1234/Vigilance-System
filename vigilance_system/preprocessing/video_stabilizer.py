"""
Video stabilization module for reducing camera shake.

This module provides functionality to stabilize video frames from shaky camera feeds.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Deque
from collections import deque

from vigilance_system.utils.logger import get_logger
from vigilance_system.utils.config import config

# Initialize logger
logger = get_logger(__name__)


class VideoStabilizer:
    """
    Stabilizes video frames to reduce camera shake.
    
    Uses optical flow or feature matching to estimate and correct motion between frames.
    """
    
    def __init__(self, smoothing_radius: int = 15, method: str = 'optical_flow'):
        """
        Initialize the video stabilizer.
        
        Args:
            smoothing_radius: Number of frames to consider for smoothing motion
            method: Stabilization method ('optical_flow' or 'feature_matching')
        """
        self.smoothing_radius = smoothing_radius
        self.method = method
        self.prev_gray = None
        self.transforms = deque(maxlen=smoothing_radius)
        self.smoothed_transforms = deque(maxlen=smoothing_radius)
        
        # Parameters for feature matching
        self.feature_detector = cv2.SIFT_create()
        self.feature_matcher = cv2.BFMatcher()
        
        logger.info(f"Initialized video stabilizer with method: {method}, "
                   f"smoothing radius: {smoothing_radius}")
    
    def stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Stabilize a single frame.
        
        Args:
            frame: Input frame to stabilize
        
        Returns:
            np.ndarray: Stabilized frame
        """
        if frame is None:
            logger.warning("Received None frame for stabilization")
            return None
        
        # Convert to grayscale for motion estimation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize with first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame
        
        # Estimate transform between frames
        if self.method == 'optical_flow':
            transform = self._estimate_transform_optical_flow(gray)
        else:  # feature_matching
            transform = self._estimate_transform_feature_matching(gray)
        
        # If transform estimation failed, return original frame
        if transform is None:
            return frame
        
        # Add to transforms queue
        self.transforms.append(transform)
        
        # Calculate smoothed transform
        if len(self.transforms) < 2:
            smoothed_transform = transform
        else:
            smoothed_transform = self._smooth_transform()
        
        self.smoothed_transforms.append(smoothed_transform)
        
        # Apply smoothed transform to frame
        stabilized_frame = self._apply_transform(frame, smoothed_transform)
        
        # Update previous frame
        self.prev_gray = gray
        
        return stabilized_frame
    
    def _estimate_transform_optical_flow(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate transform between frames using optical flow.
        
        Args:
            gray: Current frame in grayscale
        
        Returns:
            Optional[np.ndarray]: 2x3 transformation matrix or None if estimation failed
        """
        try:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Calculate translation
            h, w = flow.shape[:2]
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            dx = np.median(flow_x)
            dy = np.median(flow_y)
            
            # Create transformation matrix
            transform = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            return transform
            
        except Exception as e:
            logger.error(f"Error estimating transform with optical flow: {str(e)}")
            return None
    
    def _estimate_transform_feature_matching(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate transform between frames using feature matching.
        
        Args:
            gray: Current frame in grayscale
        
        Returns:
            Optional[np.ndarray]: 2x3 transformation matrix or None if estimation failed
        """
        try:
            # Detect features
            prev_keypoints, prev_descriptors = self.feature_detector.detectAndCompute(self.prev_gray, None)
            curr_keypoints, curr_descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            if prev_descriptors is None or curr_descriptors is None or len(prev_descriptors) < 2 or len(curr_descriptors) < 2:
                logger.warning("Not enough features detected for matching")
                return None
            
            # Match features
            matches = self.feature_matcher.knnMatch(prev_descriptors, curr_descriptors, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) < 4:
                logger.warning("Not enough good matches for homography")
                return None
            
            # Extract matched keypoints
            prev_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            curr_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            transform, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
            return transform
            
        except Exception as e:
            logger.error(f"Error estimating transform with feature matching: {str(e)}")
            return None
    
    def _smooth_transform(self) -> np.ndarray:
        """
        Smooth the transformation to reduce jitter.
        
        Returns:
            np.ndarray: Smoothed transformation matrix
        """
        # Calculate mean of recent transforms
        mean_dx = np.mean([t[0, 2] for t in self.transforms])
        mean_dy = np.mean([t[1, 2] for t in self.transforms])
        
        # Get the latest transform
        latest_transform = self.transforms[-1].copy()
        
        # Apply smoothing to translation components
        latest_transform[0, 2] = mean_dx
        latest_transform[1, 2] = mean_dy
        
        return latest_transform
    
    def _apply_transform(self, frame: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Apply transformation to frame.
        
        Args:
            frame: Input frame
            transform: Transformation matrix to apply
        
        Returns:
            np.ndarray: Transformed frame
        """
        h, w = frame.shape[:2]
        stabilized_frame = cv2.warpAffine(frame, transform, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return stabilized_frame
    
    def reset(self) -> None:
        """Reset the stabilizer state."""
        self.prev_gray = None
        self.transforms.clear()
        self.smoothed_transforms.clear()
        logger.info("Reset video stabilizer")


def create_stabilizer_from_config() -> Optional[VideoStabilizer]:
    """
    Create a video stabilizer with settings from the configuration.
    
    Returns:
        Optional[VideoStabilizer]: Configured video stabilizer or None if disabled
    """
    # Get stabilization configuration
    stabilization_config = config.get('preprocessing.stabilization', {})
    
    # Check if stabilization is enabled
    if not stabilization_config.get('enabled', False):
        logger.info("Video stabilization is disabled in configuration")
        return None
    
    # Get stabilization parameters
    smoothing_radius = stabilization_config.get('smoothing_radius', 15)
    method = stabilization_config.get('method', 'optical_flow')
    
    # Create and return stabilizer
    return VideoStabilizer(smoothing_radius, method)
