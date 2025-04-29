"""
Optical flow-based video stabilizer implementation.

This module provides a video stabilizer based on optical flow.
"""

import time
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.preprocessing.algorithms.base_stabilizer import BaseStabilizer

logger = get_logger(__name__)


class OpticalFlowStabilizer(BaseStabilizer):
    """
    Optical flow-based video stabilizer implementation.
    
    This class implements a video stabilizer that uses optical flow
    to track features and stabilize the video.
    """
    
    def __init__(self):
        """Initialize the optical flow stabilizer."""
        super().__init__()
        
        # Get optical flow specific configuration
        self.max_corners = config.get(f'{self.algorithm_config}.max_corners', 200)
        self.quality_level = config.get(f'{self.algorithm_config}.quality_level', 0.01)
        self.min_distance = config.get(f'{self.algorithm_config}.min_distance', 30)
        
        # Initialize state
        self.prev_gray = None
        self.prev_points = None
        
        logger.info(f"Initialized optical flow stabilizer with "
                   f"max_corners={self.max_corners}, quality_level={self.quality_level}, "
                   f"min_distance={self.min_distance}")
    
    def get_name(self) -> str:
        """
        Get the name of the stabilizer.
        
        Returns:
            Name of the stabilizer
        """
        return 'optical_flow'
    
    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        Stabilize a frame using optical flow.
        
        Args:
            frame: Input frame
        
        Returns:
            Stabilized frame
        """
        # Record start time for metrics
        start_time = time.time()
        
        # Increment frame count
        self.frame_count += 1
        
        # Check if stabilization is enabled
        if not self.enabled:
            # Record metrics
            self._record_metrics(start_time, 1.0)
            return frame
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize previous frame if needed
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_corners, qualityLevel=self.quality_level,
                minDistance=self.min_distance
            )
            
            # Initialize transforms with identity matrix
            self.transforms = [np.eye(2, 3, dtype=np.float32)]
            
            # Record metrics
            self._record_metrics(start_time, 1.0)
            
            return frame
        
        # Calculate optical flow
        if self.prev_points is not None and len(self.prev_points) > 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Select good points
            if next_points is not None and status is not None:
                good_old = self.prev_points[status == 1]
                good_new = next_points[status == 1]
                
                # Estimate transformation
                if len(good_old) >= 4 and len(good_new) >= 4:
                    # Estimate affine transform
                    transform = cv2.estimateAffinePartial2D(good_old, good_new)[0]
                    
                    if transform is not None:
                        # Add to transforms list
                        self.transforms.append(transform)
                        
                        # Smooth transforms
                        smoothed_transform = self._smooth_transform(transform)
                        
                        # Apply smoothed transform
                        stabilized = cv2.warpAffine(
                            frame, smoothed_transform, (frame.shape[1], frame.shape[0])
                        )
                        
                        # Calculate stability score
                        stability_score = self._calculate_stability_score(transform)
                        
                        # Update state
                        self.prev_gray = gray
                        self.prev_points = cv2.goodFeaturesToTrack(
                            gray, maxCorners=self.max_corners, qualityLevel=self.quality_level,
                            minDistance=self.min_distance
                        )
                        
                        # Record metrics
                        self._record_metrics(start_time, stability_score)
                        
                        return stabilized
        
        # Fallback: find new features
        self.prev_gray = gray
        self.prev_points = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=self.quality_level,
            minDistance=self.min_distance
        )
        
        # Record metrics
        self._record_metrics(start_time, 1.0)
        
        return frame
    
    def _smooth_transform(self, transform: np.ndarray) -> np.ndarray:
        """
        Smooth a transformation using a moving average.
        
        Args:
            transform: Current transformation matrix
        
        Returns:
            Smoothed transformation matrix
        """
        # Limit transforms list size
        if len(self.transforms) > self.smoothing_radius * 2:
            self.transforms.pop(0)
        
        # Get window of transforms
        window_size = min(len(self.transforms), self.smoothing_radius)
        window = self.transforms[-window_size:]
        
        # Calculate average transform
        avg_transform = np.zeros_like(transform)
        for t in window:
            avg_transform += t
        avg_transform /= len(window)
        
        return avg_transform
