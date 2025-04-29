"""
Feature Matching Video Stabilizer implementation.

This module provides a feature-based video stabilization approach
without using deep learning.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from vigilance_system.preprocessing.algorithms.base_stabilizer import BaseStabilizer
from vigilance_system.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureMatchingStabilizer(BaseStabilizer):
    """
    Feature Matching video stabilizer.
    
    Stabilizes video using feature detection and matching between consecutive frames.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the feature matching stabilizer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "feature_matching"
        
        # Parameters
        self.max_corners = config.get('max_corners', 200)
        self.quality_level = config.get('quality_level', 0.01)
        self.min_distance = config.get('min_distance', 30)
        self.block_size = config.get('block_size', 3)
        
        # Smoothing parameters
        self.smoothing_radius = config.get('smoothing_radius', 30)  # Number of frames for smoothing
        self.smoothing_window = []  # Window of transformations for smoothing
        
        # State
        self.prev_gray = None
        self.prev_points = None
        self.transform_history = []  # History of transformations
        
        logger.info(f"Initialized {self.name} stabilizer")
    
    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        Stabilize a video frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Stabilized frame
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize on first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, 
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size
            )
            
            # Identity transform for first frame
            transform = np.eye(2, 3, dtype=np.float32)
            self.transform_history.append(transform)
            return frame
        
        # Calculate optical flow
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, 
            gray, 
            self.prev_points, 
            None
        )
        
        # Filter out points that couldn't be tracked
        if curr_points is not None and len(curr_points) > 0:
            good_old = self.prev_points[status == 1]
            good_new = curr_points[status == 1]
            
            # Find transformation matrix
            if len(good_old) >= 3 and len(good_new) >= 3:  # Need at least 3 points
                transform = cv2.estimateAffinePartial2D(good_old, good_new)[0]
                
                if transform is None:
                    # Use identity transform if estimation fails
                    transform = np.eye(2, 3, dtype=np.float32)
            else:
                # Use identity transform if not enough points
                transform = np.eye(2, 3, dtype=np.float32)
        else:
            # Use identity transform if tracking fails
            transform = np.eye(2, 3, dtype=np.float32)
        
        # Add to transform history
        self.transform_history.append(transform)
        
        # Limit history length
        if len(self.transform_history) > self.smoothing_radius * 2:
            self.transform_history = self.transform_history[-self.smoothing_radius * 2:]
        
        # Calculate smoothed transform
        smoothed_transform = self._get_smoothed_transform()
        
        # Apply smoothed transform
        h, w = frame.shape[:2]
        stabilized_frame = cv2.warpAffine(frame, smoothed_transform, (w, h))
        
        # Update previous frame and points
        self.prev_gray = gray
        self.prev_points = cv2.goodFeaturesToTrack(
            gray, 
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size
        )
        
        return stabilized_frame
    
    def _get_smoothed_transform(self) -> np.ndarray:
        """
        Calculate smoothed transformation matrix.
        
        Returns:
            Smoothed transformation matrix
        """
        # If not enough history, return the latest transform
        if len(self.transform_history) < 2:
            return self.transform_history[-1]
        
        # Calculate cumulative transforms
        cumul_transforms = []
        cumul_transform = np.eye(2, 3, dtype=np.float32)
        
        for transform in self.transform_history:
            cumul_transform = self._combine_transforms(cumul_transform, transform)
            cumul_transforms.append(cumul_transform.copy())
        
        # Calculate smoothed trajectory
        trajectory = np.array([self._get_transform_params(t) for t in cumul_transforms])
        smoothed_trajectory = self._smooth_trajectory(trajectory)
        
        # Create smoothed transform from the latest smoothed trajectory
        smoothed_params = smoothed_trajectory[-1]
        smoothed_transform = self._get_transform_matrix(smoothed_params)
        
        return smoothed_transform
    
    def _combine_transforms(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """
        Combine two transformation matrices.
        
        Args:
            t1: First transformation matrix
            t2: Second transformation matrix
            
        Returns:
            Combined transformation matrix
        """
        # Convert to 3x3 matrices for matrix multiplication
        t1_3x3 = np.vstack([t1, [0, 0, 1]])
        t2_3x3 = np.vstack([t2, [0, 0, 1]])
        
        # Combine transforms
        result_3x3 = np.matmul(t1_3x3, t2_3x3)
        
        # Convert back to 2x3 matrix
        return result_3x3[:2, :]
    
    def _get_transform_params(self, transform: np.ndarray) -> np.ndarray:
        """
        Extract parameters from transformation matrix.
        
        Args:
            transform: Transformation matrix
            
        Returns:
            Array of parameters [dx, dy, da, ds] where:
                dx: x translation
                dy: y translation
                da: rotation angle
                ds: scale
        """
        dx = transform[0, 2]
        dy = transform[1, 2]
        
        # Calculate rotation angle
        da = np.arctan2(transform[1, 0], transform[0, 0])
        
        # Calculate scale
        ds = np.sqrt(transform[0, 0]**2 + transform[1, 0]**2)
        
        return np.array([dx, dy, da, ds])
    
    def _get_transform_matrix(self, params: np.ndarray) -> np.ndarray:
        """
        Create transformation matrix from parameters.
        
        Args:
            params: Array of parameters [dx, dy, da, ds]
            
        Returns:
            Transformation matrix
        """
        dx, dy, da, ds = params
        
        # Create transformation matrix
        transform = np.zeros((2, 3), dtype=np.float32)
        transform[0, 0] = ds * np.cos(da)
        transform[0, 1] = -ds * np.sin(da)
        transform[1, 0] = ds * np.sin(da)
        transform[1, 1] = ds * np.cos(da)
        transform[0, 2] = dx
        transform[1, 2] = dy
        
        return transform
    
    def _smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Smooth trajectory using moving average.
        
        Args:
            trajectory: Array of trajectory parameters
            
        Returns:
            Smoothed trajectory
        """
        smoothed_trajectory = np.copy(trajectory)
        
        # Apply moving average
        kernel = np.ones(self.smoothing_radius) / self.smoothing_radius
        
        for i in range(trajectory.shape[1]):
            smoothed_trajectory[:, i] = np.convolve(trajectory[:, i], kernel, mode='same')
            
            # Fix boundary effects
            for j in range(self.smoothing_radius // 2):
                # Start
                if j < len(smoothed_trajectory):
                    smoothed_trajectory[j, i] = np.mean(trajectory[:j+self.smoothing_radius//2+1, i])
                
                # End
                if len(smoothed_trajectory) - j - 1 >= 0:
                    smoothed_trajectory[-j-1, i] = np.mean(
                        trajectory[-(j+self.smoothing_radius//2+1):, i]
                    )
        
        return smoothed_trajectory
