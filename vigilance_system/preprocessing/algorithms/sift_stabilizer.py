"""
SIFT Feature-based Video Stabilizer implementation.

This module provides a SIFT feature-based video stabilization approach
without using deep learning.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from vigilance_system.preprocessing.algorithms.base_stabilizer import BaseStabilizer
from vigilance_system.utils.logger import get_logger

logger = get_logger(__name__)


class SIFTStabilizer(BaseStabilizer):
    """
    SIFT Feature-based video stabilizer.
    
    Stabilizes video using SIFT feature detection and matching between consecutive frames.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the SIFT stabilizer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "sift"
        
        # Parameters
        self.max_features = config.get('max_features', 500)
        self.match_threshold = config.get('match_threshold', 0.7)
        
        # Smoothing parameters
        self.smoothing_radius = config.get('smoothing_radius', 30)  # Number of frames for smoothing
        
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(nfeatures=self.max_features)
        
        # Initialize feature matcher
        self.matcher = cv2.BFMatcher()
        
        # State
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
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
        
        # Detect SIFT features
        kp, des = self.sift.detectAndCompute(gray, None)
        
        # Initialize on first frame
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            
            # Identity transform for first frame
            transform = np.eye(2, 3, dtype=np.float32)
            self.transform_history.append(transform)
            return frame
        
        # Match features
        if des is not None and self.prev_des is not None and len(des) > 0 and len(self.prev_des) > 0:
            matches = self.matcher.knnMatch(self.prev_des, des, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < self.match_threshold * n.distance:
                    good_matches.append(m)
            
            # Extract matched keypoints
            if len(good_matches) >= 4:  # Need at least 4 points for homography
                src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find transformation matrix
                transform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                
                if transform is None:
                    # Use identity transform if estimation fails
                    transform = np.eye(2, 3, dtype=np.float32)
            else:
                # Use identity transform if not enough matches
                transform = np.eye(2, 3, dtype=np.float32)
        else:
            # Use identity transform if feature detection fails
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
        
        # Update previous frame and features
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des
        
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
