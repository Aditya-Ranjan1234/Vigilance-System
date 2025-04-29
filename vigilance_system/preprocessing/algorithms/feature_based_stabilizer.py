"""
Feature-based video stabilizer implementation.

This module provides a video stabilizer based on feature detection and matching.
"""

import time
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.preprocessing.algorithms.base_stabilizer import BaseStabilizer

logger = get_logger(__name__)


class FeatureBasedStabilizer(BaseStabilizer):
    """
    Feature-based video stabilizer implementation.
    
    This class implements a video stabilizer that uses feature detection
    and matching to stabilize the video.
    """
    
    def __init__(self):
        """Initialize the feature-based stabilizer."""
        super().__init__()
        
        # Get feature-based specific configuration
        self.feature_type = config.get(f'{self.algorithm_config}.feature_type', 'ORB')
        self.max_features = config.get(f'{self.algorithm_config}.max_features', 500)
        
        # Initialize feature detector
        self.detector = self._create_detector()
        
        # Initialize state
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        logger.info(f"Initialized feature-based stabilizer with "
                   f"feature_type={self.feature_type}, max_features={self.max_features}")
    
    def get_name(self) -> str:
        """
        Get the name of the stabilizer.
        
        Returns:
            Name of the stabilizer
        """
        return 'feature_based'
    
    def _create_detector(self) -> Any:
        """
        Create a feature detector based on the configuration.
        
        Returns:
            Feature detector
        """
        if self.feature_type == 'ORB':
            return cv2.ORB_create(nfeatures=self.max_features)
        elif self.feature_type == 'SIFT':
            try:
                return cv2.SIFT_create(nfeatures=self.max_features)
            except AttributeError:
                logger.warning("SIFT not available, falling back to ORB")
                return cv2.ORB_create(nfeatures=self.max_features)
        else:
            logger.warning(f"Unknown feature type: {self.feature_type}, falling back to ORB")
            return cv2.ORB_create(nfeatures=self.max_features)
    
    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        Stabilize a frame using feature detection and matching.
        
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
        
        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        # Initialize previous frame if needed
        if self.prev_gray is None or self.prev_keypoints is None or self.prev_descriptors is None:
            self.prev_gray = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            
            # Initialize transforms with identity matrix
            self.transforms = [np.eye(2, 3, dtype=np.float32)]
            
            # Record metrics
            self._record_metrics(start_time, 1.0)
            
            return frame
        
        # Match features
        if descriptors is not None and self.prev_descriptors is not None and len(keypoints) > 0 and len(self.prev_keypoints) > 0:
            # Create matcher
            if self.feature_type == 'SIFT':
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches = matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
                
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                good_matches = matcher.match(self.prev_descriptors, descriptors)
            
            # Extract matched keypoints
            if len(good_matches) >= 4:
                src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Estimate transformation
                transform, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                
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
                    self.prev_keypoints = keypoints
                    self.prev_descriptors = descriptors
                    
                    # Record metrics
                    self._record_metrics(start_time, stability_score)
                    
                    return stabilized
        
        # Fallback: use current frame as reference
        self.prev_gray = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
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
