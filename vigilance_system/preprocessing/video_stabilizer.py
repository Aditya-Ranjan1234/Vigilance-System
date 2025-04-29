"""
Video stabilization module for reducing camera shake.

This module provides functionality to stabilize video frames from shaky camera feeds.
"""

import cv2
import time
import numpy as np
from typing import List, Tuple, Optional, Deque, Any
from collections import deque

from vigilance_system.utils.logger import get_logger
from vigilance_system.utils.config import config
from vigilance_system.analysis.metrics_collector import metrics_collector

# Initialize logger
logger = get_logger(__name__)


class VideoStabilizer:
    """
    Stabilizes video frames to reduce camera shake.

    Uses various algorithms to estimate and correct motion between frames.
    """

    def __init__(self, smoothing_radius: int = 15, method: str = 'optical_flow'):
        """
        Initialize the video stabilizer.

        Args:
            smoothing_radius: Number of frames to consider for smoothing motion
            method: Stabilization method ('optical_flow', 'feature_based', or 'deep_learning')
        """
        self.smoothing_radius = smoothing_radius
        self.method = method
        self.prev_gray = None
        self.transforms = deque(maxlen=smoothing_radius)
        self.smoothed_transforms = deque(maxlen=smoothing_radius)
        self.camera_name = None
        self.frame_count = 0

        # Initialize stabilizer based on method
        self.stabilizer = self._create_stabilizer()

        logger.info(f"Initialized video stabilizer with method: {method}, "
                   f"smoothing radius: {smoothing_radius}")

    def _create_stabilizer(self) -> Optional[Any]:
        """
        Create a stabilizer based on the method.

        Returns:
            Stabilizer instance or None if method is not supported
        """
        try:
            # Non-deep learning algorithms
            if self.method == 'feature_matching':
                from vigilance_system.preprocessing.algorithms.feature_matching_stabilizer import FeatureMatchingStabilizer
                return FeatureMatchingStabilizer()

            elif self.method == 'orb':
                from vigilance_system.preprocessing.algorithms.orb_stabilizer import ORBStabilizer
                return ORBStabilizer()

            elif self.method == 'sift':
                from vigilance_system.preprocessing.algorithms.sift_stabilizer import SIFTStabilizer
                return SIFTStabilizer()

            elif self.method == 'affine_transform':
                from vigilance_system.preprocessing.algorithms.affine_transform_stabilizer import AffineTransformStabilizer
                return AffineTransformStabilizer()

            # Legacy deep learning algorithms (disabled)
            elif self.method == 'optical_flow':
                logger.warning(f"Deep learning algorithm '{self.method}' is disabled. Using feature_matching instead.")
                from vigilance_system.preprocessing.algorithms.feature_matching_stabilizer import FeatureMatchingStabilizer
                return FeatureMatchingStabilizer()

            elif self.method == 'feature_based':
                logger.warning(f"Deep learning algorithm '{self.method}' is disabled. Using orb instead.")
                from vigilance_system.preprocessing.algorithms.orb_stabilizer import ORBStabilizer
                return ORBStabilizer()

            elif self.method == 'deep_learning':
                logger.warning(f"Deep learning algorithm '{self.method}' is disabled. Using sift instead.")
                from vigilance_system.preprocessing.algorithms.sift_stabilizer import SIFTStabilizer
                return SIFTStabilizer()

            else:
                logger.warning(f"Unknown method: {self.method}, using feature_matching")
                from vigilance_system.preprocessing.algorithms.feature_matching_stabilizer import FeatureMatchingStabilizer
                return FeatureMatchingStabilizer()

        except ImportError as e:
            logger.warning(f"Failed to import stabilizer for method: {self.method}, error: {str(e)}")
            logger.warning("Using legacy implementation")
            return None

    def set_camera_name(self, camera_name: str) -> None:
        """
        Set the camera name for metrics collection.

        Args:
            camera_name: Name of the camera
        """
        self.camera_name = camera_name
        if self.stabilizer is not None:
            self.stabilizer.set_camera_name(camera_name)

    def stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Stabilize a single frame.

        Args:
            frame: Input frame to stabilize

        Returns:
            np.ndarray: Stabilized frame
        """
        # Record start time for metrics
        start_time = time.time()

        # Increment frame count
        self.frame_count += 1

        if frame is None:
            logger.warning("Received None frame for stabilization")
            return None

        # Use new stabilizer if available
        if self.stabilizer is not None:
            stabilized_frame = self.stabilizer.stabilize(frame)
            return stabilized_frame

        # Legacy implementation
        # Convert to grayscale for motion estimation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize with first frame
        if self.prev_gray is None:
            self.prev_gray = gray

            # Record metrics
            metrics_collector.add_metric('preprocessing', 'processing_time', 0, self.camera_name)
            metrics_collector.add_metric('preprocessing', 'stability_score', 1.0, self.camera_name)

            return frame

        # Estimate transform between frames
        if self.method == 'optical_flow':
            transform = self._estimate_transform_optical_flow(gray)
        else:  # feature_matching
            transform = self._estimate_transform_feature_matching(gray)

        # If transform estimation failed, return original frame
        if transform is None:
            # Record metrics
            end_time = time.time()
            processing_time = end_time - start_time
            metrics_collector.add_metric('preprocessing', 'processing_time', processing_time * 1000, self.camera_name)
            metrics_collector.add_metric('preprocessing', 'stability_score', 1.0, self.camera_name)

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

        # Calculate stability score
        stability_score = self._calculate_stability_score(transform)

        # Record metrics
        end_time = time.time()
        processing_time = end_time - start_time
        metrics_collector.add_metric('preprocessing', 'processing_time', processing_time * 1000, self.camera_name)
        metrics_collector.add_metric('preprocessing', 'stability_score', stability_score, self.camera_name)

        return stabilized_frame

    def _calculate_stability_score(self, transform: np.ndarray) -> float:
        """
        Calculate a stability score based on the transformation.

        Args:
            transform: Transformation matrix

        Returns:
            Stability score (0-1, higher is better)
        """
        # Extract translation and rotation from transformation matrix
        dx = transform[0, 2]
        dy = transform[1, 2]
        angle = np.arctan2(transform[1, 0], transform[0, 0]) * 180 / np.pi

        # Calculate stability score
        translation_score = 1.0 - min(1.0, (abs(dx) + abs(dy)) / 100.0)
        rotation_score = 1.0 - min(1.0, abs(angle) / 10.0)

        # Combine scores
        stability_score = 0.7 * translation_score + 0.3 * rotation_score

        return stability_score

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

    # Get algorithm from configuration
    algorithm = config.get('preprocessing.algorithm', 'optical_flow')

    # Create and return stabilizer
    stabilizer = VideoStabilizer(smoothing_radius, algorithm)

    logger.info(f"Created video stabilizer with algorithm: {algorithm}")

    return stabilizer
