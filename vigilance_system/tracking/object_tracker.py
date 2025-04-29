"""
Object tracker factory for the Vigilance System.

This module provides a factory for creating object trackers based on the configuration.
"""

from typing import Dict, List, Any, Optional

import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import Detection
from vigilance_system.tracking.algorithms.base_tracker import BaseTracker

logger = get_logger(__name__)


class ObjectTracker:
    """
    Factory class for creating object trackers.

    This class creates and manages object trackers based on the configuration.
    """

    def __init__(self):
        """Initialize the object tracker factory."""
        self.algorithm = config.get('tracking.algorithm', 'sort')
        self.tracker = self._create_tracker()
        self.camera_name = None

    def _create_tracker(self) -> BaseTracker:
        """
        Create a tracker based on the configuration.

        Returns:
            BaseTracker: The created tracker
        """
        # Non-deep learning algorithms
        if self.algorithm == 'klt_tracker':
            from vigilance_system.tracking.algorithms.klt_tracker import KLTTracker
            return KLTTracker()

        elif self.algorithm == 'kalman_filter':
            from vigilance_system.tracking.algorithms.kalman_filter_tracker import KalmanFilterTracker
            return KalmanFilterTracker()

        elif self.algorithm == 'optical_flow':
            from vigilance_system.tracking.algorithms.optical_flow_tracker import OpticalFlowTracker
            return OpticalFlowTracker()

        # Legacy deep learning algorithms (disabled)
        elif self.algorithm == 'sort':
            logger.warning(f"Deep learning algorithm '{self.algorithm}' is disabled. Using klt_tracker instead.")
            from vigilance_system.tracking.algorithms.klt_tracker import KLTTracker
            return KLTTracker()

        elif self.algorithm == 'deep_sort':
            logger.warning(f"Deep learning algorithm '{self.algorithm}' is disabled. Using klt_tracker instead.")
            from vigilance_system.tracking.algorithms.klt_tracker import KLTTracker
            return KLTTracker()

        elif self.algorithm == 'iou':
            logger.warning(f"Deep learning algorithm '{self.algorithm}' is disabled. Using kalman_filter instead.")
            from vigilance_system.tracking.algorithms.kalman_filter_tracker import KalmanFilterTracker
            return KalmanFilterTracker()

        else:
            logger.warning(f"Unknown algorithm: {self.algorithm}, using klt_tracker")
            from vigilance_system.tracking.algorithms.klt_tracker import KLTTracker
            return KLTTracker()

    def set_camera_name(self, camera_name: str) -> None:
        """
        Set the camera name for metrics collection.

        Args:
            camera_name: Name of the camera
        """
        self.camera_name = camera_name
        self.tracker.set_camera_name(camera_name)

    def update(self, detections: List[Detection], frame: np.ndarray = None) -> List[Detection]:
        """
        Update the tracker with new detections.

        Args:
            detections: List of new detections
            frame: Input frame (required for some trackers)

        Returns:
            List of tracked detections
        """
        if self.algorithm in ['deep_sort', 'optical_flow']:
            return self.tracker.update(detections, frame)
        else:
            return self.tracker.update(detections)


def create_tracker_from_config() -> ObjectTracker:
    """
    Create an object tracker with settings from the configuration.

    Returns:
        ObjectTracker: Configured object tracker
    """
    return ObjectTracker()
