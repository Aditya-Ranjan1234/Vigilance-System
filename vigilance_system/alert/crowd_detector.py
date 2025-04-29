"""
Crowd detector factory for the Vigilance System.

This module provides a factory for creating crowd detectors based on the configuration.
"""

from typing import Dict, List, Any, Optional

import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import Detection
from vigilance_system.alert.algorithms.base_crowd_detector import BaseCrowdDetector, CrowdEvent

logger = get_logger(__name__)


class CrowdDetector:
    """
    Factory class for creating crowd detectors.

    This class creates and manages crowd detectors based on the configuration.
    """

    def __init__(self):
        """Initialize the crowd detector factory."""
        self.algorithm = config.get('alerts.crowd.algorithm', 'count_threshold')
        self.detector = self._create_detector()
        self.camera_name = None

    def _create_detector(self) -> BaseCrowdDetector:
        """
        Create a detector based on the configuration.

        Returns:
            BaseCrowdDetector: The created detector
        """
        # Non-deep learning algorithms
        if self.algorithm == 'blob_counting':
            from vigilance_system.alert.algorithms.blob_counting_crowd import BlobCountingCrowdDetector
            return BlobCountingCrowdDetector()

        elif self.algorithm == 'contour_counting':
            from vigilance_system.alert.algorithms.contour_counting_crowd import ContourCountingCrowdDetector
            return ContourCountingCrowdDetector()

        elif self.algorithm == 'kmeans_clustering':
            from vigilance_system.alert.algorithms.kmeans_clustering_crowd import KMeansClusteringCrowdDetector
            return KMeansClusteringCrowdDetector()

        # Legacy deep learning algorithms (disabled)
        elif self.algorithm == 'count_threshold':
            logger.warning(f"Deep learning algorithm '{self.algorithm}' is disabled. Using blob_counting instead.")
            from vigilance_system.alert.algorithms.blob_counting_crowd import BlobCountingCrowdDetector
            return BlobCountingCrowdDetector()

        elif self.algorithm == 'density_map':
            logger.warning(f"Deep learning algorithm '{self.algorithm}' is disabled. Using contour_counting instead.")
            from vigilance_system.alert.algorithms.contour_counting_crowd import ContourCountingCrowdDetector
            return ContourCountingCrowdDetector()

        elif self.algorithm == 'clustering':
            logger.warning(f"Deep learning algorithm '{self.algorithm}' is disabled. Using kmeans_clustering instead.")
            from vigilance_system.alert.algorithms.kmeans_clustering_crowd import KMeansClusteringCrowdDetector
            return KMeansClusteringCrowdDetector()

        else:
            logger.warning(f"Unknown algorithm: {self.algorithm}, using blob_counting")
            from vigilance_system.alert.algorithms.blob_counting_crowd import BlobCountingCrowdDetector
            return BlobCountingCrowdDetector()

    def set_camera_name(self, camera_name: str) -> None:
        """
        Set the camera name for metrics collection.

        Args:
            camera_name: Name of the camera
        """
        self.camera_name = camera_name
        self.detector.set_camera_name(camera_name)

    def update(self, detections: List[Detection], frame: np.ndarray, frame_id: int) -> List[CrowdEvent]:
        """
        Update the crowd detector with new detections.

        Args:
            detections: List of detections
            frame: Current frame
            frame_id: ID of the current frame

        Returns:
            List of active crowd events
        """
        return self.detector.update(detections, frame, frame_id)

    def draw_events(self, frame: np.ndarray, events: List[CrowdEvent]) -> np.ndarray:
        """
        Draw crowd events on a frame.

        Args:
            frame: Input frame
            events: List of crowd events to draw

        Returns:
            Frame with crowd events drawn
        """
        return self.detector.draw_events(frame, events)


def create_crowd_detector_from_config() -> CrowdDetector:
    """
    Create a crowd detector with settings from the configuration.

    Returns:
        CrowdDetector: Configured crowd detector
    """
    return CrowdDetector()
