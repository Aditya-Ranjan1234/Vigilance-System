"""
Loitering detector factory for the Vigilance System.

This module provides a factory for creating loitering detectors based on the configuration.
"""

from typing import Dict, List, Any, Optional

import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import Detection
from vigilance_system.alert.algorithms.base_loitering_detector import BaseLoiteringDetector, LoiteringEvent

logger = get_logger(__name__)


class LoiteringDetector:
    """
    Factory class for creating loitering detectors.

    This class creates and manages loitering detectors based on the configuration.
    """

    def __init__(self):
        """Initialize the loitering detector factory."""
        self.algorithm = config.get('alerts.loitering.algorithm', 'time_threshold')
        self.detector = self._create_detector()
        self.camera_name = None

    def _create_detector(self) -> BaseLoiteringDetector:
        """
        Create a detector based on the configuration.

        Returns:
            BaseLoiteringDetector: The created detector
        """
        # Non-deep learning algorithms
        if self.algorithm == 'rule_based':
            from vigilance_system.alert.algorithms.rule_based_loitering import RuleBasedLoiteringDetector
            return RuleBasedLoiteringDetector()

        elif self.algorithm == 'timer_threshold':
            from vigilance_system.alert.algorithms.timer_threshold_loitering import TimerThresholdLoiteringDetector
            return TimerThresholdLoiteringDetector()

        elif self.algorithm == 'decision_tree':
            from vigilance_system.alert.algorithms.decision_tree_loitering import DecisionTreeLoiteringDetector
            return DecisionTreeLoiteringDetector()

        # Legacy deep learning algorithms (disabled)
        elif self.algorithm == 'time_threshold':
            logger.warning(f"Deep learning algorithm '{self.algorithm}' is disabled. Using rule_based instead.")
            from vigilance_system.alert.algorithms.rule_based_loitering import RuleBasedLoiteringDetector
            return RuleBasedLoiteringDetector()

        elif self.algorithm == 'trajectory_heatmap':
            logger.warning(f"Deep learning algorithm '{self.algorithm}' is disabled. Using timer_threshold instead.")
            from vigilance_system.alert.algorithms.timer_threshold_loitering import TimerThresholdLoiteringDetector
            return TimerThresholdLoiteringDetector()

        elif self.algorithm == 'lstm_prediction':
            logger.warning(f"Deep learning algorithm '{self.algorithm}' is disabled. Using decision_tree instead.")
            from vigilance_system.alert.algorithms.decision_tree_loitering import DecisionTreeLoiteringDetector
            return DecisionTreeLoiteringDetector()

        else:
            logger.warning(f"Unknown algorithm: {self.algorithm}, using rule_based")
            from vigilance_system.alert.algorithms.rule_based_loitering import RuleBasedLoiteringDetector
            return RuleBasedLoiteringDetector()

    def set_camera_name(self, camera_name: str) -> None:
        """
        Set the camera name for metrics collection.

        Args:
            camera_name: Name of the camera
        """
        self.camera_name = camera_name
        self.detector.set_camera_name(camera_name)

    def update(self, detections: List[Detection], frame: np.ndarray, frame_id: int) -> List[LoiteringEvent]:
        """
        Update the loitering detector with new detections.

        Args:
            detections: List of detections with tracking IDs
            frame: Current frame
            frame_id: ID of the current frame

        Returns:
            List of active loitering events
        """
        return self.detector.update(detections, frame, frame_id)

    def draw_events(self, frame: np.ndarray, events: List[LoiteringEvent]) -> np.ndarray:
        """
        Draw loitering events on a frame.

        Args:
            frame: Input frame
            events: List of loitering events to draw

        Returns:
            Frame with loitering events drawn
        """
        return self.detector.draw_events(frame, events)


def create_loitering_detector_from_config() -> LoiteringDetector:
    """
    Create a loitering detector with settings from the configuration.

    Returns:
        LoiteringDetector: Configured loitering detector
    """
    return LoiteringDetector()
