"""
KNN Background Subtraction Detector implementation.

This module provides a KNN background subtraction based detector
that uses classical computer vision techniques instead of deep learning.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any

from vigilance_system.detection.algorithms.base_detector import BaseDetector
from vigilance_system.utils.logger import get_logger

logger = get_logger(__name__)


class KNNDetector(BaseDetector):
    """
    KNN Background Subtraction based object detector.

    Uses KNN background subtraction algorithm to detect moving objects in the frame,
    then applies simple classification to identify humans.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the KNN detector.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "knn"
        self.min_contour_area = 500
        self.confidence_threshold = 0.6

        # Additional parameters specific to KNN
        self.learning_rate = 0.01

        if config:
            self.min_contour_area = config.get('min_contour_area', 500)
            self.confidence_threshold = config.get('confidence_threshold', 0.6)
            self.learning_rate = config.get('learning_rate', 0.01)

        # Initialize the background subtractor
        self.bg_subtractor = None
        self.load_model()

        logger.info(f"Initialized {self.name} detector")

    def load_model(self) -> None:
        """
        Load the KNN background subtraction model.

        For KNN, this creates the background subtractor object.
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=500,
            dist2Threshold=400.0,
            detectShadows=True
        )

        logger.info("KNN background subtraction model loaded")

    def detect(self, frame: np.ndarray) -> List[Any]:
        """
        Detect objects in the frame using KNN background subtraction.

        Args:
            frame: Input frame

        Returns:
            List of Detection objects
        """
        from vigilance_system.detection.algorithms.base_detector import Detection

        # Record start time for metrics
        start_time = time.time()

        # Apply KNN background subtraction with learning rate
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)

        # Remove shadows (gray pixels)
        _, binary_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < self.min_contour_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Simple human classification based on aspect ratio and size
            aspect_ratio = h / w
            is_human = 1.5 < aspect_ratio < 4.0  # Typical human aspect ratio

            # Calculate confidence based on contour area and aspect ratio
            confidence = 0.0
            if is_human:
                # Normalize area between min_contour_area and 5000
                area_score = min(1.0, (cv2.contourArea(contour) - self.min_contour_area) / (5000 - self.min_contour_area))
                # Aspect ratio score (1.0 when aspect_ratio is 2.5, decreasing as it deviates)
                ratio_score = 1.0 - min(1.0, abs(aspect_ratio - 2.5) / 1.0)
                confidence = 0.5 + 0.5 * (0.7 * area_score + 0.3 * ratio_score)

            # Only add detections with sufficient confidence
            if confidence >= self.confidence_threshold:
                # Create a Detection object
                bbox = (x, y, x + w, y + h)
                class_id = 0  # 0 = person in COCO dataset
                class_name = 'person'

                detection = Detection(
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence
                )

                detections.append(detection)

        # Record metrics
        self._record_metrics(start_time, detections)

        return detections
