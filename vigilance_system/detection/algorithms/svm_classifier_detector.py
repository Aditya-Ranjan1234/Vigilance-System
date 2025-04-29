"""
SVM Classifier Detector implementation.

This module provides an SVM-based detector that uses HOG features
and classical computer vision techniques instead of deep learning.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any

from vigilance_system.detection.algorithms.base_detector import BaseDetector
from vigilance_system.utils.logger import get_logger

logger = get_logger(__name__)


class SVMClassifierDetector(BaseDetector):
    """
    SVM Classifier based object detector.

    Uses HOG (Histogram of Oriented Gradients) features with SVM classifier
    to detect humans in the frame.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the SVM classifier detector.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "svm_classifier"

        # Parameters
        self.scale = 1.05
        self.win_stride = (8, 8)
        self.padding = (16, 16)
        self.confidence_threshold = 0.3

        if config:
            self.scale = config.get('scale', 1.05)
            self.win_stride = config.get('win_stride', (8, 8))
            self.padding = config.get('padding', (16, 16))
            self.confidence_threshold = config.get('confidence_threshold', 0.3)

        # Initialize HOG descriptor
        self.hog = None
        self.load_model()

        logger.info(f"Initialized {self.name} detector")

    def load_model(self) -> None:
        """
        Load the SVM classifier model.

        For SVM classifier, this initializes the HOG descriptor and sets the SVM detector.
        """
        self.hog = cv2.HOGDescriptor()
        # Use the pre-trained SVM coefficients for people detection
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        logger.info("SVM classifier model loaded")

    def detect(self, frame: np.ndarray) -> List[Any]:
        """
        Detect objects in the frame using HOG+SVM.

        Args:
            frame: Input frame

        Returns:
            List of Detection objects
        """
        from vigilance_system.detection.algorithms.base_detector import Detection

        # Record start time for metrics
        start_time = time.time()

        # Convert to grayscale for better performance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect people using HOG+SVM
        boxes, weights = self.hog.detectMultiScale(
            gray,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale
        )

        detections = []
        for i, (x, y, w, h) in enumerate(boxes):
            confidence = float(weights[i])

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
