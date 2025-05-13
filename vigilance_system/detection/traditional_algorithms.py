"""
Traditional computer vision algorithms for object detection.

This module provides implementations of traditional computer vision algorithms
for object detection, including background subtraction and HOG+SVM.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque

from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.detection_types import Detection

# Initialize logger
logger = get_logger(__name__)


class BackgroundSubtractor:
    """Background subtraction-based object detector."""

    def __init__(self, history: int = 500, threshold: float = 16, detect_shadows: bool = True, learning_rate: float = 0.01):
        """
        Initialize the background subtractor.

        Args:
            history: Number of frames to use for background model
            threshold: Threshold for foreground detection
            detect_shadows: Whether to detect shadows
            learning_rate: Learning rate for background model update
        """
        self.history = history
        self.threshold = threshold
        self.detect_shadows = detect_shadows
        self.learning_rate = learning_rate

        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=threshold,
            detectShadows=detect_shadows
        )

        logger.info(f"Initialized background subtractor with history={history}, threshold={threshold}, "
                   f"detect_shadows={detect_shadows}, learning_rate={learning_rate}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame using background subtraction.

        Args:
            frame: Input frame

        Returns:
            List[Detection]: List of detections
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)

        # Remove shadows (they are marked as gray in the mask)
        if self.detect_shadows:
            fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)[1]

        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create detections from contours
        detections = []
        for contour in contours:
            # Filter out small contours
            if cv2.contourArea(contour) < 500:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Create detection
            detection = Detection(
                bbox=(x, y, x+w, y+h),  # Format as (x1, y1, x2, y2) instead of (x, y, w, h)
                class_id=0,  # Default to person class
                class_name="person",
                confidence=0.8,  # Fixed confidence for background subtraction
                frame_id=0,  # Will be set by the caller
                timestamp=0.0  # Will be set by the caller
            )

            detections.append(detection)

        return detections


class HOGSVMDetector:
    """HOG+SVM-based object detector."""

    def __init__(self, scale: float = 1.05, hit_threshold: float = 0.0, win_stride: Tuple[int, int] = (8, 8)):
        """
        Initialize the HOG+SVM detector.

        Args:
            scale: Scale factor for the detection window
            hit_threshold: Threshold for detection
            win_stride: Window stride for detection
        """
        self.scale = scale
        self.hit_threshold = hit_threshold
        self.win_stride = win_stride

        # Initialize HOG descriptor
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        logger.info(f"Initialized HOG+SVM detector with scale={scale}, hit_threshold={hit_threshold}, "
                   f"win_stride={win_stride}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame using HOG+SVM.

        Args:
            frame: Input frame

        Returns:
            List[Detection]: List of detections
        """
        # Resize frame for faster detection
        height, width = frame.shape[:2]
        max_dimension = 640
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            inverse_scale = 1.0 / scale
        else:
            inverse_scale = 1.0

        # Detect people
        boxes, weights = self.hog.detectMultiScale(
            frame,
            winStride=self.win_stride,
            padding=(16, 16),
            scale=self.scale,
            hitThreshold=self.hit_threshold
        )

        # Create detections from boxes
        detections = []
        for (x, y, w, h), weight in zip(boxes, weights):
            # Scale back to original size
            x = int(x * inverse_scale)
            y = int(y * inverse_scale)
            w = int(w * inverse_scale)
            h = int(h * inverse_scale)

            # Create detection
            detection = Detection(
                bbox=(x, y, x+w, y+h),  # Format as (x1, y1, x2, y2) instead of (x, y, w, h)
                class_id=0,  # Person class
                class_name="person",
                confidence=float(weight[0]),  # Extract the scalar value from the weight array
                frame_id=0,  # Will be set by the caller
                timestamp=0.0  # Will be set by the caller
            )

            detections.append(detection)

        return detections
