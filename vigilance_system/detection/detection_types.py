"""
Detection types module for object detection.

This module provides common data types and classes used in object detection.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class Detection:
    """
    Represents a single object detection.

    Stores information about a detected object, including its bounding box,
    class, confidence, and other metadata.
    """

    def __init__(
        self,
        bbox: List[float],
        class_id: int,
        class_name: str,
        confidence: float,
        frame_id: int = 0,
        timestamp: float = 0.0
    ):
        """
        Initialize a detection.

        Args:
            bbox: Bounding box coordinates [x, y, width, height]
            class_id: Class ID of the detected object
            class_name: Class name of the detected object
            confidence: Confidence score of the detection
            frame_id: ID of the frame this detection belongs to
            timestamp: Timestamp of the detection
        """
        self.bbox = bbox
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.frame_id = frame_id
        self.timestamp = timestamp

        # Calculate center coordinates
        self.center_x = bbox[0] + bbox[2] / 2
        self.center_y = bbox[1] + bbox[3] / 2

    def __str__(self) -> str:
        """
        Get string representation of the detection.

        Returns:
            str: String representation
        """
        return (f"Detection(class={self.class_name}, confidence={self.confidence:.2f}, "
                f"bbox={self.bbox}, center=({self.center_x:.1f}, {self.center_y:.1f}))")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert detection to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'bbox': self.bbox,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'center_x': self.center_x,
            'center_y': self.center_y
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Detection':
        """
        Create detection from dictionary.

        Args:
            data: Dictionary with detection data

        Returns:
            Detection: New detection instance
        """
        detection = cls(
            bbox=data['bbox'],
            class_id=data['class_id'],
            class_name=data['class_name'],
            confidence=data['confidence'],
            frame_id=data.get('frame_id', 0),
            timestamp=data.get('timestamp', 0.0)
        )
        return detection
