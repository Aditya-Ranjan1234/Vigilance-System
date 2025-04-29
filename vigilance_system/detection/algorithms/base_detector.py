"""
Base detector class for all object detection algorithms.

This module provides a base class that all object detection algorithms should inherit from.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.analysis.metrics_collector import metrics_collector

logger = get_logger(__name__)


class Detection:
    """
    Class representing a detection result.
    
    Attributes:
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
        class_id (int): Class ID of the detected object
        class_name (str): Class name of the detected object
        confidence (float): Confidence score of the detection
        tracking_id (int, optional): Tracking ID for the object
    """
    
    def __init__(self, bbox: Tuple[float, float, float, float], class_id: int, 
                class_name: str, confidence: float, tracking_id: Optional[int] = None):
        """
        Initialize a detection.
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            class_id: Class ID of the detected object
            class_name: Class name of the detected object
            confidence: Confidence score of the detection
            tracking_id: Optional tracking ID for the object
        """
        self.bbox = bbox
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.tracking_id = tracking_id
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the detection to a dictionary.
        
        Returns:
            Dictionary representation of the detection
        """
        return {
            'bbox': self.bbox,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'tracking_id': self.tracking_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Detection':
        """
        Create a detection from a dictionary.
        
        Args:
            data: Dictionary representation of a detection
        
        Returns:
            Detection object
        """
        return cls(
            bbox=tuple(data['bbox']),
            class_id=data['class_id'],
            class_name=data['class_name'],
            confidence=data['confidence'],
            tracking_id=data.get('tracking_id')
        )


class BaseDetector(ABC):
    """
    Base class for all object detection algorithms.
    
    This class defines the interface that all object detection algorithms
    should implement.
    """
    
    def __init__(self, config_prefix: str = 'detection'):
        """
        Initialize the detector.
        
        Args:
            config_prefix: Prefix for configuration keys
        """
        self.config_prefix = config_prefix
        self.confidence_threshold = config.get(f'{config_prefix}.confidence_threshold', 0.5)
        self.nms_threshold = config.get(f'{config_prefix}.nms_threshold', 0.45)
        self.classes_of_interest = config.get(f'{config_prefix}.classes_of_interest', None)
        self.device = config.get(f'{config_prefix}.device', 'cpu')
        self.model = None
        self.class_names = []
        self.camera_name = None
        
        logger.info(f"Initializing {self.__class__.__name__} with confidence threshold {self.confidence_threshold}")
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the detection model.
        
        This method should be implemented by each detector to load its specific model.
        """
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame (BGR image)
        
        Returns:
            List of Detection objects
        """
        pass
    
    def set_camera_name(self, camera_name: str) -> None:
        """
        Set the camera name for metrics collection.
        
        Args:
            camera_name: Name of the camera
        """
        self.camera_name = camera_name
    
    def filter_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Filter detections based on classes of interest.
        
        Args:
            detections: List of Detection objects
        
        Returns:
            Filtered list of Detection objects
        """
        if self.classes_of_interest is None:
            return detections
        
        return [d for d in detections if d.class_id in self.classes_of_interest]
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detections on a frame.
        
        Args:
            frame: Input frame (BGR image)
            detections: List of Detection objects
        
        Returns:
            Frame with detections drawn
        """
        result = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            
            # Generate a color based on class ID
            color = self._get_color(detection.class_id)
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            if detection.tracking_id is not None:
                label = f"ID: {detection.tracking_id} | {label}"
                
            # Get text size
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(result, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """
        Get a color for a class ID.
        
        Args:
            class_id: Class ID
        
        Returns:
            BGR color tuple
        """
        # List of distinct colors
        colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 0, 0),    # Dark blue
            (0, 128, 0),    # Dark green
            (0, 0, 128),    # Dark red
            (128, 128, 0),  # Dark cyan
            (128, 0, 128),  # Dark magenta
            (0, 128, 128),  # Dark yellow
            (192, 192, 192) # Light gray
        ]
        
        return colors[class_id % len(colors)]
    
    def _record_metrics(self, start_time: float, detections: List[Detection]) -> None:
        """
        Record metrics for the detection.
        
        Args:
            start_time: Start time of the detection
            detections: List of Detection objects
        """
        # Calculate FPS
        end_time = time.time()
        processing_time = end_time - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        # Record metrics
        metrics_collector.add_metric('detection', 'fps', fps, self.camera_name)
        metrics_collector.add_metric('detection', 'processing_time', processing_time * 1000, self.camera_name)
        metrics_collector.add_metric('detection', 'detection_count', len(detections), self.camera_name)
        
        # Record class-specific metrics
        class_counts = {}
        for detection in detections:
            class_counts[detection.class_id] = class_counts.get(detection.class_id, 0) + 1
        
        for class_id, count in class_counts.items():
            metrics_collector.add_metric('detection', f'class_{class_id}_count', count, self.camera_name)
