"""
Base loitering detector class for all loitering detection algorithms.

This module provides a base class that all loitering detection algorithms should inherit from.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import Detection
from vigilance_system.analysis.metrics_collector import metrics_collector

logger = get_logger(__name__)


class LoiteringEvent:
    """
    Class representing a loitering event.
    
    Attributes:
        id (int): Unique ID for the event
        track_id (int): ID of the track that triggered the event
        start_time (float): Time when the loitering started
        duration (float): Duration of the loitering in seconds
        location (Tuple[float, float]): Location of the loitering (x, y)
        confidence (float): Confidence score of the loitering detection
        frame_id (int): ID of the frame where the loitering was detected
        bbox (Tuple[float, float, float, float]): Bounding box of the loitering object
    """
    
    def __init__(self, track_id: int, location: Tuple[float, float], bbox: Tuple[float, float, float, float], 
                frame_id: int, confidence: float = 1.0):
        """
        Initialize a loitering event.
        
        Args:
            track_id: ID of the track that triggered the event
            location: Location of the loitering (x, y)
            bbox: Bounding box of the loitering object (x1, y1, x2, y2)
            frame_id: ID of the frame where the loitering was detected
            confidence: Confidence score of the loitering detection
        """
        self.id = None  # Will be set by the detector
        self.track_id = track_id
        self.start_time = time.time()
        self.duration = 0.0
        self.location = location
        self.confidence = confidence
        self.frame_id = frame_id
        self.bbox = bbox
        self.is_active = True
    
    def update(self, location: Tuple[float, float], bbox: Tuple[float, float, float, float], 
              frame_id: int, confidence: float = None) -> None:
        """
        Update the loitering event.
        
        Args:
            location: New location of the loitering (x, y)
            bbox: New bounding box of the loitering object
            frame_id: ID of the current frame
            confidence: New confidence score (if None, keep the current one)
        """
        self.location = location
        self.bbox = bbox
        self.frame_id = frame_id
        if confidence is not None:
            self.confidence = confidence
        self.duration = time.time() - self.start_time
    
    def end(self) -> None:
        """End the loitering event."""
        self.is_active = False
        self.duration = time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the loitering event to a dictionary.
        
        Returns:
            Dictionary representation of the loitering event
        """
        return {
            'id': self.id,
            'track_id': self.track_id,
            'start_time': self.start_time,
            'duration': self.duration,
            'location': self.location,
            'confidence': self.confidence,
            'frame_id': self.frame_id,
            'bbox': self.bbox,
            'is_active': self.is_active
        }


class BaseLoiteringDetector(ABC):
    """
    Base class for all loitering detection algorithms.
    
    This class defines the interface that all loitering detection algorithms
    should implement.
    """
    
    def __init__(self, config_prefix: str = 'alerts.loitering'):
        """
        Initialize the loitering detector.
        
        Args:
            config_prefix: Prefix for configuration keys
        """
        self.config_prefix = config_prefix
        self.algorithm_name = self.get_name()
        self.algorithm_config = f'{config_prefix}.algorithms.{self.algorithm_name}'
        self.loitering_events = []
        self.next_event_id = 1
        self.frame_count = 0
        self.camera_name = None
        
        logger.info(f"Initializing {self.__class__.__name__}")
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the loitering detector.
        
        Returns:
            Name of the loitering detector
        """
        pass
    
    @abstractmethod
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
        pass
    
    def set_camera_name(self, camera_name: str) -> None:
        """
        Set the camera name for metrics collection.
        
        Args:
            camera_name: Name of the camera
        """
        self.camera_name = camera_name
    
    def _record_metrics(self, start_time: float, events: List[LoiteringEvent]) -> None:
        """
        Record metrics for the loitering detection.
        
        Args:
            start_time: Start time of the loitering detection
            events: List of loitering events
        """
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Record metrics
        metrics_collector.add_metric('loitering', 'processing_time', processing_time * 1000, self.camera_name)
        metrics_collector.add_metric('loitering', 'event_count', len(events), self.camera_name)
        
        # Record true positives, false positives, etc.
        # These would need to be calculated based on ground truth
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        metrics_collector.add_metric('loitering', 'true_positives', true_positives, self.camera_name)
        metrics_collector.add_metric('loitering', 'false_positives', false_positives, self.camera_name)
        metrics_collector.add_metric('loitering', 'false_negatives', false_negatives, self.camera_name)
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        metrics_collector.add_metric('loitering', 'precision', precision, self.camera_name)
        metrics_collector.add_metric('loitering', 'recall', recall, self.camera_name)
    
    def draw_events(self, frame: np.ndarray, events: List[LoiteringEvent]) -> np.ndarray:
        """
        Draw loitering events on a frame.
        
        Args:
            frame: Input frame
            events: List of loitering events to draw
        
        Returns:
            Frame with loitering events drawn
        """
        result = frame.copy()
        
        for event in events:
            if not event.is_active:
                continue
                
            # Get bounding box
            x1, y1, x2, y2 = map(int, event.bbox)
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label = f"Loitering: {event.duration:.1f}s"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 0, 255), -1)
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw location point
            cx, cy = map(int, event.location)
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
        
        return result
