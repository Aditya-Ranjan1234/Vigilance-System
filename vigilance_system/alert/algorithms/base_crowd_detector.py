"""
Base crowd detector class for all crowd detection algorithms.

This module provides a base class that all crowd detection algorithms should inherit from.
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


class CrowdEvent:
    """
    Class representing a crowd event.
    
    Attributes:
        id (int): Unique ID for the event
        start_time (float): Time when the crowd event started
        duration (float): Duration of the crowd event in seconds
        location (Tuple[float, float]): Center location of the crowd (x, y)
        confidence (float): Confidence score of the crowd detection
        frame_id (int): ID of the frame where the crowd was detected
        count (int): Number of people in the crowd
        bbox (Tuple[float, float, float, float]): Bounding box of the crowd area
    """
    
    def __init__(self, location: Tuple[float, float], bbox: Tuple[float, float, float, float], 
                frame_id: int, count: int, confidence: float = 1.0):
        """
        Initialize a crowd event.
        
        Args:
            location: Center location of the crowd (x, y)
            bbox: Bounding box of the crowd area (x1, y1, x2, y2)
            frame_id: ID of the frame where the crowd was detected
            count: Number of people in the crowd
            confidence: Confidence score of the crowd detection
        """
        self.id = None  # Will be set by the detector
        self.start_time = time.time()
        self.duration = 0.0
        self.location = location
        self.confidence = confidence
        self.frame_id = frame_id
        self.count = count
        self.bbox = bbox
        self.is_active = True
    
    def update(self, location: Tuple[float, float], bbox: Tuple[float, float, float, float], 
              frame_id: int, count: int, confidence: float = None) -> None:
        """
        Update the crowd event.
        
        Args:
            location: New center location of the crowd (x, y)
            bbox: New bounding box of the crowd area
            frame_id: ID of the current frame
            count: New number of people in the crowd
            confidence: New confidence score (if None, keep the current one)
        """
        self.location = location
        self.bbox = bbox
        self.frame_id = frame_id
        self.count = count
        if confidence is not None:
            self.confidence = confidence
        self.duration = time.time() - self.start_time
    
    def end(self) -> None:
        """End the crowd event."""
        self.is_active = False
        self.duration = time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the crowd event to a dictionary.
        
        Returns:
            Dictionary representation of the crowd event
        """
        return {
            'id': self.id,
            'start_time': self.start_time,
            'duration': self.duration,
            'location': self.location,
            'confidence': self.confidence,
            'frame_id': self.frame_id,
            'count': self.count,
            'bbox': self.bbox,
            'is_active': self.is_active
        }


class BaseCrowdDetector(ABC):
    """
    Base class for all crowd detection algorithms.
    
    This class defines the interface that all crowd detection algorithms
    should implement.
    """
    
    def __init__(self, config_prefix: str = 'alerts.crowd'):
        """
        Initialize the crowd detector.
        
        Args:
            config_prefix: Prefix for configuration keys
        """
        self.config_prefix = config_prefix
        self.algorithm_name = self.get_name()
        self.algorithm_config = f'{config_prefix}.algorithms.{self.algorithm_name}'
        self.crowd_events = []
        self.next_event_id = 1
        self.frame_count = 0
        self.camera_name = None
        
        logger.info(f"Initializing {self.__class__.__name__}")
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the crowd detector.
        
        Returns:
            Name of the crowd detector
        """
        pass
    
    @abstractmethod
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
        pass
    
    def set_camera_name(self, camera_name: str) -> None:
        """
        Set the camera name for metrics collection.
        
        Args:
            camera_name: Name of the camera
        """
        self.camera_name = camera_name
    
    def _record_metrics(self, start_time: float, events: List[CrowdEvent]) -> None:
        """
        Record metrics for the crowd detection.
        
        Args:
            start_time: Start time of the crowd detection
            events: List of crowd events
        """
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Record metrics
        metrics_collector.add_metric('crowd', 'processing_time', processing_time * 1000, self.camera_name)
        metrics_collector.add_metric('crowd', 'event_count', len(events), self.camera_name)
        
        # Record crowd sizes
        if events:
            avg_count = sum(e.count for e in events) / len(events)
            max_count = max(e.count for e in events)
            metrics_collector.add_metric('crowd', 'average_count', avg_count, self.camera_name)
            metrics_collector.add_metric('crowd', 'max_count', max_count, self.camera_name)
        
        # Record MAE and MSE (these would need ground truth)
        mae = 0.0
        mse = 0.0
        metrics_collector.add_metric('crowd', 'mae', mae, self.camera_name)
        metrics_collector.add_metric('crowd', 'mse', mse, self.camera_name)
        
        # Record accuracy, precision, recall (these would need ground truth)
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        metrics_collector.add_metric('crowd', 'accuracy', accuracy, self.camera_name)
        metrics_collector.add_metric('crowd', 'precision', precision, self.camera_name)
        metrics_collector.add_metric('crowd', 'recall', recall, self.camera_name)
    
    def draw_events(self, frame: np.ndarray, events: List[CrowdEvent]) -> np.ndarray:
        """
        Draw crowd events on a frame.
        
        Args:
            frame: Input frame
            events: List of crowd events to draw
        
        Returns:
            Frame with crowd events drawn
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
            label = f"Crowd: {event.count} people"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 0, 255), -1)
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center point
            cx, cy = map(int, event.location)
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
        
        return result
