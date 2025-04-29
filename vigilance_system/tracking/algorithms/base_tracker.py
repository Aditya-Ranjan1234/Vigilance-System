"""
Base tracker class for all object tracking algorithms.

This module provides a base class that all object tracking algorithms should inherit from.
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


class Track:
    """
    Class representing a tracked object.
    
    Attributes:
        id (int): Unique ID for the track
        detection (Detection): Latest detection associated with this track
        history (List[Tuple[float, float]]): History of center positions (x, y)
        age (int): Number of frames since the track was created
        time_since_update (int): Number of frames since the track was last updated
        hits (int): Number of times the track was matched with a detection
        start_time (float): Time when the track was created
        last_update_time (float): Time when the track was last updated
    """
    
    def __init__(self, detection: Detection, track_id: int):
        """
        Initialize a track.
        
        Args:
            detection: Initial detection for this track
            track_id: Unique ID for this track
        """
        self.id = track_id
        self.detection = detection
        self.history = [(detection.bbox[0] + detection.bbox[2]) / 2, 
                        (detection.bbox[1] + detection.bbox[3]) / 2]
        self.age = 1
        self.time_since_update = 0
        self.hits = 1
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.class_id = detection.class_id
        self.class_name = detection.class_name
    
    def update(self, detection: Detection) -> None:
        """
        Update the track with a new detection.
        
        Args:
            detection: New detection to update the track with
        """
        self.detection = detection
        self.history.append(((detection.bbox[0] + detection.bbox[2]) / 2, 
                            (detection.bbox[1] + detection.bbox[3]) / 2))
        self.hits += 1
        self.time_since_update = 0
        self.last_update_time = time.time()
    
    def predict(self) -> None:
        """
        Predict the next position of the track.
        
        This is a simple implementation that just increments the age and time_since_update.
        Subclasses can override this to implement more sophisticated prediction.
        """
        self.age += 1
        self.time_since_update += 1
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the track.
        
        Returns:
            Dictionary with the current state
        """
        return {
            'id': self.id,
            'bbox': self.detection.bbox,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.detection.confidence,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'duration': time.time() - self.start_time
        }
    
    def to_detection(self) -> Detection:
        """
        Convert the track to a detection.
        
        Returns:
            Detection object with tracking ID
        """
        detection = Detection(
            bbox=self.detection.bbox,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.detection.confidence,
            tracking_id=self.id
        )
        return detection


class BaseTracker(ABC):
    """
    Base class for all object tracking algorithms.
    
    This class defines the interface that all object tracking algorithms
    should implement.
    """
    
    def __init__(self, config_prefix: str = 'tracking'):
        """
        Initialize the tracker.
        
        Args:
            config_prefix: Prefix for configuration keys
        """
        self.config_prefix = config_prefix
        self.max_age = config.get(f'{config_prefix}.algorithms.{self.get_name()}.max_age', 30)
        self.min_hits = config.get(f'{config_prefix}.algorithms.{self.get_name()}.min_hits', 3)
        self.iou_threshold = config.get(f'{config_prefix}.algorithms.{self.get_name()}.iou_threshold', 0.3)
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        self.camera_name = None
        
        logger.info(f"Initializing {self.__class__.__name__} with max_age={self.max_age}, "
                   f"min_hits={self.min_hits}, iou_threshold={self.iou_threshold}")
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the tracker.
        
        Returns:
            Name of the tracker
        """
        pass
    
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Update the tracker with new detections.
        
        Args:
            detections: List of new detections
        
        Returns:
            List of tracked detections
        """
        pass
    
    def set_camera_name(self, camera_name: str) -> None:
        """
        Set the camera name for metrics collection.
        
        Args:
            camera_name: Name of the camera
        """
        self.camera_name = camera_name
    
    def _record_metrics(self, start_time: float, tracks: List[Track]) -> None:
        """
        Record metrics for the tracking.
        
        Args:
            start_time: Start time of the tracking
            tracks: List of tracks
        """
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Record metrics
        metrics_collector.add_metric('tracking', 'processing_time', processing_time * 1000, self.camera_name)
        metrics_collector.add_metric('tracking', 'track_count', len(tracks), self.camera_name)
        
        # Record ID switches
        id_switches = 0  # This would need to be calculated based on ground truth
        metrics_collector.add_metric('tracking', 'id_switches', id_switches, self.camera_name)
        
        # Record MOTA and MOTP
        mota = 0.0  # This would need to be calculated based on ground truth
        motp = 0.0  # This would need to be calculated based on ground truth
        metrics_collector.add_metric('tracking', 'mota', mota, self.camera_name)
        metrics_collector.add_metric('tracking', 'motp', motp, self.camera_name)
        
        # Record mostly tracked and mostly lost
        mostly_tracked = 0  # This would need to be calculated based on ground truth
        mostly_lost = 0  # This would need to be calculated based on ground truth
        metrics_collector.add_metric('tracking', 'mostly_tracked', mostly_tracked, self.camera_name)
        metrics_collector.add_metric('tracking', 'mostly_lost', mostly_lost, self.camera_name)
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
        
        Returns:
            IoU value
        """
        # Calculate intersection area
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
