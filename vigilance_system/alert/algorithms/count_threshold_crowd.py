"""
Count threshold-based crowd detector implementation.

This module provides a simple count threshold-based crowd detector.
"""

import time
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import Detection
from vigilance_system.alert.algorithms.base_crowd_detector import BaseCrowdDetector, CrowdEvent

logger = get_logger(__name__)


class CountThresholdCrowdDetector(BaseCrowdDetector):
    """
    Count threshold-based crowd detector implementation.
    
    This class implements a simple crowd detector that triggers an alert
    when the number of people in an area exceeds a threshold.
    """
    
    def __init__(self):
        """Initialize the count threshold crowd detector."""
        super().__init__()
        
        # Get configuration
        self.threshold = config.get(f'{self.algorithm_config}.threshold', 5)
        self.min_distance = config.get(f'{self.algorithm_config}.min_distance', 100)
        
        # Initialize state
        self.active_event = None
        
        logger.info(f"Initialized count threshold crowd detector with "
                   f"threshold={self.threshold}, min_distance={self.min_distance}")
    
    def get_name(self) -> str:
        """
        Get the name of the crowd detector.
        
        Returns:
            Name of the crowd detector
        """
        return 'count_threshold'
    
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
        # Record start time for metrics
        start_time = time.time()
        
        # Increment frame count
        self.frame_count += 1
        
        # Filter person detections
        person_detections = [d for d in detections if d.class_name.lower() == 'person']
        
        # Count people
        person_count = len(person_detections)
        
        # Check if count exceeds threshold
        if person_count >= self.threshold:
            # Calculate crowd center and bounding box
            if person_detections:
                # Calculate center as average of all person centers
                centers = [(d.bbox[0] + d.bbox[2]) / 2 for d in person_detections]
                centers_y = [(d.bbox[1] + d.bbox[3]) / 2 for d in person_detections]
                center_x = sum(centers) / len(centers)
                center_y = sum(centers_y) / len(centers_y)
                center = (center_x, center_y)
                
                # Calculate bounding box that contains all people
                x1 = min(d.bbox[0] for d in person_detections)
                y1 = min(d.bbox[1] for d in person_detections)
                x2 = max(d.bbox[2] for d in person_detections)
                y2 = max(d.bbox[3] for d in person_detections)
                bbox = (x1, y1, x2, y2)
                
                # Calculate confidence based on how much the count exceeds the threshold
                confidence = min(1.0, (person_count - self.threshold + 1) / self.threshold)
                
                if self.active_event is None:
                    # Create new crowd event
                    event = CrowdEvent(
                        location=center,
                        bbox=bbox,
                        frame_id=frame_id,
                        count=person_count,
                        confidence=confidence
                    )
                    event.id = self.next_event_id
                    self.next_event_id += 1
                    
                    # Add to events list
                    self.crowd_events.append(event)
                    self.active_event = event
                else:
                    # Update existing event
                    self.active_event.update(center, bbox, frame_id, person_count, confidence)
            else:
                # No people detected (shouldn't happen since person_count >= threshold)
                if self.active_event is not None:
                    self.active_event.end()
                    self.active_event = None
        else:
            # Count below threshold
            if self.active_event is not None:
                self.active_event.end()
                self.active_event = None
        
        # Get active events
        active_events = [e for e in self.crowd_events if e.is_active]
        
        # Record metrics
        self._record_metrics(start_time, active_events)
        
        return active_events
