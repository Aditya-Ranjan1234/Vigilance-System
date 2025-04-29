"""
Time threshold-based loitering detector implementation.

This module provides a simple time threshold-based loitering detector.
"""

import time
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import Detection
from vigilance_system.alert.algorithms.base_loitering_detector import BaseLoiteringDetector, LoiteringEvent

logger = get_logger(__name__)


class TimeThresholdLoiteringDetector(BaseLoiteringDetector):
    """
    Time threshold-based loitering detector implementation.
    
    This class implements a simple loitering detector that triggers an alert
    when a person stays in the same area for a certain amount of time.
    """
    
    def __init__(self):
        """Initialize the time threshold loitering detector."""
        super().__init__()
        
        # Get configuration
        self.person_loitering_time = config.get(f'{self.algorithm_config}.person_loitering_time', 30)
        self.distance_threshold = config.get(f'{self.algorithm_config}.distance_threshold', 50)
        
        # Initialize track history
        self.track_history = {}
        
        logger.info(f"Initialized time threshold loitering detector with "
                   f"person_loitering_time={self.person_loitering_time}, "
                   f"distance_threshold={self.distance_threshold}")
    
    def get_name(self) -> str:
        """
        Get the name of the loitering detector.
        
        Returns:
            Name of the loitering detector
        """
        return 'time_threshold'
    
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
        # Record start time for metrics
        start_time = time.time()
        
        # Increment frame count
        self.frame_count += 1
        
        # Filter person detections
        person_detections = [d for d in detections if d.class_name.lower() == 'person' and hasattr(d, 'tracking_id')]
        
        # Update track history
        current_time = time.time()
        
        for detection in person_detections:
            track_id = detection.tracking_id
            
            # Calculate center point
            center_x = (detection.bbox[0] + detection.bbox[2]) / 2
            center_y = (detection.bbox[1] + detection.bbox[3]) / 2
            center = (center_x, center_y)
            
            if track_id not in self.track_history:
                # New track
                self.track_history[track_id] = {
                    'positions': [center],
                    'timestamps': [current_time],
                    'start_time': current_time,
                    'last_update_time': current_time,
                    'loitering_start_time': None,
                    'loitering_position': None,
                    'event_id': None
                }
            else:
                # Existing track
                history = self.track_history[track_id]
                
                # Update history
                history['positions'].append(center)
                history['timestamps'].append(current_time)
                history['last_update_time'] = current_time
                
                # Limit history size
                if len(history['positions']) > 100:
                    history['positions'].pop(0)
                    history['timestamps'].pop(0)
                
                # Check if person is loitering
                if self._is_loitering(history, center):
                    # Person is loitering
                    if history['loitering_start_time'] is None:
                        # Start of loitering
                        history['loitering_start_time'] = current_time
                        history['loitering_position'] = center
                    
                    # Check if loitering duration exceeds threshold
                    loitering_duration = current_time - history['loitering_start_time']
                    
                    if loitering_duration >= self.person_loitering_time:
                        # Loitering detected
                        if history['event_id'] is None:
                            # Create new loitering event
                            event = LoiteringEvent(
                                track_id=track_id,
                                location=center,
                                bbox=detection.bbox,
                                frame_id=frame_id,
                                confidence=detection.confidence
                            )
                            event.id = self.next_event_id
                            self.next_event_id += 1
                            
                            # Add to events list
                            self.loitering_events.append(event)
                            
                            # Update history
                            history['event_id'] = event.id
                        else:
                            # Update existing event
                            event_id = history['event_id']
                            event = next((e for e in self.loitering_events if e.id == event_id), None)
                            
                            if event is not None and event.is_active:
                                event.update(center, detection.bbox, frame_id, detection.confidence)
                else:
                    # Person is not loitering
                    if history['loitering_start_time'] is not None:
                        # End of loitering
                        history['loitering_start_time'] = None
                        history['loitering_position'] = None
                        
                        # End event if exists
                        if history['event_id'] is not None:
                            event_id = history['event_id']
                            event = next((e for e in self.loitering_events if e.id == event_id), None)
                            
                            if event is not None and event.is_active:
                                event.end()
                            
                            history['event_id'] = None
        
        # Remove old tracks
        current_track_ids = [d.tracking_id for d in person_detections]
        tracks_to_remove = []
        
        for track_id, history in self.track_history.items():
            if track_id not in current_track_ids:
                # Track lost
                if current_time - history['last_update_time'] > 5:  # 5 seconds timeout
                    tracks_to_remove.append(track_id)
                    
                    # End event if exists
                    if history['event_id'] is not None:
                        event_id = history['event_id']
                        event = next((e for e in self.loitering_events if e.id == event_id), None)
                        
                        if event is not None and event.is_active:
                            event.end()
        
        # Remove tracks
        for track_id in tracks_to_remove:
            del self.track_history[track_id]
        
        # Get active events
        active_events = [e for e in self.loitering_events if e.is_active]
        
        # Record metrics
        self._record_metrics(start_time, active_events)
        
        return active_events
    
    def _is_loitering(self, history: Dict[str, Any], current_position: Tuple[float, float]) -> bool:
        """
        Check if a person is loitering.
        
        Args:
            history: Track history
            current_position: Current position (x, y)
        
        Returns:
            True if the person is loitering, False otherwise
        """
        # Check if we have enough history
        if len(history['positions']) < 10:
            return False
        
        # Calculate average position
        positions = history['positions']
        avg_x = sum(p[0] for p in positions) / len(positions)
        avg_y = sum(p[1] for p in positions) / len(positions)
        avg_position = (avg_x, avg_y)
        
        # Calculate distance from average position
        dx = current_position[0] - avg_position[0]
        dy = current_position[1] - avg_position[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        
        # Check if distance is below threshold
        return distance < self.distance_threshold
