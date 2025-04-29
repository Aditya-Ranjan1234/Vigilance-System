"""
Timer Threshold Loitering Detection implementation.

This module provides a simple timer-based approach to detect loitering
behavior without using deep learning.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import time

from vigilance_system.alert.algorithms.base_loitering_detector import BaseLoiteringDetector
from vigilance_system.utils.logger import get_logger

logger = get_logger(__name__)


class TimerThresholdLoiteringDetector(BaseLoiteringDetector):
    """
    Timer Threshold loitering detector.
    
    Detects loitering behavior based on the time an object spends in a specific zone.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the timer threshold loitering detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "timer_threshold"
        
        # Parameters
        self.min_loitering_time = config.get('min_loitering_time', 10.0)  # Minimum time in seconds to consider loitering
        self.zone_size = config.get('zone_size', 100)  # Size of zones for spatial analysis
        
        # State
        self.object_zones = {}  # Dictionary mapping object_id to current zone
        self.object_zone_times = {}  # Dictionary mapping object_id to time spent in current zone
        self.object_last_update = {}  # Dictionary mapping object_id to last update time
        
        logger.info(f"Initialized {self.name} loitering detector")
    
    def detect(self, tracked_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect loitering behavior in tracked objects.
        
        Args:
            tracked_objects: List of tracked objects from the tracker
            
        Returns:
            List of objects exhibiting loitering behavior
        """
        current_time = time.time()
        loitering_objects = []
        
        # Process each tracked object
        for obj in tracked_objects:
            obj_id = obj['object_id']
            position = obj['centroid']
            current_zone = self._get_zone(position)
            
            # Initialize tracking for new objects
            if obj_id not in self.object_zones:
                self.object_zones[obj_id] = current_zone
                self.object_zone_times[obj_id] = 0.0
                self.object_last_update[obj_id] = current_time
                continue
            
            # Calculate time delta since last update
            time_delta = current_time - self.object_last_update[obj_id]
            self.object_last_update[obj_id] = current_time
            
            # Check if object has changed zones
            if current_zone != self.object_zones[obj_id]:
                # Reset time if zone changed
                self.object_zones[obj_id] = current_zone
                self.object_zone_times[obj_id] = 0.0
            else:
                # Increment time in current zone
                self.object_zone_times[obj_id] += time_delta
            
            # Check if object has been in the same zone for too long
            if self.object_zone_times[obj_id] >= self.min_loitering_time:
                loitering_objects.append({
                    'object_id': obj_id,
                    'duration': self.object_zone_times[obj_id],
                    'detection': obj
                })
        
        # Clean up objects that are no longer tracked
        self._cleanup_objects(tracked_objects)
        
        return loitering_objects
    
    def _get_zone(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """Convert a position to a zone identifier."""
        x, y = position
        return (x // self.zone_size, y // self.zone_size)
    
    def _cleanup_objects(self, tracked_objects: List[Dict[str, Any]]) -> None:
        """Remove objects that are no longer tracked."""
        current_ids = {obj['object_id'] for obj in tracked_objects}
        
        # Remove objects that are no longer tracked
        for obj_id in list(self.object_zones.keys()):
            if obj_id not in current_ids:
                self.object_zones.pop(obj_id, None)
                self.object_zone_times.pop(obj_id, None)
                self.object_last_update.pop(obj_id, None)
