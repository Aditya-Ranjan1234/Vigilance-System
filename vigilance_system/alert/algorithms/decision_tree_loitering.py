"""
Decision Tree Loitering Detection implementation.

This module provides a decision tree based approach to detect loitering
behavior without using deep learning.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import time
from collections import deque

from vigilance_system.alert.algorithms.base_loitering_detector import BaseLoiteringDetector
from vigilance_system.utils.logger import get_logger

logger = get_logger(__name__)


class DecisionTreeLoiteringDetector(BaseLoiteringDetector):
    """
    Decision Tree loitering detector.
    
    Detects loitering behavior using a rule-based decision tree approach
    based on various features extracted from object trajectories.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the decision tree loitering detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "decision_tree"
        
        # Parameters
        self.min_trajectory_length = config.get('min_trajectory_length', 20)  # Minimum trajectory length to analyze
        self.max_speed_threshold = config.get('max_speed_threshold', 10.0)  # Maximum speed to consider loitering
        self.min_direction_changes = config.get('min_direction_changes', 3)  # Minimum number of direction changes
        self.min_time_in_area = config.get('min_time_in_area', 8.0)  # Minimum time in seconds to consider loitering
        
        # State
        self.object_trajectories = {}  # Dictionary mapping object_id to trajectory (list of positions)
        self.object_start_times = {}  # Dictionary mapping object_id to first detection time
        self.object_speeds = {}  # Dictionary mapping object_id to list of speeds
        self.object_directions = {}  # Dictionary mapping object_id to list of directions
        
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
            
            # Initialize tracking for new objects
            if obj_id not in self.object_trajectories:
                self.object_trajectories[obj_id] = deque(maxlen=100)  # Limit trajectory length
                self.object_start_times[obj_id] = current_time
                self.object_speeds[obj_id] = deque(maxlen=50)  # Limit speed history
                self.object_directions[obj_id] = deque(maxlen=50)  # Limit direction history
            
            # Update trajectory
            trajectory = self.object_trajectories[obj_id]
            if trajectory and len(trajectory) > 0:
                # Calculate speed and direction
                prev_pos = trajectory[-1]
                speed = self._calculate_speed(prev_pos, position, 1.0)  # Assuming 1 second between frames
                direction = self._calculate_direction(prev_pos, position)
                
                # Update speed and direction history
                self.object_speeds[obj_id].append(speed)
                self.object_directions[obj_id].append(direction)
            
            # Add current position to trajectory
            trajectory.append(position)
            
            # Check if trajectory is long enough to analyze
            if len(trajectory) >= self.min_trajectory_length:
                # Calculate time spent
                time_spent = current_time - self.object_start_times[obj_id]
                
                # Apply decision tree to detect loitering
                if self._detect_loitering_decision_tree(obj_id, time_spent):
                    loitering_objects.append({
                        'object_id': obj_id,
                        'duration': time_spent,
                        'detection': obj
                    })
        
        # Clean up objects that are no longer tracked
        self._cleanup_objects(tracked_objects)
        
        return loitering_objects
    
    def _detect_loitering_decision_tree(self, obj_id: int, time_spent: float) -> bool:
        """
        Apply decision tree to detect loitering behavior.
        
        Args:
            obj_id: Object ID
            time_spent: Time spent tracking the object
            
        Returns:
            True if loitering is detected, False otherwise
        """
        # Check if object has been tracked long enough
        if time_spent < self.min_time_in_area:
            return False
        
        # Check average speed
        speeds = self.object_speeds[obj_id]
        if not speeds:
            return False
        
        avg_speed = sum(speeds) / len(speeds)
        if avg_speed > self.max_speed_threshold:
            return False
        
        # Check direction changes
        directions = self.object_directions[obj_id]
        if len(directions) < 2:
            return False
        
        direction_changes = 0
        prev_direction = directions[0]
        for direction in directions[1:]:
            # Check if direction changed significantly (more than 45 degrees)
            if abs(direction - prev_direction) > 45:
                direction_changes += 1
            prev_direction = direction
        
        if direction_changes < self.min_direction_changes:
            return False
        
        # Check spatial distribution
        trajectory = self.object_trajectories[obj_id]
        if not trajectory:
            return False
        
        # Calculate bounding box of trajectory
        x_coords = [p[0] for p in trajectory]
        y_coords = [p[1] for p in trajectory]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        area = width * height
        
        # If area is small relative to time spent, object is likely loitering
        area_time_ratio = area / time_spent
        
        return area_time_ratio < 1000  # Threshold determined empirically
    
    def _calculate_speed(self, pos1: Tuple[int, int], pos2: Tuple[int, int], time_delta: float) -> float:
        """Calculate speed between two positions."""
        distance = self._calculate_distance(pos1, pos2)
        return distance / time_delta if time_delta > 0 else 0
    
    def _calculate_direction(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate direction angle in degrees between two positions."""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        # Calculate angle in degrees (0-360)
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360
            
        return angle
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _cleanup_objects(self, tracked_objects: List[Dict[str, Any]]) -> None:
        """Remove objects that are no longer tracked."""
        current_ids = {obj['object_id'] for obj in tracked_objects}
        
        # Remove objects that are no longer tracked
        for obj_id in list(self.object_trajectories.keys()):
            if obj_id not in current_ids:
                self.object_trajectories.pop(obj_id, None)
                self.object_start_times.pop(obj_id, None)
                self.object_speeds.pop(obj_id, None)
                self.object_directions.pop(obj_id, None)
