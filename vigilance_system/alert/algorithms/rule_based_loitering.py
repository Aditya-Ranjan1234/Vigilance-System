"""
Rule-based Loitering Detection implementation.

This module provides a simple rule-based approach to detect loitering
behavior without using deep learning.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import time

from vigilance_system.alert.algorithms.base_loitering_detector import BaseLoiteringDetector
from vigilance_system.utils.logger import get_logger

logger = get_logger(__name__)


class RuleBasedLoiteringDetector(BaseLoiteringDetector):
    """
    Rule-based loitering detector.
    
    Detects loitering behavior based on simple rules like time spent in an area
    and movement patterns.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the rule-based loitering detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "rule_based"
        
        # Parameters
        self.min_loitering_time = config.get('min_loitering_time', 10.0)  # Minimum time in seconds to consider loitering
        self.max_movement_distance = config.get('max_movement_distance', 50)  # Maximum distance to move while still considered loitering
        self.grid_size = config.get('grid_size', 50)  # Size of grid cells for spatial analysis
        
        # State
        self.object_start_times = {}  # Dictionary mapping object_id to first detection time
        self.object_positions = {}  # Dictionary mapping object_id to list of positions
        self.object_grids = {}  # Dictionary mapping object_id to set of visited grid cells
        
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
            if obj_id not in self.object_start_times:
                self.object_start_times[obj_id] = current_time
                self.object_positions[obj_id] = [position]
                self.object_grids[obj_id] = {self._get_grid_cell(position)}
                continue
            
            # Update position history
            self.object_positions[obj_id].append(position)
            self.object_grids[obj_id].add(self._get_grid_cell(position))
            
            # Limit history length
            if len(self.object_positions[obj_id]) > 100:
                self.object_positions[obj_id] = self.object_positions[obj_id][-100:]
            
            # Calculate time spent
            time_spent = current_time - self.object_start_times[obj_id]
            
            # Check loitering conditions
            if time_spent >= self.min_loitering_time:
                # Check if object has stayed within a limited area
                if self._check_limited_movement(obj_id):
                    # Check if object has revisited the same areas
                    if self._check_area_revisits(obj_id):
                        loitering_objects.append({
                            'object_id': obj_id,
                            'duration': time_spent,
                            'detection': obj
                        })
        
        # Clean up objects that are no longer tracked
        self._cleanup_objects(tracked_objects)
        
        return loitering_objects
    
    def _check_limited_movement(self, obj_id: int) -> bool:
        """Check if an object has limited movement (staying in a small area)."""
        positions = self.object_positions[obj_id]
        if len(positions) < 10:
            return False
        
        # Calculate maximum distance between any two positions
        max_distance = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = self._calculate_distance(positions[i], positions[j])
                max_distance = max(max_distance, distance)
        
        return max_distance <= self.max_movement_distance
    
    def _check_area_revisits(self, obj_id: int) -> bool:
        """Check if an object has revisited the same areas."""
        # If the number of unique grid cells is small compared to the number of positions,
        # it means the object has revisited the same areas
        positions = self.object_positions[obj_id]
        grid_cells = self.object_grids[obj_id]
        
        if len(positions) < 20:
            return False
        
        # Calculate ratio of unique grid cells to positions
        ratio = len(grid_cells) / len(positions)
        
        # If ratio is small, object has revisited the same areas
        return ratio < 0.5
    
    def _get_grid_cell(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """Convert a position to a grid cell."""
        x, y = position
        return (x // self.grid_size, y // self.grid_size)
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _cleanup_objects(self, tracked_objects: List[Dict[str, Any]]) -> None:
        """Remove objects that are no longer tracked."""
        current_ids = {obj['object_id'] for obj in tracked_objects}
        
        # Remove objects that are no longer tracked
        for obj_id in list(self.object_start_times.keys()):
            if obj_id not in current_ids:
                self.object_start_times.pop(obj_id, None)
                self.object_positions.pop(obj_id, None)
                self.object_grids.pop(obj_id, None)
