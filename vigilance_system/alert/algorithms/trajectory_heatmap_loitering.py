"""
Trajectory heatmap-based loitering detector implementation.

This module provides a loitering detector based on trajectory heatmaps.
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


class TrajectoryHeatmapLoiteringDetector(BaseLoiteringDetector):
    """
    Trajectory heatmap-based loitering detector implementation.
    
    This class implements a loitering detector that uses trajectory heatmaps
    to detect when a person stays in the same area for a certain amount of time.
    """
    
    def __init__(self):
        """Initialize the trajectory heatmap loitering detector."""
        super().__init__()
        
        # Get configuration
        self.cell_size = config.get(f'{self.algorithm_config}.cell_size', 20)
        self.threshold = config.get(f'{self.algorithm_config}.threshold', 0.7)
        self.min_time = config.get(f'{self.algorithm_config}.min_time', 20)
        self.decay_factor = config.get(f'{self.algorithm_config}.decay_factor', 0.95)
        
        # Initialize heatmaps
        self.heatmaps = {}
        self.frame_size = None
        
        logger.info(f"Initialized trajectory heatmap loitering detector with "
                   f"cell_size={self.cell_size}, threshold={self.threshold}, "
                   f"min_time={self.min_time}, decay_factor={self.decay_factor}")
    
    def get_name(self) -> str:
        """
        Get the name of the loitering detector.
        
        Returns:
            Name of the loitering detector
        """
        return 'trajectory_heatmap'
    
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
        
        # Initialize frame size if not set
        if self.frame_size is None and frame is not None:
            self.frame_size = (frame.shape[1], frame.shape[0])
            logger.info(f"Initialized frame size: {self.frame_size}")
        
        # Filter person detections
        person_detections = [d for d in detections if d.class_name.lower() == 'person' and hasattr(d, 'tracking_id')]
        
        # Update heatmaps
        current_time = time.time()
        current_track_ids = []
        
        for detection in person_detections:
            track_id = detection.tracking_id
            current_track_ids.append(track_id)
            
            # Calculate center point
            center_x = (detection.bbox[0] + detection.bbox[2]) / 2
            center_y = (detection.bbox[1] + detection.bbox[3]) / 2
            center = (center_x, center_y)
            
            # Initialize heatmap if not exists
            if track_id not in self.heatmaps:
                self._initialize_heatmap(track_id)
            
            # Update heatmap
            self._update_heatmap(track_id, center, current_time)
            
            # Check for loitering
            is_loitering, loitering_position = self._check_loitering(track_id)
            
            if is_loitering:
                # Check if track has an active event
                event = next((e for e in self.loitering_events if e.track_id == track_id and e.is_active), None)
                
                if event is None:
                    # Create new loitering event
                    event = LoiteringEvent(
                        track_id=track_id,
                        location=loitering_position,
                        bbox=detection.bbox,
                        frame_id=frame_id,
                        confidence=detection.confidence
                    )
                    event.id = self.next_event_id
                    self.next_event_id += 1
                    
                    # Add to events list
                    self.loitering_events.append(event)
                else:
                    # Update existing event
                    event.update(loitering_position, detection.bbox, frame_id, detection.confidence)
            else:
                # End active event if exists
                event = next((e for e in self.loitering_events if e.track_id == track_id and e.is_active), None)
                if event is not None:
                    event.end()
        
        # Apply decay to all heatmaps
        for track_id in list(self.heatmaps.keys()):
            if track_id not in current_track_ids:
                # Track lost, apply stronger decay
                self._decay_heatmap(track_id, self.decay_factor * 0.5)
                
                # End active event if exists
                event = next((e for e in self.loitering_events if e.track_id == track_id and e.is_active), None)
                if event is not None:
                    event.end()
                
                # Remove heatmap if all values are very low
                if np.max(self.heatmaps[track_id]['heatmap']) < 0.01:
                    del self.heatmaps[track_id]
            else:
                # Track active, apply normal decay
                self._decay_heatmap(track_id, self.decay_factor)
        
        # Get active events
        active_events = [e for e in self.loitering_events if e.is_active]
        
        # Record metrics
        self._record_metrics(start_time, active_events)
        
        return active_events
    
    def _initialize_heatmap(self, track_id: int) -> None:
        """
        Initialize a heatmap for a track.
        
        Args:
            track_id: ID of the track
        """
        if self.frame_size is None:
            logger.warning("Frame size not initialized, using default (640, 480)")
            self.frame_size = (640, 480)
        
        # Calculate grid size
        grid_width = self.frame_size[0] // self.cell_size + 1
        grid_height = self.frame_size[1] // self.cell_size + 1
        
        # Initialize heatmap
        self.heatmaps[track_id] = {
            'heatmap': np.zeros((grid_height, grid_width), dtype=np.float32),
            'last_update_time': time.time(),
            'start_time': time.time(),
            'positions': []
        }
    
    def _update_heatmap(self, track_id: int, position: Tuple[float, float], current_time: float) -> None:
        """
        Update a heatmap with a new position.
        
        Args:
            track_id: ID of the track
            position: Position (x, y)
            current_time: Current time
        """
        if track_id not in self.heatmaps:
            self._initialize_heatmap(track_id)
        
        # Get heatmap
        heatmap_data = self.heatmaps[track_id]
        heatmap = heatmap_data['heatmap']
        
        # Calculate grid position
        grid_x = int(position[0] // self.cell_size)
        grid_y = int(position[1] // self.cell_size)
        
        # Ensure grid position is within bounds
        grid_x = max(0, min(grid_x, heatmap.shape[1] - 1))
        grid_y = max(0, min(grid_y, heatmap.shape[0] - 1))
        
        # Update heatmap
        heatmap[grid_y, grid_x] += 0.1
        
        # Apply Gaussian blur to spread the heat
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        
        # Normalize heatmap
        if np.max(heatmap) > 1.0:
            heatmap = heatmap / np.max(heatmap)
        
        # Update heatmap data
        heatmap_data['heatmap'] = heatmap
        heatmap_data['last_update_time'] = current_time
        heatmap_data['positions'].append(position)
        
        # Limit positions history
        if len(heatmap_data['positions']) > 100:
            heatmap_data['positions'].pop(0)
    
    def _decay_heatmap(self, track_id: int, decay_factor: float) -> None:
        """
        Apply decay to a heatmap.
        
        Args:
            track_id: ID of the track
            decay_factor: Decay factor (0-1)
        """
        if track_id not in self.heatmaps:
            return
        
        # Get heatmap
        heatmap_data = self.heatmaps[track_id]
        heatmap = heatmap_data['heatmap']
        
        # Apply decay
        heatmap = heatmap * decay_factor
        
        # Update heatmap data
        heatmap_data['heatmap'] = heatmap
    
    def _check_loitering(self, track_id: int) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """
        Check if a track is loitering.
        
        Args:
            track_id: ID of the track
        
        Returns:
            Tuple of (is_loitering, loitering_position)
        """
        if track_id not in self.heatmaps:
            return False, None
        
        # Get heatmap
        heatmap_data = self.heatmaps[track_id]
        heatmap = heatmap_data['heatmap']
        
        # Check if track has been active for minimum time
        track_duration = time.time() - heatmap_data['start_time']
        if track_duration < self.min_time:
            return False, None
        
        # Find maximum value in heatmap
        max_value = np.max(heatmap)
        
        # Check if maximum value exceeds threshold
        if max_value >= self.threshold:
            # Find position of maximum value
            max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            grid_y, grid_x = max_pos
            
            # Convert to image coordinates
            x = (grid_x + 0.5) * self.cell_size
            y = (grid_y + 0.5) * self.cell_size
            
            return True, (x, y)
        
        return False, None
    
    def draw_heatmap(self, frame: np.ndarray, track_id: Optional[int] = None) -> np.ndarray:
        """
        Draw a heatmap on a frame.
        
        Args:
            frame: Input frame
            track_id: ID of the track to draw (if None, draw all)
        
        Returns:
            Frame with heatmap drawn
        """
        result = frame.copy()
        
        if track_id is not None:
            # Draw specific heatmap
            if track_id in self.heatmaps:
                result = self._draw_single_heatmap(result, track_id)
        else:
            # Draw all heatmaps
            for track_id in self.heatmaps:
                result = self._draw_single_heatmap(result, track_id)
        
        return result
    
    def _draw_single_heatmap(self, frame: np.ndarray, track_id: int) -> np.ndarray:
        """
        Draw a single heatmap on a frame.
        
        Args:
            frame: Input frame
            track_id: ID of the track to draw
        
        Returns:
            Frame with heatmap drawn
        """
        if track_id not in self.heatmaps:
            return frame
        
        # Get heatmap
        heatmap_data = self.heatmaps[track_id]
        heatmap = heatmap_data['heatmap']
        
        # Create colored heatmap
        colored_heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        colored_heatmap = cv2.applyColorMap((colored_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original frame
        alpha = 0.3
        result = cv2.addWeighted(frame, 1 - alpha, colored_heatmap, alpha, 0)
        
        return result
