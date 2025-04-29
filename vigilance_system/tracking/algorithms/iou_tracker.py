"""
IoU tracker implementation.

This module provides a simple IoU-based tracker implementation.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import Detection
from vigilance_system.tracking.algorithms.base_tracker import BaseTracker, Track

logger = get_logger(__name__)


class IoUTracker(BaseTracker):
    """
    IoU tracker implementation.
    
    This class implements a simple IoU-based tracker.
    """
    
    def __init__(self):
        """Initialize the IoU tracker."""
        super().__init__()
    
    def get_name(self) -> str:
        """
        Get the name of the tracker.
        
        Returns:
            Name of the tracker
        """
        return 'iou'
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Update the tracker with new detections.
        
        Args:
            detections: List of new detections
        
        Returns:
            List of tracked detections
        """
        # Record start time for metrics
        start_time = time.time()
        
        # Increment frame count
        self.frame_count += 1
        
        # Predict new locations of tracks (just increment age and time_since_update)
        for track in self.tracks:
            track.predict()
        
        # Associate detections with tracks
        if len(self.tracks) > 0 and len(detections) > 0:
            # Calculate IoU between all tracks and detections
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, detection in enumerate(detections):
                    iou_matrix[i, j] = self._calculate_iou(track.detection.bbox, detection.bbox)
            
            # Use Hungarian algorithm to find best matches
            from scipy.optimize import linear_sum_assignment
            track_indices, detection_indices = linear_sum_assignment(-iou_matrix)
            
            # Update matched tracks
            for track_idx, det_idx in zip(track_indices, detection_indices):
                if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                    self.tracks[track_idx].update(detections[det_idx])
                else:
                    # IoU too low, treat as unmatched
                    pass
            
            # Get unmatched tracks and detections
            unmatched_tracks = [i for i in range(len(self.tracks)) if i not in track_indices]
            unmatched_detections = [i for i in range(len(detections)) if i not in detection_indices]
            
            # Add new tracks for unmatched detections
            for i in unmatched_detections:
                self.tracks.append(Track(detections[i], self.next_id))
                self.next_id += 1
        else:
            # No tracks or no detections
            if len(detections) > 0:
                # Create new tracks for all detections
                for detection in detections:
                    self.tracks.append(Track(detection, self.next_id))
                    self.next_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Get tracked detections
        tracked_detections = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update == 0:
                tracked_detection = track.to_detection()
                tracked_detections.append(tracked_detection)
        
        # Record metrics
        self._record_metrics(start_time, self.tracks)
        
        return tracked_detections
