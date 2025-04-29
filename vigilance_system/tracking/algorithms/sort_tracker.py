"""
SORT (Simple Online and Realtime Tracking) implementation.

This module provides an implementation of the SORT tracking algorithm.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import Detection
from vigilance_system.tracking.algorithms.base_tracker import BaseTracker, Track

logger = get_logger(__name__)


class KalmanTrack(Track):
    """
    Track class with Kalman filter for SORT tracker.
    
    This class extends the base Track class with Kalman filter functionality.
    """
    
    def __init__(self, detection: Detection, track_id: int):
        """
        Initialize a Kalman track.
        
        Args:
            detection: Initial detection for this track
            track_id: Unique ID for this track
        """
        super().__init__(detection, track_id)
        
        # Initialize Kalman filter
        self.kf = self._init_kalman_filter(detection)
    
    def _init_kalman_filter(self, detection: Detection) -> Any:
        """
        Initialize a Kalman filter for this track.
        
        Args:
            detection: Initial detection for this track
        
        Returns:
            Initialized Kalman filter
        """
        import cv2
        
        # Get bounding box
        bbox = detection.bbox
        
        # Convert to center, width, height format
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Initialize Kalman filter
        kf = cv2.KalmanFilter(8, 4)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        
        kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        # Initialize state
        kf.statePost = np.array([[cx], [cy], [width], [height], [0], [0], [0], [0]], np.float32)
        
        return kf
    
    def predict(self) -> Tuple[float, float, float, float]:
        """
        Predict the next position of the track using the Kalman filter.
        
        Returns:
            Predicted bounding box (x1, y1, x2, y2)
        """
        # Predict next state
        state = self.kf.predict()
        
        # Extract center, width, height
        cx = state[0, 0]
        cy = state[1, 0]
        width = state[2, 0]
        height = state[3, 0]
        
        # Convert to bounding box format
        x1 = cx - width / 2
        y1 = cy - height / 2
        x2 = cx + width / 2
        y2 = cy + height / 2
        
        # Update track
        super().predict()
        
        return (x1, y1, x2, y2)
    
    def update(self, detection: Detection) -> None:
        """
        Update the track with a new detection.
        
        Args:
            detection: New detection to update the track with
        """
        # Get bounding box
        bbox = detection.bbox
        
        # Convert to center, width, height format
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Update Kalman filter
        measurement = np.array([[cx], [cy], [width], [height]], np.float32)
        self.kf.correct(measurement)
        
        # Update track
        super().update(detection)


class SORTTracker(BaseTracker):
    """
    SORT (Simple Online and Realtime Tracking) implementation.
    
    This class implements the SORT tracking algorithm.
    """
    
    def __init__(self):
        """Initialize the SORT tracker."""
        super().__init__()
    
    def get_name(self) -> str:
        """
        Get the name of the tracker.
        
        Returns:
            Name of the tracker
        """
        return 'sort'
    
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
        
        # Predict new locations of tracks
        predicted_tracks = []
        for track in self.tracks:
            bbox = track.predict()
            predicted_tracks.append(track)
        
        # Associate detections with tracks
        if len(predicted_tracks) > 0 and len(detections) > 0:
            # Calculate IoU between all tracks and detections
            iou_matrix = np.zeros((len(predicted_tracks), len(detections)))
            for i, track in enumerate(predicted_tracks):
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
            unmatched_tracks = [i for i in range(len(predicted_tracks)) if i not in track_indices]
            unmatched_detections = [i for i in range(len(detections)) if i not in detection_indices]
            
            # Add new tracks for unmatched detections
            for i in unmatched_detections:
                self.tracks.append(KalmanTrack(detections[i], self.next_id))
                self.next_id += 1
        else:
            # No tracks or no detections
            if len(detections) > 0:
                # Create new tracks for all detections
                for detection in detections:
                    self.tracks.append(KalmanTrack(detection, self.next_id))
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
