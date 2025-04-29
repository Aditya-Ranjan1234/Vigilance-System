"""
Optical Flow tracker implementation.

This module provides an implementation of an optical flow-based tracker.
"""

import time
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import Detection
from vigilance_system.tracking.algorithms.base_tracker import BaseTracker, Track

logger = get_logger(__name__)


class OpticalFlowTrack(Track):
    """
    Track class for Optical Flow tracker.
    
    This class extends the base Track class with optical flow functionality.
    """
    
    def __init__(self, detection: Detection, track_id: int):
        """
        Initialize an Optical Flow track.
        
        Args:
            detection: Initial detection for this track
            track_id: Unique ID for this track
        """
        super().__init__(detection, track_id)
        
        # Initialize optical flow points
        self.prev_points = None
        self.prev_gray = None
        self.flow_points = None
    
    def update_flow_points(self, frame: np.ndarray) -> None:
        """
        Update optical flow points for this track.
        
        Args:
            frame: Current frame
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get bounding box
        x1, y1, x2, y2 = map(int, self.detection.bbox)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Skip if bounding box is invalid
        if x1 >= x2 or y1 >= y2:
            return
        
        # Extract ROI
        roi = gray[y1:y2, x1:x2]
        
        # Find good features to track
        points = cv2.goodFeaturesToTrack(roi, maxCorners=10, qualityLevel=0.01, minDistance=5)
        
        if points is not None:
            # Adjust points to global coordinates
            points[:, 0, 0] += x1
            points[:, 0, 1] += y1
            
            self.prev_points = points
            self.prev_gray = gray.copy()
    
    def predict_from_flow(self, frame: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Predict the next position of the track using optical flow.
        
        Args:
            frame: Current frame
        
        Returns:
            Predicted bounding box (x1, y1, x2, y2) or None if prediction failed
        """
        if self.prev_points is None or self.prev_gray is None:
            return None
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        if next_points is None:
            return None
        
        # Select good points
        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]
        
        if len(good_new) < 3:
            return None
        
        # Calculate average movement
        dx = np.mean(good_new[:, 0] - good_old[:, 0])
        dy = np.mean(good_new[:, 1] - good_old[:, 1])
        
        # Update bounding box
        x1, y1, x2, y2 = self.detection.bbox
        new_bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        
        # Update flow points
        self.prev_points = good_new.reshape(-1, 1, 2)
        self.prev_gray = gray.copy()
        
        return new_bbox


class OpticalFlowTracker(BaseTracker):
    """
    Optical Flow tracker implementation.
    
    This class implements an optical flow-based tracker.
    """
    
    def __init__(self):
        """Initialize the Optical Flow tracker."""
        super().__init__()
        
        # Get Optical Flow specific configuration
        self.max_corners = config.get('tracking.algorithms.optical_flow.max_corners', 200)
        self.quality_level = config.get('tracking.algorithms.optical_flow.quality_level', 0.01)
        self.min_distance = config.get('tracking.algorithms.optical_flow.min_distance', 30)
        
        # Initialize previous frame
        self.prev_frame = None
        
        logger.info(f"Initialized Optical Flow tracker with max_corners={self.max_corners}, "
                   f"quality_level={self.quality_level}, min_distance={self.min_distance}")
    
    def get_name(self) -> str:
        """
        Get the name of the tracker.
        
        Returns:
            Name of the tracker
        """
        return 'optical_flow'
    
    def update(self, detections: List[Detection], frame: np.ndarray = None) -> List[Detection]:
        """
        Update the tracker with new detections.
        
        Args:
            detections: List of new detections
            frame: Current frame (required for optical flow)
        
        Returns:
            List of tracked detections
        """
        # Record start time for metrics
        start_time = time.time()
        
        # Increment frame count
        self.frame_count += 1
        
        # Skip if frame is None
        if frame is None:
            logger.warning("Frame is None, skipping optical flow tracking")
            return detections
        
        # Initialize previous frame if needed
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            
            # Initialize tracks with detections
            for detection in detections:
                track = OpticalFlowTrack(detection, self.next_id)
                track.update_flow_points(frame)
                self.tracks.append(track)
                self.next_id += 1
            
            # Return detections as is
            return detections
        
        # Predict new locations of tracks using optical flow
        predicted_tracks = []
        for track in self.tracks:
            # Predict using optical flow
            if isinstance(track, OpticalFlowTrack):
                bbox = track.predict_from_flow(frame)
                if bbox is not None:
                    # Update detection with predicted bbox
                    track.detection.bbox = bbox
            
            # Standard prediction (increment age and time_since_update)
            track.predict()
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
                    
                    # Update optical flow points
                    if isinstance(self.tracks[track_idx], OpticalFlowTrack):
                        self.tracks[track_idx].update_flow_points(frame)
                else:
                    # IoU too low, treat as unmatched
                    pass
            
            # Get unmatched tracks and detections
            unmatched_tracks = [i for i in range(len(predicted_tracks)) if i not in track_indices]
            unmatched_detections = [i for i in range(len(detections)) if i not in detection_indices]
            
            # Add new tracks for unmatched detections
            for i in unmatched_detections:
                track = OpticalFlowTrack(detections[i], self.next_id)
                track.update_flow_points(frame)
                self.tracks.append(track)
                self.next_id += 1
        else:
            # No tracks or no detections
            if len(detections) > 0:
                # Create new tracks for all detections
                for detection in detections:
                    track = OpticalFlowTrack(detection, self.next_id)
                    track.update_flow_points(frame)
                    self.tracks.append(track)
                    self.next_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Get tracked detections
        tracked_detections = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update == 0:
                tracked_detection = track.to_detection()
                tracked_detections.append(tracked_detection)
        
        # Update previous frame
        self.prev_frame = frame.copy()
        
        # Record metrics
        self._record_metrics(start_time, self.tracks)
        
        return tracked_detections
