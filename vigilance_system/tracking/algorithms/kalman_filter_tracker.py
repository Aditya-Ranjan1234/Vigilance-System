"""
Kalman Filter Tracker implementation.

This module provides a Kalman filter based tracker for tracking objects
across frames without using deep learning.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple

from vigilance_system.tracking.algorithms.base_tracker import BaseTracker
from vigilance_system.utils.logger import get_logger

logger = get_logger(__name__)


class KalmanFilterTracker(BaseTracker):
    """
    Kalman Filter based tracker.
    
    Tracks objects using Kalman filter to predict object positions
    and matches them with new detections.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Kalman filter tracker.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "kalman_filter"
        
        # Tracking parameters
        self.max_age = config.get('max_age', 10)  # Maximum frames to keep a track alive without matching
        self.min_hits = config.get('min_hits', 3)  # Minimum detection matches before track is confirmed
        self.iou_threshold = config.get('iou_threshold', 0.3)  # IoU threshold for matching
        
        # Tracking state
        self.tracks = {}  # Dictionary mapping object_id to track data
        self.next_id = 0
        
        logger.info(f"Initialized {self.name} tracker")
    
    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update the tracker with new detections.
        
        Args:
            frame: Current frame
            detections: List of detections from the detector
            
        Returns:
            List of tracked objects with keys:
                - object_id: Unique object ID
                - label: Class label
                - confidence: Detection confidence
                - bbox: Bounding box as (x1, y1, x2, y2)
                - centroid: Object centroid as (x, y)
        """
        # Predict new locations of existing tracks
        self._predict()
        
        # Match detections with existing tracks
        matched_tracks, unmatched_tracks, unmatched_detections = self._match_detections(detections)
        
        # Update matched tracks
        for track_id, detection_idx in matched_tracks:
            self._update_track(track_id, detections[detection_idx])
        
        # Handle unmatched tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id]['time_since_update'] += 1
            
            # Delete old tracks
            if self.tracks[track_id]['time_since_update'] > self.max_age:
                self.tracks[track_id]['active'] = False
        
        # Create new tracks for unmatched detections
        for idx in unmatched_detections:
            self._create_track(detections[idx])
        
        # Return active tracks
        results = []
        for obj_id, track in self.tracks.items():
            if track['active'] and track['hits'] >= self.min_hits:
                results.append({
                    'object_id': obj_id,
                    'label': track['label'],
                    'confidence': track['confidence'],
                    'bbox': track['bbox'],
                    'centroid': self._calculate_centroid(track['bbox'])
                })
        
        return results
    
    def _create_track(self, detection: Dict[str, Any]) -> int:
        """Create a new track from a detection."""
        obj_id = self.next_id
        self.next_id += 1
        
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Initialize Kalman filter
        kalman = cv2.KalmanFilter(8, 4)  # 8 state variables, 4 measurement variables
        
        # State: [x, y, w, h, vx, vy, vw, vh]
        # Measurement: [x, y, w, h]
        
        # Transition matrix (state update matrix)
        kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1]   # vh = vh
        ], dtype=np.float32)
        
        # Measurement matrix
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0, 0],  # w
            [0, 0, 0, 1, 0, 0, 0, 0]   # h
        ], dtype=np.float32)
        
        # Process noise covariance
        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Error covariance
        kalman.errorCovPost = np.eye(8, dtype=np.float32)
        
        # Initial state
        w = x2 - x1
        h = y2 - y1
        kalman.statePost = np.array([x1, y1, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(-1, 1)
        
        # Create track
        self.tracks[obj_id] = {
            'kalman': kalman,
            'bbox': bbox,
            'label': detection['label'],
            'confidence': detection['confidence'],
            'history': [self._calculate_centroid(bbox)],
            'hits': 1,
            'time_since_update': 0,
            'active': True,
            'age': 0
        }
        
        return obj_id
    
    def _predict(self) -> None:
        """Predict new locations of all tracks using Kalman filter."""
        for track_id, track in self.tracks.items():
            if not track['active']:
                continue
                
            # Predict next state
            kalman = track['kalman']
            prediction = kalman.predict()
            
            # Extract predicted bounding box
            x = prediction[0, 0]
            y = prediction[1, 0]
            w = prediction[2, 0]
            h = prediction[3, 0]
            
            # Update track with prediction
            track['bbox'] = (int(x), int(y), int(x + w), int(y + h))
            track['age'] += 1
    
    def _match_detections(self, detections: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections with existing tracks using IoU."""
        if not self.tracks or not detections:
            return [], list(self.tracks.keys()), list(range(len(detections)))
        
        # Get active tracks
        active_tracks = {k: v for k, v in self.tracks.items() if v['active']}
        if not active_tracks:
            return [], [], list(range(len(detections)))
        
        # Calculate IoU between tracks and detections
        iou_matrix = np.zeros((len(active_tracks), len(detections)))
        track_ids = list(active_tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(active_tracks[track_id]['bbox'], detection['bbox'])
        
        # Use Hungarian algorithm for optimal assignment
        matched_indices = []
        
        # Simple greedy matching for now
        # For each track, find the detection with highest IoU
        for i in range(len(track_ids)):
            if np.max(iou_matrix[i]) >= self.iou_threshold:
                j = np.argmax(iou_matrix[i])
                matched_indices.append((track_ids[i], j))
        
        # Get unmatched tracks and detections
        matched_track_ids = [t for t, _ in matched_indices]
        matched_detection_indices = [d for _, d in matched_indices]
        
        unmatched_tracks = [t for t in track_ids if t not in matched_track_ids]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detection_indices]
        
        return matched_indices, unmatched_tracks, unmatched_detections
    
    def _update_track(self, track_id: int, detection: Dict[str, Any]) -> None:
        """Update a track with a matched detection."""
        track = self.tracks[track_id]
        bbox = detection['bbox']
        
        # Update Kalman filter with measurement
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        measurement = np.array([x1, y1, w, h], dtype=np.float32).reshape(-1, 1)
        track['kalman'].correct(measurement)
        
        # Update track
        track['bbox'] = bbox
        track['label'] = detection['label']
        track['confidence'] = detection['confidence']
        track['hits'] += 1
        track['time_since_update'] = 0
        
        # Add centroid to history
        centroid = self._calculate_centroid(bbox)
        track['history'].append(centroid)
        
        # Limit history length
        if len(track['history']) > 30:
            track['history'] = track['history'][-30:]
    
    def _calculate_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Calculate the centroid of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                       bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
