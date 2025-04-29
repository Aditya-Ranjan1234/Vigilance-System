"""
KLT (Kanade-Lucas-Tomasi) Tracker implementation.

This module provides a feature-based tracker using the KLT algorithm
for tracking objects across frames.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple

from vigilance_system.tracking.algorithms.base_tracker import BaseTracker
from vigilance_system.utils.logger import get_logger

logger = get_logger(__name__)


class KLTTracker(BaseTracker):
    """
    KLT (Kanade-Lucas-Tomasi) feature-based tracker.

    Tracks objects using the KLT algorithm by tracking feature points
    within each bounding box.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the KLT tracker.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "klt_tracker"

        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Tracking state
        self.prev_gray = None
        self.prev_points = {}  # Dictionary mapping object_id to feature points
        self.object_features = {}  # Dictionary mapping object_id to feature descriptors
        self.tracks = {}  # Dictionary mapping object_id to track history
        self.next_id = 0

        logger.info(f"Initialized {self.name} tracker")

    def update(self, detections: List[Any], frame: np.ndarray = None) -> List[Any]:
        """
        Update the tracker with new detections.

        Args:
            detections: List of detections from the detector
            frame: Current frame

        Returns:
            List of tracked objects with keys:
                - object_id: Unique object ID
                - label: Class label
                - confidence: Detection confidence
                - bbox: Bounding box as (x1, y1, x2, y2)
                - centroid: Object centroid as (x, y)
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize on first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            return self._initialize_tracks(detections)

        # Track existing objects using KLT
        tracked_objects = self._track_objects(gray)

        # Match new detections with existing tracks
        matched_tracks, unmatched_detections = self._match_detections(tracked_objects, detections)

        # Update matched tracks with new detections
        for track_id, detection_idx in matched_tracks:
            self._update_track(track_id, detections[detection_idx], gray)

        # Create new tracks for unmatched detections
        for idx in unmatched_detections:
            self._create_track(detections[idx], gray)

        # Update previous frame
        self.prev_gray = gray

        # Convert tracks to output format
        results = []
        for obj_id, track in self.tracks.items():
            if track['active']:
                results.append({
                    'object_id': obj_id,
                    'label': track['label'],
                    'confidence': track['confidence'],
                    'bbox': track['bbox'],
                    'centroid': self._calculate_centroid(track['bbox'])
                })

        return results

    def _initialize_tracks(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Initialize tracks from detections on the first frame."""
        results = []
        for detection in detections:
            obj_id = self._create_track(detection, self.prev_gray)
            bbox = detection['bbox']
            results.append({
                'object_id': obj_id,
                'label': detection['label'],
                'confidence': detection['confidence'],
                'bbox': bbox,
                'centroid': self._calculate_centroid(bbox)
            })
        return results

    def _create_track(self, detection: Dict[str, Any], gray: np.ndarray) -> int:
        """Create a new track from a detection."""
        obj_id = self.next_id
        self.next_id += 1

        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox

        # Extract region of interest
        roi = gray[y1:y2, x1:x2]

        # Find good features to track in the ROI
        if roi.size > 0:
            # Find corners in the ROI
            points = cv2.goodFeaturesToTrack(
                roi,
                mask=None,
                **self.feature_params
            )

            if points is not None:
                # Adjust points to global coordinates
                points = points + np.array([x1, y1], dtype=np.float32)

                # Store points for this object
                self.prev_points[obj_id] = points

                # Create track
                self.tracks[obj_id] = {
                    'bbox': bbox,
                    'label': detection['label'],
                    'confidence': detection['confidence'],
                    'history': [self._calculate_centroid(bbox)],
                    'age': 0,
                    'active': True,
                    'frames_since_detection': 0
                }

        return obj_id

    def _track_objects(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Track existing objects using KLT optical flow."""
        tracked_objects = []

        # For each tracked object
        for obj_id, points in list(self.prev_points.items()):
            if points is None or len(points) == 0:
                continue

            # Calculate optical flow
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                points,
                None,
                **self.lk_params
            )

            # Filter out points that couldn't be tracked
            good_new = new_points[status == 1]
            good_old = points[status == 1]

            if len(good_new) > 0:
                # Calculate new bounding box based on point movement
                old_bbox = self.tracks[obj_id]['bbox']
                x1, y1, x2, y2 = old_bbox

                # Calculate the average movement of points
                dx = np.mean(good_new[:, 0] - good_old[:, 0])
                dy = np.mean(good_new[:, 1] - good_old[:, 1])

                # Update bounding box
                new_bbox = (
                    int(x1 + dx),
                    int(y1 + dy),
                    int(x2 + dx),
                    int(y2 + dy)
                )

                # Update track
                self.tracks[obj_id]['bbox'] = new_bbox
                self.tracks[obj_id]['age'] += 1
                self.tracks[obj_id]['frames_since_detection'] += 1

                # Add to tracked objects
                tracked_objects.append({
                    'object_id': obj_id,
                    'bbox': new_bbox,
                    'points': good_new
                })

                # Update points
                self.prev_points[obj_id] = good_new
            else:
                # Not enough points to track
                self.tracks[obj_id]['active'] = False
                del self.prev_points[obj_id]

        return tracked_objects

    def _match_detections(self, tracked_objects: List[Dict[str, Any]],
                          detections: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Match tracked objects with new detections."""
        if not tracked_objects or not detections:
            return [], list(range(len(detections)))

        # Calculate IoU between tracked objects and detections
        iou_matrix = np.zeros((len(tracked_objects), len(detections)))
        for i, tracked in enumerate(tracked_objects):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(tracked['bbox'], detection['bbox'])

        # Match based on IoU
        matched_indices = []
        unmatched_detections = list(range(len(detections)))

        # Find matches with IoU > threshold
        iou_threshold = 0.3
        for i in range(len(tracked_objects)):
            # Find the detection with highest IoU
            if np.max(iou_matrix[i]) > iou_threshold:
                j = np.argmax(iou_matrix[i])
                matched_indices.append((tracked_objects[i]['object_id'], j))
                if j in unmatched_detections:
                    unmatched_detections.remove(j)

        return matched_indices, unmatched_detections

    def _update_track(self, track_id: int, detection: Dict[str, Any], gray: np.ndarray) -> None:
        """Update an existing track with a new detection."""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox

        # Extract region of interest
        roi = gray[y1:y2, x1:x2]

        # Find good features to track in the ROI
        if roi.size > 0:
            # Find corners in the ROI
            points = cv2.goodFeaturesToTrack(
                roi,
                mask=None,
                **self.feature_params
            )

            if points is not None:
                # Adjust points to global coordinates
                points = points + np.array([x1, y1], dtype=np.float32)

                # Store points for this object
                self.prev_points[track_id] = points

        # Update track
        self.tracks[track_id]['bbox'] = bbox
        self.tracks[track_id]['label'] = detection['label']
        self.tracks[track_id]['confidence'] = detection['confidence']
        self.tracks[track_id]['frames_since_detection'] = 0

        # Add centroid to history
        centroid = self._calculate_centroid(bbox)
        self.tracks[track_id]['history'].append(centroid)

        # Limit history length
        if len(self.tracks[track_id]['history']) > 30:
            self.tracks[track_id]['history'] = self.tracks[track_id]['history'][-30:]

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
