"""
Tracking algorithms for object detection.

This module provides implementations of various tracking algorithms
for object detection, including IoU, Kalman filter, and YOLOv8 tracking.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque
import time

from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.object_detector import Detection
from vigilance_system.detection.ml_algorithms import MLTracker

# Initialize logger
logger = get_logger(__name__)


class IoUTracker(MLTracker):
    """IoU-based object tracker."""

    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0, iou_threshold: float = 0.3):
        """
        Initialize the IoU tracker.

        Args:
            max_disappeared: Maximum number of frames an object can disappear before being removed
            max_distance: Maximum distance between detections to be considered the same object
            iou_threshold: Minimum IoU to consider two bounding boxes as the same object
        """
        super().__init__(max_disappeared, max_distance)
        self.iou_threshold = iou_threshold
        logger.info(f"Initialized IoU tracker with iou_threshold={iou_threshold}")

    def update(self, detections: List[Detection]) -> Dict[int, Dict[str, Any]]:
        """
        Update object tracking with new detections using IoU.

        Args:
            detections: List of new detections

        Returns:
            Dict[int, Dict[str, Any]]: Dictionary mapping object IDs to their current info
        """
        # If no objects are being tracked, register all detections
        if len(self.objects) == 0:
            for detection in detections:
                self._register(detection)
            return self.tracked_objects

        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # Remove object if it has disappeared for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            # Update durations for remaining objects
            for object_id in self.tracked_objects:
                if object_id in self.object_history:
                    self.tracked_objects[object_id]['duration'] = self.get_object_duration(object_id)

            return self.tracked_objects

        # Get bounding boxes of current objects
        object_ids = list(self.objects.keys())
        object_bboxes = []

        for obj_id in object_ids:
            if obj_id in self.object_history and self.object_history[obj_id]:
                last_detection = self.object_history[obj_id][-1]
                object_bboxes.append(last_detection.bbox)
            else:
                # If no history, use a dummy bbox
                object_bboxes.append([0, 0, 1, 1])

        object_bboxes = np.array(object_bboxes)

        # Get bounding boxes of new detections
        detection_bboxes = np.array([d.bbox for d in detections])

        # Calculate IoU between all pairs of bounding boxes
        ious = self._calculate_iou_matrix(object_bboxes, detection_bboxes)

        # Match detections to objects based on IoU
        matched_objects = set()
        matched_detections = set()

        # Sort IoUs in descending order and match greedily
        for obj_idx, det_idx in self._get_matches(ious):
            obj_id = object_ids[obj_idx]
            iou_value = ious[obj_idx, det_idx]

            # Only match if IoU is above threshold
            if iou_value >= self.iou_threshold:
                # Update object with new detection
                self.objects[obj_id] = (detections[det_idx].center_x, detections[det_idx].center_y)
                self.object_history[obj_id].append(detections[det_idx])
                self.disappeared[obj_id] = 0
                self.trajectories[obj_id].append((detections[det_idx].center_x, detections[det_idx].center_y))

                # Update tracked object info
                self.tracked_objects[obj_id] = {
                    'detection': detections[det_idx],
                    'class_id': detections[det_idx].class_id,
                    'class_name': detections[det_idx].class_name,
                    'bbox': detections[det_idx].bbox,
                    'center': (detections[det_idx].center_x, detections[det_idx].center_y),
                    'confidence': detections[det_idx].confidence,
                    'duration': self.get_object_duration(obj_id),
                    'trajectory': [(int(x), int(y)) for x, y in self.trajectories[obj_id]],
                    'iou': iou_value
                }

                matched_objects.add(obj_id)
                matched_detections.add(det_idx)

        # Check for disappeared objects
        for obj_id in set(self.objects.keys()) - matched_objects:
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self._deregister(obj_id)
            else:
                # Update duration for objects that are still tracked
                if obj_id in self.tracked_objects:
                    self.tracked_objects[obj_id]['duration'] = self.get_object_duration(obj_id)

        # Register new detections
        for i in range(len(detections)):
            if i not in matched_detections:
                self._register(detections[i])

        return self.tracked_objects

    def _calculate_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Calculate IoU between all pairs of bounding boxes.

        Args:
            boxes1: Array of bounding boxes in format [x, y, w, h]
            boxes2: Array of bounding boxes in format [x, y, w, h]

        Returns:
            np.ndarray: Matrix of IoU values
        """
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        boxes1_xyxy = np.zeros_like(boxes1)
        boxes1_xyxy[:, 0] = boxes1[:, 0]
        boxes1_xyxy[:, 1] = boxes1[:, 1]
        boxes1_xyxy[:, 2] = boxes1[:, 0] + boxes1[:, 2]
        boxes1_xyxy[:, 3] = boxes1[:, 1] + boxes1[:, 3]

        boxes2_xyxy = np.zeros_like(boxes2)
        boxes2_xyxy[:, 0] = boxes2[:, 0]
        boxes2_xyxy[:, 1] = boxes2[:, 1]
        boxes2_xyxy[:, 2] = boxes2[:, 0] + boxes2[:, 2]
        boxes2_xyxy[:, 3] = boxes2[:, 1] + boxes2[:, 3]

        # Calculate IoU
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))

        for i in range(len(boxes1)):
            for j in range(len(boxes2)):
                iou_matrix[i, j] = self._calculate_iou(boxes1_xyxy[i], boxes2_xyxy[j])

        return iou_matrix

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate IoU between two bounding boxes.

        Args:
            box1: Bounding box in format [x1, y1, x2, y2]
            box2: Bounding box in format [x1, y1, x2, y2]

        Returns:
            float: IoU value
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection_area = (x2 - x1) * (y2 - y1)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0

        return iou

    def _get_matches(self, iou_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Get matches between objects and detections based on IoU.

        Args:
            iou_matrix: Matrix of IoU values

        Returns:
            List[Tuple[int, int]]: List of (object_index, detection_index) pairs
        """
        # Flatten the matrix and get indices in descending order of IoU
        flat_indices = np.argsort(iou_matrix.flatten())[::-1]

        # Convert flat indices to 2D indices
        matches = []
        matched_rows = set()
        matched_cols = set()

        for flat_idx in flat_indices:
            row = flat_idx // iou_matrix.shape[1]
            col = flat_idx % iou_matrix.shape[1]

            # Skip if already matched or IoU is zero
            if row in matched_rows or col in matched_cols or iou_matrix[row, col] == 0:
                continue

            matches.append((row, col))
            matched_rows.add(row)
            matched_cols.add(col)

        return matches


class KalmanTracker(MLTracker):
    """Kalman filter-based object tracker."""

    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0, process_noise: float = 1e-2, measurement_noise: float = 1e-1):
        """
        Initialize the Kalman filter tracker.

        Args:
            max_disappeared: Maximum number of frames an object can disappear before being removed
            max_distance: Maximum distance between detections to be considered the same object
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        super().__init__(max_disappeared, max_distance)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.kalman_filters = {}  # Map from object ID to Kalman filter

        logger.info(f"Initialized Kalman tracker with process_noise={process_noise}, measurement_noise={measurement_noise}")

    def update(self, detections: List[Detection]) -> Dict[int, Dict[str, Any]]:
        """
        Update object tracking with new detections using Kalman filter.

        Args:
            detections: List of new detections

        Returns:
            Dict[int, Dict[str, Any]]: Dictionary mapping object IDs to their current info
        """
        # If no objects are being tracked, register all detections
        if len(self.objects) == 0:
            for detection in detections:
                self._register(detection)
                # Initialize Kalman filter for the new object
                self._init_kalman_filter(self.next_object_id - 1, detection)
            return self.tracked_objects

        # If no detections, predict new states for all objects and mark them as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                # Predict new state
                if object_id in self.kalman_filters:
                    self._predict_kalman(object_id)

                self.disappeared[object_id] += 1

                # Remove object if it has disappeared for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            # Update durations for remaining objects
            for object_id in self.tracked_objects:
                if object_id in self.object_history:
                    self.tracked_objects[object_id]['duration'] = self.get_object_duration(object_id)

            return self.tracked_objects

        # Predict new states for all objects
        for object_id in list(self.objects.keys()):
            if object_id in self.kalman_filters:
                self._predict_kalman(object_id)

        # Calculate distances between predicted positions and detections
        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[obj_id] for obj_id in object_ids])
        detection_centroids = np.array([(d.center_x, d.center_y) for d in detections])

        # Calculate distance matrix
        distances = np.zeros((len(object_ids), len(detections)))
        for i, obj_centroid in enumerate(object_centroids):
            for j, det_centroid in enumerate(detection_centroids):
                distances[i, j] = np.linalg.norm(obj_centroid - det_centroid)

        # Match detections to objects based on distance
        matched_objects = set()
        matched_detections = set()

        # Sort distances and match greedily
        for obj_idx, det_idx in self._get_matches(distances):
            obj_id = object_ids[obj_idx]
            distance = distances[obj_idx, det_idx]

            # Only match if distance is below threshold
            if distance <= self.max_distance:
                # Update Kalman filter with new measurement
                if obj_id in self.kalman_filters:
                    self._update_kalman(obj_id, detections[det_idx])

                # Update object with new detection
                self.objects[obj_id] = (detections[det_idx].center_x, detections[det_idx].center_y)
                self.object_history[obj_id].append(detections[det_idx])
                self.disappeared[obj_id] = 0
                self.trajectories[obj_id].append((detections[det_idx].center_x, detections[det_idx].center_y))

                # Update tracked object info
                self.tracked_objects[obj_id] = {
                    'detection': detections[det_idx],
                    'class_id': detections[det_idx].class_id,
                    'class_name': detections[det_idx].class_name,
                    'bbox': detections[det_idx].bbox,
                    'center': (detections[det_idx].center_x, detections[det_idx].center_y),
                    'confidence': detections[det_idx].confidence,
                    'duration': self.get_object_duration(obj_id),
                    'trajectory': [(int(x), int(y)) for x, y in self.trajectories[obj_id]],
                    'kalman_state': self.kalman_filters[obj_id][0].flatten().tolist() if obj_id in self.kalman_filters else None
                }

                matched_objects.add(obj_id)
                matched_detections.add(det_idx)

        # Check for disappeared objects
        for obj_id in set(self.objects.keys()) - matched_objects:
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self._deregister(object_id)
                # Remove Kalman filter
                if obj_id in self.kalman_filters:
                    del self.kalman_filters[obj_id]
            else:
                # Update duration for objects that are still tracked
                if obj_id in self.tracked_objects:
                    self.tracked_objects[obj_id]['duration'] = self.get_object_duration(obj_id)

        # Register new detections
        for i in range(len(detections)):
            if i not in matched_detections:
                self._register(detections[i])
                # Initialize Kalman filter for the new object
                self._init_kalman_filter(self.next_object_id - 1, detections[i])

        return self.tracked_objects

    def _init_kalman_filter(self, object_id: int, detection: Detection) -> None:
        """
        Initialize a Kalman filter for an object.

        Args:
            object_id: ID of the object
            detection: Detection to initialize the filter with
        """
        # State: [x, y, vx, vy, w, h]
        # x, y: center coordinates
        # vx, vy: velocity
        # w, h: width and height

        # Initialize state
        x = np.array([
            [detection.center_x],  # x
            [detection.center_y],  # y
            [0],                   # vx
            [0],                   # vy
            [detection.bbox[2]],   # w
            [detection.bbox[3]]    # h
        ])

        # Initialize state covariance
        P = np.eye(6) * 10

        # State transition matrix
        # x_k = x_{k-1} + vx_{k-1}
        # y_k = y_{k-1} + vy_{k-1}
        # vx_k = vx_{k-1}
        # vy_k = vy_{k-1}
        # w_k = w_{k-1}
        # h_k = h_{k-1}
        F = np.array([
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix
        # We can only measure x, y, w, h
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Process noise covariance
        Q = np.eye(6) * self.process_noise

        # Measurement noise covariance
        R = np.eye(4) * self.measurement_noise

        # Store Kalman filter parameters
        self.kalman_filters[object_id] = (x, P, F, H, Q, R)

    def _predict_kalman(self, object_id: int) -> None:
        """
        Predict the next state of an object using Kalman filter.

        Args:
            object_id: ID of the object
        """
        if object_id not in self.kalman_filters:
            return

        # Get Kalman filter parameters
        x, P, F, H, Q, R = self.kalman_filters[object_id]

        # Predict next state
        x = F @ x
        P = F @ P @ F.T + Q

        # Update Kalman filter parameters
        self.kalman_filters[object_id] = (x, P, F, H, Q, R)

        # Update object position
        self.objects[object_id] = (float(x[0, 0]), float(x[1, 0]))

    def _update_kalman(self, object_id: int, detection: Detection) -> None:
        """
        Update Kalman filter with a new measurement.

        Args:
            object_id: ID of the object
            detection: New detection
        """
        if object_id not in self.kalman_filters:
            return

        # Get Kalman filter parameters
        x, P, F, H, Q, R = self.kalman_filters[object_id]

        # Measurement
        z = np.array([
            [detection.center_x],
            [detection.center_y],
            [detection.bbox[2]],
            [detection.bbox[3]]
        ])

        # Calculate Kalman gain
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

        # Update state
        x = x + K @ (z - H @ x)
        P = (np.eye(6) - K @ H) @ P

        # Update Kalman filter parameters
        self.kalman_filters[object_id] = (x, P, F, H, Q, R)

    def _get_matches(self, distance_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Get matches between objects and detections based on distance.

        Args:
            distance_matrix: Matrix of distances

        Returns:
            List[Tuple[int, int]]: List of (object_index, detection_index) pairs
        """
        # Flatten the matrix and get indices in ascending order of distance
        flat_indices = np.argsort(distance_matrix.flatten())

        # Convert flat indices to 2D indices
        matches = []
        matched_rows = set()
        matched_cols = set()

        for flat_idx in flat_indices:
            row = flat_idx // distance_matrix.shape[1]
            col = flat_idx % distance_matrix.shape[1]

            # Skip if already matched
            if row in matched_rows or col in matched_cols:
                continue

            matches.append((row, col))
            matched_rows.add(row)
            matched_cols.add(col)

        return matches


class YOLOv8Tracker(MLTracker):
    """YOLOv8-based object tracker with advanced features."""

    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0,
                 iou_threshold: float = 0.3, confidence_threshold: float = 0.5):
        """
        Initialize the YOLOv8 tracker.

        Args:
            max_disappeared: Maximum number of frames an object can disappear before being removed
            max_distance: Maximum distance between detections to be considered the same object
            iou_threshold: Minimum IoU to consider two bounding boxes as the same object
            confidence_threshold: Minimum confidence score for detections
        """
        super().__init__(max_disappeared, max_distance)
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.last_update_time = time.time()
        self.frame_count = 0

        # Motion prediction parameters
        self.velocity_history = {}  # Store velocity history for each object
        self.velocity_window = 5    # Number of frames to average velocity

        logger.info(f"Initialized YOLOv8 tracker with iou_threshold={iou_threshold}, confidence_threshold={confidence_threshold}")

    def update(self, detections: List[Detection]) -> Dict[int, Dict[str, Any]]:
        """
        Update object tracking with new detections using YOLOv8 tracking algorithm.

        This combines IoU tracking with motion prediction and confidence filtering.

        Args:
            detections: List of new detections

        Returns:
            Dict[int, Dict[str, Any]]: Dictionary mapping object IDs to their current info
        """
        # Increment frame counter
        self.frame_count += 1

        # Filter detections by confidence
        filtered_detections = [d for d in detections if d.confidence >= self.confidence_threshold]

        # Calculate time delta for velocity calculations
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # If no objects are being tracked, register all detections
        if len(self.objects) == 0:
            for detection in filtered_detections:
                self._register(detection)
                # Initialize velocity history
                self.velocity_history[self.next_object_id - 1] = deque(maxlen=self.velocity_window)
            return self.tracked_objects

        # If no detections, predict object positions based on velocity and mark as disappeared
        if len(filtered_detections) == 0:
            for object_id in list(self.disappeared.keys()):
                # Predict new position based on velocity
                self._predict_position(object_id, dt)

                self.disappeared[object_id] += 1

                # Remove object if it has disappeared for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            # Update durations for remaining objects
            for object_id in self.tracked_objects:
                if object_id in self.object_history:
                    self.tracked_objects[object_id]['duration'] = self.get_object_duration(object_id)

            return self.tracked_objects

        # Predict new positions for all objects based on their velocity
        for object_id in list(self.objects.keys()):
            self._predict_position(object_id, dt)

        # Get bounding boxes of current objects
        object_ids = list(self.objects.keys())
        object_bboxes = []

        for obj_id in object_ids:
            if obj_id in self.object_history and self.object_history[obj_id]:
                last_detection = self.object_history[obj_id][-1]
                object_bboxes.append(last_detection.bbox)
            else:
                # If no history, use a dummy bbox
                object_bboxes.append([0, 0, 1, 1])

        object_bboxes = np.array(object_bboxes)

        # Get bounding boxes of new detections
        detection_bboxes = np.array([d.bbox for d in filtered_detections])

        # Calculate IoU between all pairs of bounding boxes
        ious = self._calculate_iou_matrix(object_bboxes, detection_bboxes)

        # Match detections to objects based on IoU
        matched_objects = set()
        matched_detections = set()

        # Sort IoUs in descending order and match greedily
        for obj_idx, det_idx in self._get_matches(ious):
            obj_id = object_ids[obj_idx]
            iou_value = ious[obj_idx, det_idx]

            # Only match if IoU is above threshold
            if iou_value >= self.iou_threshold:
                # Update object with new detection
                new_x = filtered_detections[det_idx].center_x
                new_y = filtered_detections[det_idx].center_y

                # Calculate velocity if we have previous positions
                if obj_id in self.objects:
                    prev_x, prev_y = self.objects[obj_id]
                    vx = (new_x - prev_x) / dt if dt > 0 else 0
                    vy = (new_y - prev_y) / dt if dt > 0 else 0

                    # Add to velocity history
                    if obj_id in self.velocity_history:
                        self.velocity_history[obj_id].append((vx, vy))

                # Update object position
                self.objects[obj_id] = (new_x, new_y)
                self.object_history[obj_id].append(filtered_detections[det_idx])
                self.disappeared[obj_id] = 0
                self.trajectories[obj_id].append((new_x, new_y))

                # Update tracked object info
                self.tracked_objects[obj_id] = {
                    'detection': filtered_detections[det_idx],
                    'class_id': filtered_detections[det_idx].class_id,
                    'class_name': filtered_detections[det_idx].class_name,
                    'bbox': filtered_detections[det_idx].bbox,
                    'center': (new_x, new_y),
                    'confidence': filtered_detections[det_idx].confidence,
                    'duration': self.get_object_duration(obj_id),
                    'trajectory': [(int(x), int(y)) for x, y in self.trajectories[obj_id]],
                    'iou': iou_value,
                    'velocity': self._get_average_velocity(obj_id)
                }

                matched_objects.add(obj_id)
                matched_detections.add(det_idx)

        # Check for disappeared objects
        for obj_id in set(self.objects.keys()) - matched_objects:
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self._deregister(obj_id)
                # Remove velocity history
                if obj_id in self.velocity_history:
                    del self.velocity_history[obj_id]
            else:
                # Update duration for objects that are still tracked
                if obj_id in self.tracked_objects:
                    self.tracked_objects[obj_id]['duration'] = self.get_object_duration(obj_id)

        # Register new detections
        for i in range(len(filtered_detections)):
            if i not in matched_detections:
                self._register(filtered_detections[i])
                # Initialize velocity history
                self.velocity_history[self.next_object_id - 1] = deque(maxlen=self.velocity_window)

        return self.tracked_objects

    def _predict_position(self, object_id: int, dt: float) -> None:
        """
        Predict the new position of an object based on its velocity.

        Args:
            object_id: ID of the object
            dt: Time delta since last update
        """
        if object_id not in self.objects:
            return

        # Get average velocity
        vx, vy = self._get_average_velocity(object_id)

        # Get current position
        x, y = self.objects[object_id]

        # Predict new position
        new_x = x + vx * dt
        new_y = y + vy * dt

        # Update object position
        self.objects[object_id] = (new_x, new_y)

    def _get_average_velocity(self, object_id: int) -> Tuple[float, float]:
        """
        Get the average velocity of an object.

        Args:
            object_id: ID of the object

        Returns:
            Tuple[float, float]: Average velocity (vx, vy)
        """
        if object_id not in self.velocity_history or not self.velocity_history[object_id]:
            return 0.0, 0.0

        velocities = list(self.velocity_history[object_id])
        vx = sum(v[0] for v in velocities) / len(velocities)
        vy = sum(v[1] for v in velocities) / len(velocities)

        return vx, vy

    def _calculate_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Calculate IoU between all pairs of bounding boxes.

        Args:
            boxes1: Array of bounding boxes in format [x, y, w, h]
            boxes2: Array of bounding boxes in format [x, y, w, h]

        Returns:
            np.ndarray: Matrix of IoU values
        """
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        boxes1_xyxy = np.zeros_like(boxes1)
        boxes1_xyxy[:, 0] = boxes1[:, 0]
        boxes1_xyxy[:, 1] = boxes1[:, 1]
        boxes1_xyxy[:, 2] = boxes1[:, 0] + boxes1[:, 2]
        boxes1_xyxy[:, 3] = boxes1[:, 1] + boxes1[:, 3]

        boxes2_xyxy = np.zeros_like(boxes2)
        boxes2_xyxy[:, 0] = boxes2[:, 0]
        boxes2_xyxy[:, 1] = boxes2[:, 1]
        boxes2_xyxy[:, 2] = boxes2[:, 0] + boxes2[:, 2]
        boxes2_xyxy[:, 3] = boxes2[:, 1] + boxes2[:, 3]

        # Calculate IoU
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))

        for i in range(len(boxes1)):
            for j in range(len(boxes2)):
                iou_matrix[i, j] = self._calculate_iou(boxes1_xyxy[i], boxes2_xyxy[j])

        return iou_matrix

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate IoU between two bounding boxes.

        Args:
            box1: Bounding box in format [x1, y1, x2, y2]
            box2: Bounding box in format [x1, y1, x2, y2]

        Returns:
            float: IoU value
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection_area = (x2 - x1) * (y2 - y1)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0

        return iou

    def _get_matches(self, iou_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Get matches between objects and detections based on IoU.

        Args:
            iou_matrix: Matrix of IoU values

        Returns:
            List[Tuple[int, int]]: List of (object_index, detection_index) pairs
        """
        # Flatten the matrix and get indices in descending order of IoU
        flat_indices = np.argsort(iou_matrix.flatten())[::-1]

        # Convert flat indices to 2D indices
        matches = []
        matched_rows = set()
        matched_cols = set()

        for flat_idx in flat_indices:
            row = flat_idx // iou_matrix.shape[1]
            col = flat_idx % iou_matrix.shape[1]

            # Skip if already matched or IoU is zero
            if row in matched_rows or col in matched_cols or iou_matrix[row, col] == 0:
                continue

            matches.append((row, col))
            matched_rows.add(row)
            matched_cols.add(col)

        return matches
