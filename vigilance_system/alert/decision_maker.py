"""
Decision maker module for analyzing detections and making alert decisions.

This module provides functionality to analyze object detections and determine
when to trigger alerts based on various rules and thresholds.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque

from vigilance_system.utils.logger import get_logger
from vigilance_system.utils.config import config
from vigilance_system.detection.object_detector import Detection

# Initialize logger
logger = get_logger(__name__)


class DetectionTracker:
    """
    Tracks objects across multiple frames.

    Uses simple heuristics to associate detections across frames and track
    objects over time.
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0):
        """
        Initialize the detection tracker.

        Args:
            max_disappeared: Maximum number of frames an object can disappear before being removed
            max_distance: Maximum distance between detections to be considered the same object
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_object_id = 0
        self.objects = {}  # Map from object ID to centroid
        self.disappeared = defaultdict(int)  # Map from object ID to number of frames disappeared
        self.object_history = defaultdict(list)  # Map from object ID to list of detections

        logger.info(f"Initialized detection tracker with max_disappeared={max_disappeared}, "
                   f"max_distance={max_distance}")

    def update(self, detections: List[Detection]) -> Dict[int, Detection]:
        """
        Update object tracking with new detections.

        Args:
            detections: List of new detections

        Returns:
            Dict[int, Detection]: Dictionary mapping object IDs to their current detections
        """
        # If no objects are being tracked, register all detections
        if len(self.objects) == 0:
            for detection in detections:
                self._register(detection)
            return {obj_id: self.object_history[obj_id][-1] for obj_id in self.objects}

        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # Remove object if it has disappeared for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            return {obj_id: self.object_history[obj_id][-1] for obj_id in self.objects}

        # Match detections to existing objects
        object_centroids = {obj_id: (obj[0], obj[1]) for obj_id, obj in self.objects.items()}
        detection_centroids = [(d.center_x, d.center_y) for d in detections]

        # Calculate distances between all objects and detections
        distances = {}
        for obj_id, obj_centroid in object_centroids.items():
            for i, det_centroid in enumerate(detection_centroids):
                distance = ((obj_centroid[0] - det_centroid[0]) ** 2 +
                           (obj_centroid[1] - det_centroid[1]) ** 2) ** 0.5
                if distance <= self.max_distance:
                    distances[(obj_id, i)] = distance

        # Sort distances and match objects to detections
        matched_objects = set()
        matched_detections = set()

        # Sort by distance and match greedily
        for (obj_id, det_idx), distance in sorted(distances.items(), key=lambda x: x[1]):
            if obj_id not in matched_objects and det_idx not in matched_detections:
                self.objects[obj_id] = (detection_centroids[det_idx][0], detection_centroids[det_idx][1])
                self.object_history[obj_id].append(detections[det_idx])
                self.disappeared[obj_id] = 0
                matched_objects.add(obj_id)
                matched_detections.add(det_idx)

        # Check for disappeared objects
        for obj_id in set(self.objects.keys()) - matched_objects:
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self._deregister(obj_id)

        # Register new detections
        for i in range(len(detections)):
            if i not in matched_detections:
                self._register(detections[i])

        # Return current detections for each tracked object
        return {obj_id: self.object_history[obj_id][-1] for obj_id in self.objects}

    def _register(self, detection: Detection) -> None:
        """
        Register a new object.

        Args:
            detection: Detection to register as a new object
        """
        self.objects[self.next_object_id] = (detection.center_x, detection.center_y)
        self.object_history[self.next_object_id].append(detection)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id: int) -> None:
        """
        Deregister an object.

        Args:
            object_id: ID of the object to deregister
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.object_history[object_id]

    def get_object_history(self, object_id: int) -> List[Detection]:
        """
        Get the history of detections for an object.

        Args:
            object_id: ID of the object

        Returns:
            List[Detection]: List of detections for the object
        """
        return self.object_history.get(object_id, [])

    def get_object_duration(self, object_id: int) -> float:
        """
        Get the duration an object has been tracked.

        Args:
            object_id: ID of the object

        Returns:
            float: Duration in seconds
        """
        history = self.object_history.get(object_id, [])
        if len(history) < 2:
            return 0.0
        return history[-1].timestamp - history[0].timestamp

    def get_all_objects(self) -> Dict[int, List[Detection]]:
        """
        Get all tracked objects and their histories.

        Returns:
            Dict[int, List[Detection]]: Dictionary mapping object IDs to their detection histories
        """
        return dict(self.object_history)


class DecisionMaker:
    """
    Makes decisions about when to trigger alerts based on detections.

    Analyzes tracked objects and applies rules to determine when security
    conditions are met that warrant an alert.
    """

    def __init__(self):
        """Initialize the decision maker."""
        # Load configuration
        self.motion_threshold = config.get('alerts.motion_threshold', 0.2)
        self.person_loitering_time = config.get('alerts.person_loitering_time', 30)

        # Initialize trackers for each camera
        self.trackers = {}

        # Initialize alert state
        self.last_alert_time = defaultdict(float)
        self.alert_cooldown = config.get('alerts.cooldown', 60.0)  # seconds - default to 60 seconds

        logger.info(f"Initialized decision maker with motion_threshold={self.motion_threshold}, "
                   f"person_loitering_time={self.person_loitering_time}, "
                   f"alert_cooldown={self.alert_cooldown} seconds")

    def get_tracker(self, camera_name: str) -> DetectionTracker:
        """
        Get or create a tracker for a camera.

        Args:
            camera_name: Name of the camera

        Returns:
            DetectionTracker: Tracker for the camera
        """
        if camera_name not in self.trackers:
            self.trackers[camera_name] = DetectionTracker()
        return self.trackers[camera_name]

    def process_detections(self, camera_name: str, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Process detections and generate alerts if necessary.

        Args:
            camera_name: Name of the camera
            detections: List of detections from the camera

        Returns:
            List[Dict[str, Any]]: List of alerts generated
        """
        # Get tracker for this camera
        tracker = self.get_tracker(camera_name)

        # Update tracker with new detections
        tracked_objects = tracker.update(detections)

        # Check for alert conditions
        alerts = []

        # Check for person loitering
        person_alerts = self._check_person_loitering(camera_name, tracked_objects)
        alerts.extend(person_alerts)

        # Check for multiple people
        crowd_alerts = self._check_crowd_detection(camera_name, tracked_objects)
        alerts.extend(crowd_alerts)

        return alerts

    def _check_person_loitering(self, camera_name: str, tracked_objects: Dict[int, Detection]) -> List[Dict[str, Any]]:
        """
        Check for people loitering in the scene.

        Args:
            camera_name: Name of the camera
            tracked_objects: Dictionary of tracked objects

        Returns:
            List[Dict[str, Any]]: List of loitering alerts
        """
        alerts = []
        tracker = self.trackers[camera_name]

        for obj_id, detection in tracked_objects.items():
            # Check if object is a person
            if detection.class_name.lower() != 'person':
                continue

            # Check if person has been present for too long
            duration = tracker.get_object_duration(obj_id)

            if duration >= self.person_loitering_time:
                # Check cooldown using the new method
                if self._check_alert_cooldown(camera_name, f"loitering_{obj_id}"):
                    current_time = time.time()

                    alert = {
                        'type': 'loitering',
                        'camera': camera_name,
                        'object_id': obj_id,
                        'duration': duration,
                        'detection': detection.to_dict(),
                        'timestamp': current_time,
                        'message': f"Person loitering detected on camera '{camera_name}' for {duration:.1f} seconds"
                    }

                    alerts.append(alert)
                    logger.warning(f"Loitering alert: Person detected for {duration:.1f} seconds on camera '{camera_name}'")

        return alerts

    def _check_crowd_detection(self, camera_name: str, tracked_objects: Dict[int, Detection]) -> List[Dict[str, Any]]:
        """
        Check for crowds (multiple people) in the scene.

        Args:
            camera_name: Name of the camera
            tracked_objects: Dictionary of tracked objects

        Returns:
            List[Dict[str, Any]]: List of crowd alerts
        """
        alerts = []

        # Count people
        people_count = sum(1 for d in tracked_objects.values() if d.class_name.lower() == 'person')

        # Alert threshold (configurable)
        crowd_threshold = config.get('alerts.crowd_threshold', 3)

        if people_count >= crowd_threshold:
            # Check cooldown using the new method
            if self._check_alert_cooldown(camera_name, "crowd"):
                current_time = time.time()

                alert = {
                    'type': 'crowd',
                    'camera': camera_name,
                    'people_count': people_count,
                    'timestamp': current_time,
                    'message': f"Crowd detected on camera '{camera_name}': {people_count} people"
                }

                alerts.append(alert)
                logger.warning(f"Crowd alert: {people_count} people detected on camera '{camera_name}'")

        return alerts

    def reset_tracker(self, camera_name: str) -> None:
        """
        Reset the tracker for a camera.

        Args:
            camera_name: Name of the camera
        """
        if camera_name in self.trackers:
            del self.trackers[camera_name]
            logger.info(f"Reset tracker for camera '{camera_name}'")

    def _check_alert_cooldown(self, camera_name: str, alert_type: str) -> bool:
        """
        Check if an alert is in cooldown.

        Args:
            camera_name: Camera name
            alert_type: Alert type

        Returns:
            True if alert is allowed (not in cooldown), False otherwise
        """
        key = f"{camera_name}_{alert_type}"
        current_time = time.time()

        # Check if alert is in cooldown
        if current_time - self.last_alert_time[key] < self.alert_cooldown:
            # Log that alert is in cooldown
            logger.debug(f"Alert {alert_type} for {camera_name} is in cooldown. "
                        f"Next alert in {self.alert_cooldown - (current_time - self.last_alert_time[key]):.1f} seconds")
            return False

        # Update last alert time
        self.last_alert_time[key] = current_time
        logger.debug(f"Alert {alert_type} for {camera_name} triggered. Cooldown set for {self.alert_cooldown} seconds")
        return True


# Create a default instance
decision_maker = DecisionMaker()
