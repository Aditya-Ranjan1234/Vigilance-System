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
from vigilance_system.detection.ml_algorithms import (
    KNNTracker, SVMTracker, NaiveBayesTracker,
    DecisionTreeTracker, RandomForestTracker
)
from vigilance_system.detection.tracking_algorithms import (
    IoUTracker, KalmanTracker, YOLOv8Tracker
)

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

    def update(self, detections: List[Detection]) -> Dict[int, Dict[str, Any]]:
        """
        Update object tracking with new detections.

        Args:
            detections: List of new detections

        Returns:
            Dict[int, Dict[str, Any]]: Dictionary mapping object IDs to their current info
        """
        # If no objects are being tracked, register all detections
        if len(self.objects) == 0:
            for detection in detections:
                self._register(detection)

            # Return in the same format as KNNTracker and SVMTracker
            return {obj_id: {
                'detection': self.object_history[obj_id][-1],
                'class_id': self.object_history[obj_id][-1].class_id,
                'class_name': self.object_history[obj_id][-1].class_name,
                'bbox': self.object_history[obj_id][-1].bbox,
                'center': (self.object_history[obj_id][-1].center_x, self.object_history[obj_id][-1].center_y),
                'confidence': self.object_history[obj_id][-1].confidence,
                'duration': self.get_object_duration(obj_id),
                'trajectory': [(int(self.object_history[obj_id][i].center_x),
                               int(self.object_history[obj_id][i].center_y))
                              for i in range(len(self.object_history[obj_id]))]
            } for obj_id in self.objects}

        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # Remove object if it has disappeared for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            # Return in the same format as KNNTracker and SVMTracker
            return {obj_id: {
                'detection': self.object_history[obj_id][-1],
                'class_id': self.object_history[obj_id][-1].class_id,
                'class_name': self.object_history[obj_id][-1].class_name,
                'bbox': self.object_history[obj_id][-1].bbox,
                'center': (self.object_history[obj_id][-1].center_x, self.object_history[obj_id][-1].center_y),
                'confidence': self.object_history[obj_id][-1].confidence,
                'duration': self.get_object_duration(obj_id),
                'trajectory': [(int(self.object_history[obj_id][i].center_x),
                               int(self.object_history[obj_id][i].center_y))
                              for i in range(len(self.object_history[obj_id]))]
            } for obj_id in self.objects}

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

        # Return in the same format as KNNTracker and SVMTracker
        result = {}
        for obj_id in self.objects:
            if obj_id in self.object_history and len(self.object_history[obj_id]) > 0:
                detection = self.object_history[obj_id][-1]
                result[obj_id] = {
                    'detection': detection,
                    'class_id': detection.class_id,
                    'class_name': detection.class_name,
                    'bbox': detection.bbox,
                    'center': (detection.center_x, detection.center_y),
                    'confidence': detection.confidence,
                    'duration': self.get_object_duration(obj_id),
                    'trajectory': [(int(self.object_history[obj_id][i].center_x),
                                   int(self.object_history[obj_id][i].center_y))
                                  for i in range(len(self.object_history[obj_id]))]
                }

        return result

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

        # Tracking parameters
        self.max_disappeared = config.get('tracking.max_disappeared', 30)
        self.max_distance = config.get('tracking.max_distance', 50.0)

        # Initialize trackers for each camera
        self.trackers = {}

        # Initialize alert state
        self.last_alert_time = defaultdict(float)
        self.alert_cooldown = 10.0  # seconds

        # Initialize tracked objects for visualization
        self.tracked_objects = {}  # Map from camera_name to dict of tracked objects
        self.crowd_threshold = config.get('alerts.crowd_threshold', 3)

        # Initialize classifier and analysis algorithms
        self.classifier_algorithm = config.get('classification.algorithm', 'svm')
        self.analysis_algorithm = config.get('analysis.algorithm', 'basic')

        logger.info(f"Initialized decision maker with motion_threshold={self.motion_threshold}, "
                   f"person_loitering_time={self.person_loitering_time}")

    def get_tracker(self, camera_name: str) -> Any:
        """
        Get or create a tracker for a camera.

        Args:
            camera_name: Name of the camera

        Returns:
            Any: Tracker for the camera (DetectionTracker, KNNTracker, or SVMTracker)
        """
        # Always get the current tracking algorithm from config
        tracking_algorithm = config.get('tracking.algorithm', 'centroid').lower()

        # Check if we need to create a new tracker or if the algorithm has changed
        create_new_tracker = False

        if camera_name not in self.trackers:
            create_new_tracker = True
            logger.info(f"Creating new tracker for camera '{camera_name}' with algorithm '{tracking_algorithm}'")
        else:
            # Check if the current tracker matches the configured algorithm
            current_tracker = self.trackers[camera_name]
            if (tracking_algorithm == 'knn' and not isinstance(current_tracker, KNNTracker)) or \
               (tracking_algorithm == 'svm' and not isinstance(current_tracker, SVMTracker)) or \
               (tracking_algorithm == 'naive_bayes' and not isinstance(current_tracker, NaiveBayesTracker)) or \
               (tracking_algorithm == 'decision_tree' and not isinstance(current_tracker, DecisionTreeTracker)) or \
               (tracking_algorithm == 'random_forest' and not isinstance(current_tracker, RandomForestTracker)) or \
               (tracking_algorithm == 'iou' and not isinstance(current_tracker, IoUTracker)) or \
               (tracking_algorithm == 'kalman' and not isinstance(current_tracker, KalmanTracker)) or \
               (tracking_algorithm == 'yolov8' and not isinstance(current_tracker, YOLOv8Tracker)) or \
               (tracking_algorithm == 'centroid' and not isinstance(current_tracker, DetectionTracker)):
                create_new_tracker = True
                logger.info(f"Tracker algorithm changed for camera '{camera_name}' to '{tracking_algorithm}', creating new tracker")

        if create_new_tracker:
            if tracking_algorithm == 'knn':
                # Create KNN tracker
                n_neighbors = config.get('tracking.knn.n_neighbors', 3)
                self.trackers[camera_name] = KNNTracker(
                    max_disappeared=self.max_disappeared,
                    max_distance=self.max_distance,
                    n_neighbors=n_neighbors
                )
                logger.info(f"Created KNN tracker for camera '{camera_name}' with n_neighbors={n_neighbors}")
            elif tracking_algorithm == 'svm':
                # Create SVM tracker
                C = config.get('tracking.svm.C', 1.0)
                self.trackers[camera_name] = SVMTracker(
                    max_disappeared=self.max_disappeared,
                    max_distance=self.max_distance,
                    C=C
                )
                logger.info(f"Created SVM tracker for camera '{camera_name}' with C={C}")
            elif tracking_algorithm == 'naive_bayes':
                # Create Naive Bayes tracker
                self.trackers[camera_name] = NaiveBayesTracker(
                    max_disappeared=self.max_disappeared,
                    max_distance=self.max_distance
                )
                logger.info(f"Created Naive Bayes tracker for camera '{camera_name}'")
            elif tracking_algorithm == 'decision_tree':
                # Create Decision Tree tracker
                max_depth = config.get('tracking.decision_tree.max_depth', 5)
                self.trackers[camera_name] = DecisionTreeTracker(
                    max_disappeared=self.max_disappeared,
                    max_distance=self.max_distance,
                    max_depth=max_depth
                )
                logger.info(f"Created Decision Tree tracker for camera '{camera_name}' with max_depth={max_depth}")
            elif tracking_algorithm == 'random_forest':
                # Create Random Forest tracker
                n_estimators = config.get('tracking.random_forest.n_estimators', 10)
                self.trackers[camera_name] = RandomForestTracker(
                    max_disappeared=self.max_disappeared,
                    max_distance=self.max_distance,
                    n_estimators=n_estimators
                )
                logger.info(f"Created Random Forest tracker for camera '{camera_name}' with n_estimators={n_estimators}")
            elif tracking_algorithm == 'iou':
                # Create IoU tracker
                iou_threshold = config.get('tracking.iou.threshold', 0.3)
                self.trackers[camera_name] = IoUTracker(
                    max_disappeared=self.max_disappeared,
                    max_distance=self.max_distance,
                    iou_threshold=iou_threshold
                )
                logger.info(f"Created IoU tracker for camera '{camera_name}' with iou_threshold={iou_threshold}")
            elif tracking_algorithm == 'kalman':
                # Create Kalman filter tracker
                process_noise = config.get('tracking.kalman.process_noise', 1e-2)
                measurement_noise = config.get('tracking.kalman.measurement_noise', 1e-1)
                self.trackers[camera_name] = KalmanTracker(
                    max_disappeared=self.max_disappeared,
                    max_distance=self.max_distance,
                    process_noise=process_noise,
                    measurement_noise=measurement_noise
                )
                logger.info(f"Created Kalman tracker for camera '{camera_name}' with process_noise={process_noise}, measurement_noise={measurement_noise}")
            elif tracking_algorithm == 'yolov8':
                # Create YOLOv8 tracker
                iou_threshold = config.get('tracking.yolov8.iou_threshold', 0.3)
                confidence_threshold = config.get('tracking.yolov8.confidence_threshold', 0.5)
                self.trackers[camera_name] = YOLOv8Tracker(
                    max_disappeared=self.max_disappeared,
                    max_distance=self.max_distance,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold
                )
                logger.info(f"Created YOLOv8 tracker for camera '{camera_name}' with iou_threshold={iou_threshold}, confidence_threshold={confidence_threshold}")
            else:
                # Default to centroid tracker
                self.trackers[camera_name] = DetectionTracker(
                    max_disappeared=self.max_disappeared,
                    max_distance=self.max_distance
                )
                logger.info(f"Created centroid tracker for camera '{camera_name}'")

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

        # Store tracked objects for visualization
        self.tracked_objects[camera_name] = tracked_objects

        # Check for alert conditions
        alerts = []

        # Check for person loitering
        person_alerts = self._check_person_loitering(camera_name, tracked_objects)
        alerts.extend(person_alerts)

        # Check for multiple people
        crowd_alerts = self._check_crowd_detection(camera_name, tracked_objects)
        alerts.extend(crowd_alerts)

        return alerts

    def _check_person_loitering(self, camera_name: str, tracked_objects: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check for people loitering in the scene.

        Args:
            camera_name: Name of the camera
            tracked_objects: Dictionary of tracked objects

        Returns:
            List[Dict[str, Any]]: List of loitering alerts
        """
        alerts = []

        for obj_id, obj_info in tracked_objects.items():
            # Check if object is a person
            if obj_info['class_name'].lower() != 'person':
                continue

            # Check if person has been present for too long
            duration = obj_info['duration']

            if duration >= self.person_loitering_time:
                # Check cooldown
                alert_key = f"{camera_name}_loitering_{obj_id}"
                current_time = time.time()

                if current_time - self.last_alert_time[alert_key] >= self.alert_cooldown:
                    self.last_alert_time[alert_key] = current_time

                    # Get detection object
                    detection = obj_info['detection']

                    # Create alert
                    alert = {
                        'type': 'loitering',
                        'camera': camera_name,
                        'object_id': obj_id,
                        'duration': duration,
                        'detection': detection.to_dict() if hasattr(detection, 'to_dict') else detection,
                        'timestamp': current_time,
                        'message': f"Person loitering detected on camera '{camera_name}' for {duration:.1f} seconds"
                    }

                    # Add algorithm info if available
                    if self.classifier_algorithm == 'svm' and 'svm_confidence' in obj_info:
                        alert['svm_confidence'] = obj_info['svm_confidence']
                    elif self.classifier_algorithm == 'naive_bayes' and 'nb_confidence' in obj_info:
                        alert['nb_confidence'] = obj_info['nb_confidence']
                    elif self.classifier_algorithm == 'random_forest' and 'rf_confidence' in obj_info:
                        alert['rf_confidence'] = obj_info['rf_confidence']

                    alerts.append(alert)
                    logger.warning(f"Loitering alert: Person detected for {duration:.1f} seconds on camera '{camera_name}'")

        return alerts

    def _check_crowd_detection(self, camera_name: str, tracked_objects: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        people_count = sum(1 for obj_info in tracked_objects.values() if obj_info['class_name'].lower() == 'person')

        # Use the instance variable for crowd threshold
        if people_count >= self.crowd_threshold:
            # Check cooldown
            alert_key = f"{camera_name}_crowd"
            current_time = time.time()

            if current_time - self.last_alert_time[alert_key] >= self.alert_cooldown:
                self.last_alert_time[alert_key] = current_time

                # Create alert
                alert = {
                    'type': 'crowd',
                    'camera': camera_name,
                    'people_count': people_count,
                    'timestamp': current_time,
                    'message': f"Crowd detected on camera '{camera_name}': {people_count} people",
                    'algorithm': self.classifier_algorithm
                }

                # Add people locations
                people_locations = [
                    {
                        'object_id': obj_id,
                        'center': obj_info['center'],
                        'bbox': obj_info['bbox']
                    }
                    for obj_id, obj_info in tracked_objects.items()
                    if obj_info['class_name'].lower() == 'person'
                ]
                alert['people_locations'] = people_locations

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

    def reset_all_trackers(self) -> None:
        """
        Reset all trackers and recreate them with the current algorithm settings.
        This ensures that algorithm changes are properly applied.
        """
        # Store camera names before resetting
        camera_names = list(self.trackers.keys())

        # Clear all trackers and tracked objects
        self.trackers = {}
        self.tracked_objects = {}

        # Recreate trackers for each camera with the current algorithm settings
        for camera_name in camera_names:
            # This will create a new tracker with the current algorithm settings
            self.get_tracker(camera_name)

        logger.info(f"Reset all trackers using algorithm: {self.classifier_algorithm}")
        logger.info(f"Current tracking algorithm: {config.get('tracking.algorithm', 'centroid')}")

    def set_classifier_algorithm(self, algorithm_name: str) -> None:
        """
        Set the classifier algorithm to use.

        Args:
            algorithm_name: Name of the classifier algorithm
        """
        logger.info(f"Setting classifier algorithm to {algorithm_name}")

        # Store the algorithm name for use in processing
        self.classifier_algorithm = algorithm_name

        # Save to config to ensure persistence
        config.set('classification.algorithm', algorithm_name, save=True)
        logger.info(f"Saved classifier algorithm {algorithm_name} to config")

        # Reset any classifier-specific state
        self.reset_all_trackers()
        logger.info(f"Reset trackers for new classifier algorithm: {algorithm_name}")

    def set_analysis_algorithm(self, algorithm_name: str) -> None:
        """
        Set the analysis algorithm to use.

        Args:
            algorithm_name: Name of the analysis algorithm
        """
        logger.info(f"Setting analysis algorithm to {algorithm_name}")

        # Store the algorithm name for use in processing
        self.analysis_algorithm = algorithm_name

        # Save to config to ensure persistence
        config.set('analysis.algorithm', algorithm_name, save=True)
        logger.info(f"Saved analysis algorithm {algorithm_name} to config")

        # Reset any analysis-specific state
        self.reset_all_trackers()
        logger.info(f"Reset trackers for new analysis algorithm: {algorithm_name}")

    def set_tracking_algorithm(self, algorithm_name: str) -> None:
        """
        Set the tracking algorithm to use.

        Args:
            algorithm_name: Name of the tracking algorithm
        """
        logger.info(f"Setting tracking algorithm to {algorithm_name}")

        # Save to config to ensure persistence
        config.set('tracking.algorithm', algorithm_name, save=True)
        logger.info(f"Saved tracking algorithm {algorithm_name} to config")

        # Reset all trackers to apply the new algorithm
        self.reset_all_trackers()
        logger.info(f"Reset trackers for new tracking algorithm: {algorithm_name}")


# Create a default instance
decision_maker = DecisionMaker()
