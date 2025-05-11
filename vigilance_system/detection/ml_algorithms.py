"""
Machine learning algorithms for object tracking and classification.

This module provides implementations of various machine learning algorithms
for object tracking and classification, including KNN, SVM, and Naive Bayes.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque

from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.object_detector import Detection

# Initialize logger
logger = get_logger(__name__)


class MLTracker:
    """Base class for ML-based trackers."""

    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0):
        """
        Initialize the ML tracker.

        Args:
            max_disappeared: Maximum number of frames an object can disappear before being removed
            max_distance: Maximum distance between detections to be considered the same object
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_object_id = 0
        self.objects = {}  # Map from object ID to centroid
        self.disappeared = {}  # Map from object ID to number of frames disappeared
        self.object_history = {}  # Map from object ID to list of detections
        self.trajectories = {}  # Map from object ID to trajectory points

        # For visualization
        self.tracked_objects = {}  # Map from object ID to tracked object info

        logger.info(f"Initialized ML tracker with max_disappeared={max_disappeared}, "
                   f"max_distance={max_distance}")

    def update(self, detections: List[Detection]) -> Dict[int, Dict[str, Any]]:
        """
        Update object tracking with new detections.

        Args:
            detections: List of new detections

        Returns:
            Dict[int, Dict[str, Any]]: Dictionary mapping object IDs to their current info
        """
        # Implement in subclasses
        raise NotImplementedError("Subclasses must implement update method")

    def _register(self, detection: Detection) -> int:
        """
        Register a new object.

        Args:
            detection: Detection to register as a new object

        Returns:
            int: ID of the registered object
        """
        object_id = self.next_object_id
        self.objects[object_id] = (detection.center_x, detection.center_y)
        self.object_history[object_id] = [detection]
        self.disappeared[object_id] = 0
        self.trajectories[object_id] = deque([(detection.center_x, detection.center_y)], maxlen=50)

        # Update tracked objects for visualization
        self.tracked_objects[object_id] = {
            'detection': detection,
            'class_id': detection.class_id,
            'class_name': detection.class_name,
            'bbox': detection.bbox,
            'center': (detection.center_x, detection.center_y),
            'confidence': detection.confidence,
            'duration': 0,
            'trajectory': [(int(x), int(y)) for x, y in self.trajectories[object_id]]
        }

        self.next_object_id += 1
        return object_id

    def _deregister(self, object_id: int) -> None:
        """
        Deregister an object.

        Args:
            object_id: ID of the object to deregister
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.object_history[object_id]
        del self.trajectories[object_id]

        if object_id in self.tracked_objects:
            del self.tracked_objects[object_id]

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

        # Calculate duration based on frame timestamps
        return history[-1].timestamp - history[0].timestamp


class KNNTracker(MLTracker):
    """KNN-based object tracker."""

    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0, n_neighbors: int = 3):
        """
        Initialize the KNN tracker.

        Args:
            max_disappeared: Maximum number of frames an object can disappear before being removed
            max_distance: Maximum distance between detections to be considered the same object
            n_neighbors: Number of neighbors to use for KNN
        """
        super().__init__(max_disappeared, max_distance)
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.is_fitted = False

        logger.info(f"Initialized KNN tracker with n_neighbors={n_neighbors}")

    def update(self, detections: List[Detection]) -> Dict[int, Dict[str, Any]]:
        """
        Update object tracking with new detections using KNN.

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

        # Extract features from current objects and detections
        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[obj_id] for obj_id in object_ids])
        detection_centroids = np.array([(d.center_x, d.center_y) for d in detections])

        # If we have enough data, use KNN to match detections to objects
        if len(object_ids) >= self.n_neighbors and self.is_fitted:
            # Scale features
            scaled_objects = self.scaler.transform(object_centroids)
            scaled_detections = self.scaler.transform(detection_centroids)

            # Predict closest objects for each detection
            distances, indices = self.knn.kneighbors(scaled_detections)

            # Match detections to objects based on KNN results
            matched_objects = set()
            matched_detections = set()

            # Sort by distance and match greedily
            for det_idx, (dist, idx) in enumerate(zip(distances, indices)):
                # Find the closest unmatched object
                for d, i in zip(dist, idx):
                    if i < len(object_ids) and object_ids[i] not in matched_objects and det_idx not in matched_detections:
                        obj_id = object_ids[i]

                        # Only match if distance is below threshold
                        if d <= self.max_distance:
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
                                'trajectory': [(int(x), int(y)) for x, y in self.trajectories[obj_id]]
                            }

                            matched_objects.add(obj_id)
                            matched_detections.add(det_idx)
                            break
        else:
            # Fall back to simple distance-based matching
            distances = {}
            for obj_id, obj_centroid in zip(object_ids, object_centroids):
                for i, det_centroid in enumerate(detection_centroids):
                    distance = np.linalg.norm(obj_centroid - det_centroid)
                    if distance <= self.max_distance:
                        distances[(obj_id, i)] = distance

            # Sort by distance and match greedily
            matched_objects = set()
            matched_detections = set()

            for (obj_id, det_idx), distance in sorted(distances.items(), key=lambda x: x[1]):
                if obj_id not in matched_objects and det_idx not in matched_detections:
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
                        'trajectory': list(self.trajectories[obj_id])
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

        # Update KNN model with current objects
        if len(self.objects) >= self.n_neighbors:
            object_centroids = np.array([self.objects[obj_id] for obj_id in self.objects.keys()])
            object_ids = np.array(list(self.objects.keys()))

            # Scale features
            self.scaler.fit(object_centroids)
            scaled_centroids = self.scaler.transform(object_centroids)

            # Fit KNN model
            self.knn.fit(scaled_centroids, object_ids)
            self.is_fitted = True

        return self.tracked_objects


class SVMTracker(MLTracker):
    """SVM-based object tracker."""

    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0, C: float = 1.0):
        """
        Initialize the SVM tracker.

        Args:
            max_disappeared: Maximum number of frames an object can disappear before being removed
            max_distance: Maximum distance between detections to be considered the same object
            C: Regularization parameter for SVM
        """
        super().__init__(max_disappeared, max_distance)
        self.C = C
        self.svm = SVC(C=C, kernel='rbf', probability=True)
        self.scaler = StandardScaler()
        self.is_fitted = False

        logger.info(f"Initialized SVM tracker with C={C}")

    def update(self, detections: List[Detection]) -> Dict[int, Dict[str, Any]]:
        """
        Update object tracking with new detections using SVM.

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

        # Extract features from current objects and detections
        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[obj_id] for obj_id in object_ids])
        detection_centroids = np.array([(d.center_x, d.center_y) for d in detections])

        # If we have enough data and classes, use SVM to match detections to objects
        if len(object_ids) >= 2 and len(set(object_ids)) >= 2 and self.is_fitted:
            # Scale features
            scaled_objects = self.scaler.transform(object_centroids)
            scaled_detections = self.scaler.transform(detection_centroids)

            # Predict probabilities for each detection
            probabilities = self.svm.predict_proba(scaled_detections)

            # Match detections to objects based on SVM probabilities
            matched_objects = set()
            matched_detections = set()

            # For each detection, find the most probable object
            for det_idx, probs in enumerate(probabilities):
                # Get object ID with highest probability
                max_prob_idx = np.argmax(probs)
                max_prob = probs[max_prob_idx]
                predicted_obj_id = self.svm.classes_[max_prob_idx]

                # Only match if probability is high enough
                if max_prob >= 0.6 and predicted_obj_id in object_ids and predicted_obj_id not in matched_objects:
                    # Calculate distance to ensure it's reasonable
                    obj_centroid = self.objects[predicted_obj_id]
                    det_centroid = (detections[det_idx].center_x, detections[det_idx].center_y)
                    distance = np.linalg.norm(np.array(obj_centroid) - np.array(det_centroid))

                    if distance <= self.max_distance:
                        # Update object with new detection
                        self.objects[predicted_obj_id] = det_centroid
                        self.object_history[predicted_obj_id].append(detections[det_idx])
                        self.disappeared[predicted_obj_id] = 0
                        self.trajectories[predicted_obj_id].append(det_centroid)

                        # Update tracked object info
                        self.tracked_objects[predicted_obj_id] = {
                            'detection': detections[det_idx],
                            'class_id': detections[det_idx].class_id,
                            'class_name': detections[det_idx].class_name,
                            'bbox': detections[det_idx].bbox,
                            'center': det_centroid,
                            'confidence': detections[det_idx].confidence,
                            'duration': self.get_object_duration(predicted_obj_id),
                            'trajectory': [(int(x), int(y)) for x, y in self.trajectories[predicted_obj_id]],
                            'svm_confidence': max_prob
                        }

                        matched_objects.add(predicted_obj_id)
                        matched_detections.add(det_idx)
        else:
            # Fall back to simple distance-based matching
            distances = {}
            for obj_id, obj_centroid in zip(object_ids, object_centroids):
                for i, det_centroid in enumerate(detection_centroids):
                    distance = np.linalg.norm(obj_centroid - det_centroid)
                    if distance <= self.max_distance:
                        distances[(obj_id, i)] = distance

            # Sort by distance and match greedily
            matched_objects = set()
            matched_detections = set()

            for (obj_id, det_idx), distance in sorted(distances.items(), key=lambda x: x[1]):
                if obj_id not in matched_objects and det_idx not in matched_detections:
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
                        'trajectory': list(self.trajectories[obj_id])
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

        # Update SVM model with current objects if we have enough data
        if len(self.objects) >= 2 and len(set(self.objects.keys())) >= 2:
            object_centroids = np.array([self.objects[obj_id] for obj_id in self.objects.keys()])
            object_ids = np.array(list(self.objects.keys()))

            # Scale features
            self.scaler.fit(object_centroids)
            scaled_centroids = self.scaler.transform(object_centroids)

            # Fit SVM model
            try:
                self.svm.fit(scaled_centroids, object_ids)
                self.is_fitted = True
            except Exception as e:
                logger.warning(f"Error fitting SVM model: {str(e)}")

        return self.tracked_objects


class NaiveBayesTracker(MLTracker):
    """Naive Bayes-based object tracker."""

    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0):
        """
        Initialize the Naive Bayes tracker.

        Args:
            max_disappeared: Maximum number of frames an object can disappear before being removed
            max_distance: Maximum distance between detections to be considered the same object
        """
        super().__init__(max_disappeared, max_distance)
        self.nb = GaussianNB()
        self.scaler = StandardScaler()
        self.is_fitted = False

        logger.info(f"Initialized Naive Bayes tracker")

    def update(self, detections: List[Detection]) -> Dict[int, Dict[str, Any]]:
        """
        Update object tracking with new detections using Naive Bayes.

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

        # Extract features from current objects and detections
        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[obj_id] for obj_id in object_ids])
        detection_centroids = np.array([(d.center_x, d.center_y) for d in detections])

        # If we have enough data and classes, use Naive Bayes to match detections to objects
        if len(object_ids) >= 2 and len(set(object_ids)) >= 2 and self.is_fitted:
            # Scale features
            scaled_detections = self.scaler.transform(detection_centroids)

            # Predict probabilities for each detection
            try:
                probabilities = self.nb.predict_proba(scaled_detections)

                # Match detections to objects based on Naive Bayes probabilities
                matched_objects = set()
                matched_detections = set()

                # For each detection, find the most probable object
                for det_idx, probs in enumerate(probabilities):
                    # Get object ID with highest probability
                    max_prob_idx = np.argmax(probs)
                    max_prob = probs[max_prob_idx]
                    predicted_obj_id = self.nb.classes_[max_prob_idx]

                    # Only match if probability is high enough
                    if max_prob >= 0.6 and predicted_obj_id in object_ids and predicted_obj_id not in matched_objects:
                        # Calculate distance to ensure it's reasonable
                        obj_centroid = self.objects[predicted_obj_id]
                        det_centroid = (detections[det_idx].center_x, detections[det_idx].center_y)
                        distance = np.linalg.norm(np.array(obj_centroid) - np.array(det_centroid))

                        if distance <= self.max_distance:
                            # Update object with new detection
                            self.objects[predicted_obj_id] = det_centroid
                            self.object_history[predicted_obj_id].append(detections[det_idx])
                            self.disappeared[predicted_obj_id] = 0
                            self.trajectories[predicted_obj_id].append(det_centroid)

                            # Update tracked object info
                            self.tracked_objects[predicted_obj_id] = {
                                'detection': detections[det_idx],
                                'class_id': detections[det_idx].class_id,
                                'class_name': detections[det_idx].class_name,
                                'bbox': detections[det_idx].bbox,
                                'center': det_centroid,
                                'confidence': detections[det_idx].confidence,
                                'duration': self.get_object_duration(predicted_obj_id),
                                'trajectory': [(int(x), int(y)) for x, y in self.trajectories[predicted_obj_id]],
                                'nb_confidence': max_prob
                            }

                            matched_objects.add(predicted_obj_id)
                            matched_detections.add(det_idx)
            except Exception as e:
                logger.warning(f"Error using Naive Bayes for prediction: {str(e)}")
                # Fall back to distance-based matching
                matched_objects = set()
                matched_detections = set()
        else:
            # Fall back to simple distance-based matching
            distances = {}
            for obj_id, obj_centroid in zip(object_ids, object_centroids):
                for i, det_centroid in enumerate(detection_centroids):
                    distance = np.linalg.norm(obj_centroid - det_centroid)
                    if distance <= self.max_distance:
                        distances[(obj_id, i)] = distance

            # Sort by distance and match greedily
            matched_objects = set()
            matched_detections = set()

            for (obj_id, det_idx), distance in sorted(distances.items(), key=lambda x: x[1]):
                if obj_id not in matched_objects and det_idx not in matched_detections:
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
                        'trajectory': list(self.trajectories[obj_id])
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

        # Update Naive Bayes model with current objects if we have enough data
        if len(self.objects) >= 2 and len(set(self.objects.keys())) >= 2:
            object_centroids = np.array([self.objects[obj_id] for obj_id in self.objects.keys()])
            object_ids = np.array(list(self.objects.keys()))

            # Scale features
            self.scaler.fit(object_centroids)
            scaled_centroids = self.scaler.transform(object_centroids)

            # Fit Naive Bayes model
            try:
                self.nb.fit(scaled_centroids, object_ids)
                self.is_fitted = True
            except Exception as e:
                logger.warning(f"Error fitting Naive Bayes model: {str(e)}")

        return self.tracked_objects


class DecisionTreeTracker(MLTracker):
    """Decision Tree-based object tracker."""

    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0, max_depth: int = 5):
        """
        Initialize the Decision Tree tracker.

        Args:
            max_disappeared: Maximum number of frames an object can disappear before being removed
            max_distance: Maximum distance between detections to be considered the same object
            max_depth: Maximum depth of the decision tree
        """
        super().__init__(max_disappeared, max_distance)
        self.max_depth = max_depth
        self.dt = DecisionTreeClassifier(max_depth=max_depth)
        self.scaler = StandardScaler()
        self.is_fitted = False

        logger.info(f"Initialized Decision Tree tracker with max_depth={max_depth}")

    def update(self, detections: List[Detection]) -> Dict[int, Dict[str, Any]]:
        """
        Update object tracking with new detections using Decision Tree.

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

        # Extract features from current objects and detections
        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[obj_id] for obj_id in object_ids])
        detection_centroids = np.array([(d.center_x, d.center_y) for d in detections])

        # If we have enough data and classes, use Decision Tree to match detections to objects
        if len(object_ids) >= 2 and len(set(object_ids)) >= 2 and self.is_fitted:
            # Scale features
            scaled_detections = self.scaler.transform(detection_centroids)

            # Predict for each detection
            try:
                predictions = self.dt.predict(scaled_detections)

                # Match detections to objects based on Decision Tree predictions
                matched_objects = set()
                matched_detections = set()

                # For each detection, find the predicted object
                for det_idx, predicted_obj_id in enumerate(predictions):
                    # Only match if the predicted object exists and hasn't been matched yet
                    if predicted_obj_id in object_ids and predicted_obj_id not in matched_objects:
                        # Calculate distance to ensure it's reasonable
                        obj_centroid = self.objects[predicted_obj_id]
                        det_centroid = (detections[det_idx].center_x, detections[det_idx].center_y)
                        distance = np.linalg.norm(np.array(obj_centroid) - np.array(det_centroid))

                        if distance <= self.max_distance:
                            # Update object with new detection
                            self.objects[predicted_obj_id] = det_centroid
                            self.object_history[predicted_obj_id].append(detections[det_idx])
                            self.disappeared[predicted_obj_id] = 0
                            self.trajectories[predicted_obj_id].append(det_centroid)

                            # Update tracked object info
                            self.tracked_objects[predicted_obj_id] = {
                                'detection': detections[det_idx],
                                'class_id': detections[det_idx].class_id,
                                'class_name': detections[det_idx].class_name,
                                'bbox': detections[det_idx].bbox,
                                'center': det_centroid,
                                'confidence': detections[det_idx].confidence,
                                'duration': self.get_object_duration(predicted_obj_id),
                                'trajectory': [(int(x), int(y)) for x, y in self.trajectories[predicted_obj_id]]
                            }

                            matched_objects.add(predicted_obj_id)
                            matched_detections.add(det_idx)
            except Exception as e:
                logger.warning(f"Error using Decision Tree for prediction: {str(e)}")
                # Fall back to distance-based matching
                matched_objects = set()
                matched_detections = set()
        else:
            # Fall back to simple distance-based matching
            distances = {}
            for obj_id, obj_centroid in zip(object_ids, object_centroids):
                for i, det_centroid in enumerate(detection_centroids):
                    distance = np.linalg.norm(obj_centroid - det_centroid)
                    if distance <= self.max_distance:
                        distances[(obj_id, i)] = distance

            # Sort by distance and match greedily
            matched_objects = set()
            matched_detections = set()

            for (obj_id, det_idx), distance in sorted(distances.items(), key=lambda x: x[1]):
                if obj_id not in matched_objects and det_idx not in matched_detections:
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
                        'trajectory': list(self.trajectories[obj_id])
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

        # Update Decision Tree model with current objects if we have enough data
        if len(self.objects) >= 2 and len(set(self.objects.keys())) >= 2:
            object_centroids = np.array([self.objects[obj_id] for obj_id in self.objects.keys()])
            object_ids = np.array(list(self.objects.keys()))

            # Scale features
            self.scaler.fit(object_centroids)
            scaled_centroids = self.scaler.transform(object_centroids)

            # Fit Decision Tree model
            try:
                self.dt.fit(scaled_centroids, object_ids)
                self.is_fitted = True
            except Exception as e:
                logger.warning(f"Error fitting Decision Tree model: {str(e)}")

        return self.tracked_objects


class RandomForestTracker(MLTracker):
    """Random Forest-based object tracker."""

    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0, n_estimators: int = 10):
        """
        Initialize the Random Forest tracker.

        Args:
            max_disappeared: Maximum number of frames an object can disappear before being removed
            max_distance: Maximum distance between detections to be considered the same object
            n_estimators: Number of trees in the forest
        """
        super().__init__(max_disappeared, max_distance)
        self.n_estimators = n_estimators
        self.rf = RandomForestClassifier(n_estimators=n_estimators)
        self.scaler = StandardScaler()
        self.is_fitted = False

        logger.info(f"Initialized Random Forest tracker with n_estimators={n_estimators}")

    def update(self, detections: List[Detection]) -> Dict[int, Dict[str, Any]]:
        """
        Update object tracking with new detections using Random Forest.

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

        # Extract features from current objects and detections
        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[obj_id] for obj_id in object_ids])
        detection_centroids = np.array([(d.center_x, d.center_y) for d in detections])

        # If we have enough data and classes, use Random Forest to match detections to objects
        if len(object_ids) >= 2 and len(set(object_ids)) >= 2 and self.is_fitted:
            # Scale features
            scaled_detections = self.scaler.transform(detection_centroids)

            # Predict probabilities for each detection
            try:
                probabilities = self.rf.predict_proba(scaled_detections)

                # Match detections to objects based on Random Forest probabilities
                matched_objects = set()
                matched_detections = set()

                # For each detection, find the most probable object
                for det_idx, probs in enumerate(probabilities):
                    # Get object ID with highest probability
                    max_prob_idx = np.argmax(probs)
                    max_prob = probs[max_prob_idx]
                    predicted_obj_id = self.rf.classes_[max_prob_idx]

                    # Only match if probability is high enough
                    if max_prob >= 0.6 and predicted_obj_id in object_ids and predicted_obj_id not in matched_objects:
                        # Calculate distance to ensure it's reasonable
                        obj_centroid = self.objects[predicted_obj_id]
                        det_centroid = (detections[det_idx].center_x, detections[det_idx].center_y)
                        distance = np.linalg.norm(np.array(obj_centroid) - np.array(det_centroid))

                        if distance <= self.max_distance:
                            # Update object with new detection
                            self.objects[predicted_obj_id] = det_centroid
                            self.object_history[predicted_obj_id].append(detections[det_idx])
                            self.disappeared[predicted_obj_id] = 0
                            self.trajectories[predicted_obj_id].append(det_centroid)

                            # Update tracked object info
                            self.tracked_objects[predicted_obj_id] = {
                                'detection': detections[det_idx],
                                'class_id': detections[det_idx].class_id,
                                'class_name': detections[det_idx].class_name,
                                'bbox': detections[det_idx].bbox,
                                'center': det_centroid,
                                'confidence': detections[det_idx].confidence,
                                'duration': self.get_object_duration(predicted_obj_id),
                                'trajectory': [(int(x), int(y)) for x, y in self.trajectories[predicted_obj_id]],
                                'rf_confidence': max_prob
                            }

                            matched_objects.add(predicted_obj_id)
                            matched_detections.add(det_idx)
            except Exception as e:
                logger.warning(f"Error using Random Forest for prediction: {str(e)}")
                # Fall back to distance-based matching
                matched_objects = set()
                matched_detections = set()
        else:
            # Fall back to simple distance-based matching
            distances = {}
            for obj_id, obj_centroid in zip(object_ids, object_centroids):
                for i, det_centroid in enumerate(detection_centroids):
                    distance = np.linalg.norm(obj_centroid - det_centroid)
                    if distance <= self.max_distance:
                        distances[(obj_id, i)] = distance

            # Sort by distance and match greedily
            matched_objects = set()
            matched_detections = set()

            for (obj_id, det_idx), distance in sorted(distances.items(), key=lambda x: x[1]):
                if obj_id not in matched_objects and det_idx not in matched_detections:
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
                        'trajectory': list(self.trajectories[obj_id])
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

        # Update Random Forest model with current objects if we have enough data
        if len(self.objects) >= 2 and len(set(self.objects.keys())) >= 2:
            object_centroids = np.array([self.objects[obj_id] for obj_id in self.objects.keys()])
            object_ids = np.array(list(self.objects.keys()))

            # Scale features
            self.scaler.fit(object_centroids)
            scaled_centroids = self.scaler.transform(object_centroids)

            # Fit Random Forest model
            try:
                self.rf.fit(scaled_centroids, object_ids)
                self.is_fitted = True
            except Exception as e:
                logger.warning(f"Error fitting Random Forest model: {str(e)}")

        return self.tracked_objects