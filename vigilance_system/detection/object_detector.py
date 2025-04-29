"""
Object detector module for detecting objects in video frames.

This module provides functionality to detect objects in video frames using
various deep learning models.
"""

import time
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional

from vigilance_system.utils.logger import get_logger
from vigilance_system.utils.config import config
from vigilance_system.detection.model_loader import model_loader
from vigilance_system.analysis.metrics_collector import metrics_collector

# Initialize logger
logger = get_logger(__name__)


class Detection:
    """
    Represents a single object detection.

    Contains information about the detected object, including its class,
    bounding box, and confidence score.
    """

    def __init__(self, class_id: int, class_name: str, confidence: float,
                 bbox: Tuple[float, float, float, float], frame_id: int = 0):
        """
        Initialize a detection.

        Args:
            class_id: Class ID of the detected object
            class_name: Class name of the detected object
            confidence: Confidence score of the detection
            bbox: Bounding box coordinates (x1, y1, x2, y2) in pixels
            frame_id: ID of the frame where the detection occurred
        """
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.frame_id = frame_id
        self.timestamp = time.time()

        # Calculate center point and dimensions
        self.center_x = (bbox[0] + bbox[2]) / 2
        self.center_y = (bbox[1] + bbox[3]) / 2
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.area = self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert detection to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the detection
        """
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': (self.center_x, self.center_y),
            'dimensions': (self.width, self.height),
            'area': self.area,
            'frame_id': self.frame_id,
            'timestamp': self.timestamp
        }

    def __str__(self) -> str:
        """
        String representation of the detection.

        Returns:
            str: String representation
        """
        return (f"Detection(class={self.class_name}, confidence={self.confidence:.2f}, "
                f"bbox=({self.bbox[0]:.1f}, {self.bbox[1]:.1f}, {self.bbox[2]:.1f}, {self.bbox[3]:.1f}))")


class ObjectDetector:
    """
    Detects objects in video frames.

    Uses deep learning models to detect objects in frames and provides
    methods to filter and process detections.
    """

    def __init__(self, model_name: str = None, confidence_threshold: float = 0.5,
                 classes_of_interest: Optional[List[int]] = None):
        """
        Initialize the object detector.

        Args:
            model_name: Name of the model to use (if None, uses config)
            confidence_threshold: Minimum confidence score for detections
            classes_of_interest: List of class IDs to detect (if None, detects all)
        """
        # Load configuration
        self.algorithm = config.get('detection.algorithm', 'yolov5')
        self.model_name = model_name or config.get('detection.model', 'yolov5s')
        self.confidence_threshold = confidence_threshold or config.get('detection.confidence_threshold', 0.5)
        self.classes_of_interest = classes_of_interest or config.get('detection.classes_of_interest', None)

        # Initialize detector
        self.detector = None
        self.use_algorithm_detectors = config.get('detection.use_algorithm_detectors', False)

        if self.use_algorithm_detectors:
            # Use the new algorithm-based detectors
            self._initialize_algorithm_detector()
        else:
            # Use the legacy model loader
            self.model = model_loader.load_model(self.model_name)

        # Initialize counters
        self.frame_count = 0
        self.detection_count = 0
        self.camera_name = None

        logger.info(f"Initialized object detector with algorithm: {self.algorithm}, model: {self.model_name}, "
                   f"confidence threshold: {self.confidence_threshold}")

    def _initialize_algorithm_detector(self):
        """Initialize the appropriate algorithm detector."""
        try:
            # Non-deep learning algorithms
            if self.algorithm == 'background_subtraction':
                from vigilance_system.detection.algorithms.background_subtraction_detector import BackgroundSubtractionDetector
                self.detector = BackgroundSubtractionDetector()

            elif self.algorithm == 'mog2':
                from vigilance_system.detection.algorithms.mog2_detector import MOG2Detector
                self.detector = MOG2Detector()

            elif self.algorithm == 'knn':
                from vigilance_system.detection.algorithms.knn_detector import KNNDetector
                self.detector = KNNDetector()

            elif self.algorithm == 'svm_classifier':
                from vigilance_system.detection.algorithms.svm_classifier_detector import SVMClassifierDetector
                self.detector = SVMClassifierDetector()

            # Legacy deep learning algorithms (disabled)
            elif self.algorithm == 'yolov5':
                logger.warning(f"Deep learning algorithm '{self.algorithm}' is disabled. Using background_subtraction instead.")
                from vigilance_system.detection.algorithms.background_subtraction_detector import BackgroundSubtractionDetector
                self.detector = BackgroundSubtractionDetector()

            elif self.algorithm == 'yolov8':
                logger.warning(f"Deep learning algorithm '{self.algorithm}' is disabled. Using background_subtraction instead.")
                from vigilance_system.detection.algorithms.background_subtraction_detector import BackgroundSubtractionDetector
                self.detector = BackgroundSubtractionDetector()

            elif self.algorithm == 'ssd':
                logger.warning(f"Deep learning algorithm '{self.algorithm}' is disabled. Using background_subtraction instead.")
                from vigilance_system.detection.algorithms.background_subtraction_detector import BackgroundSubtractionDetector
                self.detector = BackgroundSubtractionDetector()

            elif self.algorithm == 'faster_rcnn':
                logger.warning(f"Deep learning algorithm '{self.algorithm}' is disabled. Using background_subtraction instead.")
                from vigilance_system.detection.algorithms.background_subtraction_detector import BackgroundSubtractionDetector
                self.detector = BackgroundSubtractionDetector()

            else:
                logger.warning(f"Unknown algorithm: {self.algorithm}, using background_subtraction")
                from vigilance_system.detection.algorithms.background_subtraction_detector import BackgroundSubtractionDetector
                self.detector = BackgroundSubtractionDetector()

        except Exception as e:
            logger.error(f"Error initializing algorithm detector: {str(e)}")
            logger.info("Falling back to legacy model loader")
            self.use_algorithm_detectors = False
            self.model = model_loader.load_model(self.model_name)

    def set_camera_name(self, camera_name: str) -> None:
        """
        Set the camera name for metrics collection.

        Args:
            camera_name: Name of the camera
        """
        self.camera_name = camera_name
        if self.use_algorithm_detectors and self.detector:
            self.detector.set_camera_name(camera_name)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input frame to process

        Returns:
            List[Detection]: List of detections
        """
        if frame is None:
            logger.warning("Received None frame for detection")
            return []

        try:
            # Increment frame counter
            self.frame_count += 1

            # Record start time for metrics
            start_time = time.time()

            if self.use_algorithm_detectors and self.detector:
                # Use the algorithm detector
                detections = self.detector.detect(frame)

                # Convert to our Detection class if needed
                if detections and isinstance(detections[0], Detection):
                    # Already using our Detection class
                    pass
                else:
                    # Convert to our Detection class
                    converted_detections = []
                    for d in detections:
                        # Check if d is a dictionary or an object
                        if isinstance(d, dict):
                            # Dictionary format
                            class_id = d.get('class_id', 0)
                            class_name = d.get('label', 'person')
                            confidence = d.get('confidence', 0.0)
                            bbox = d.get('bbox', (0, 0, 0, 0))
                        else:
                            # Object format (from base_detector.Detection)
                            class_id = getattr(d, 'class_id', 0)
                            class_name = getattr(d, 'class_name', 'person')
                            confidence = getattr(d, 'confidence', 0.0)
                            bbox = getattr(d, 'bbox', (0, 0, 0, 0))

                        detection = Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox,
                            frame_id=self.frame_count
                        )
                        converted_detections.append(detection)

                    detections = converted_detections
            else:
                # Use the legacy model
                # Run inference
                results = self.model(frame)

                # Process results
                detections = self._process_results(results, self.frame_count)

            # Update detection counter
            self.detection_count += len(detections)

            # Record metrics
            if not self.use_algorithm_detectors:
                # Only record metrics here if not using algorithm detectors
                # (algorithm detectors record their own metrics)
                end_time = time.time()
                processing_time = end_time - start_time
                fps = 1.0 / processing_time if processing_time > 0 else 0

                metrics_collector.add_metric('detection', 'fps', fps, self.camera_name)
                metrics_collector.add_metric('detection', 'processing_time', processing_time * 1000, self.camera_name)
                metrics_collector.add_metric('detection', 'detection_count', len(detections), self.camera_name)

            return detections

        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return []

    def _process_results(self, results: Any, frame_id: int) -> List[Detection]:
        """
        Process model results into Detection objects.

        Args:
            results: Model inference results
            frame_id: ID of the current frame

        Returns:
            List[Detection]: List of processed detections
        """
        detections = []

        # Extract detections from YOLOv5 results
        if self.model_name.startswith('yolov5'):
            # Convert to pandas DataFrame
            result_df = results.pandas().xyxy[0]

            for _, row in result_df.iterrows():
                confidence = float(row['confidence'])

                # Filter by confidence threshold
                if confidence < self.confidence_threshold:
                    continue

                class_id = int(row['class'])

                # Filter by classes of interest
                if self.classes_of_interest is not None and class_id not in self.classes_of_interest:
                    continue

                class_name = row['name']
                bbox = (float(row['xmin']), float(row['ymin']),
                        float(row['xmax']), float(row['ymax']))

                detection = Detection(class_id, class_name, confidence, bbox, frame_id)
                detections.append(detection)

        return detections

    def filter_detections(self, detections: List[Detection],
                          min_confidence: float = None,
                          classes: List[int] = None,
                          min_size: float = None) -> List[Detection]:
        """
        Filter detections based on various criteria.

        Args:
            detections: List of detections to filter
            min_confidence: Minimum confidence score
            classes: List of class IDs to include
            min_size: Minimum detection size (area) in pixels

        Returns:
            List[Detection]: Filtered list of detections
        """
        filtered = detections

        # Filter by confidence
        if min_confidence is not None:
            filtered = [d for d in filtered if d.confidence >= min_confidence]

        # Filter by class
        if classes is not None:
            filtered = [d for d in filtered if d.class_id in classes]

        # Filter by size
        if min_size is not None:
            filtered = [d for d in filtered if d.area >= min_size]

        return filtered

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detection statistics.

        Returns:
            Dict[str, Any]: Dictionary with detection statistics
        """
        return {
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'classes_of_interest': self.classes_of_interest,
            'frames_processed': self.frame_count,
            'total_detections': self.detection_count,
            'average_detections_per_frame': self.detection_count / max(1, self.frame_count)
        }

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detection bounding boxes on a frame.

        Args:
            frame: Input frame
            detections: List of detections to draw

        Returns:
            np.ndarray: Frame with drawn detections
        """
        import cv2

        # Use algorithm detector if available
        if self.use_algorithm_detectors and self.detector:
            # Convert our Detection objects to algorithm detector Detection objects if needed
            if hasattr(self.detector, 'draw_detections'):
                # If detections are our Detection class, convert to algorithm detector Detection class
                from vigilance_system.detection.algorithms.base_detector import Detection as AlgoDetection

                algo_detections = []
                for d in detections:
                    algo_detection = AlgoDetection(
                        bbox=d.bbox,
                        class_id=d.class_id,
                        class_name=d.class_name,
                        confidence=d.confidence,
                        tracking_id=getattr(d, 'tracking_id', None)
                    )
                    algo_detections.append(algo_detection)

                return self.detector.draw_detections(frame, algo_detections)

        # Fall back to default implementation
        result = frame.copy()

        for detection in detections:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, detection.bbox)

            # Determine color based on confidence (green to red)
            color = (0, int(255 * detection.confidence), int(255 * (1 - detection.confidence)))

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            if hasattr(detection, 'tracking_id') and detection.tracking_id is not None:
                label = f"ID: {detection.tracking_id} | {label}"

            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)

            # Draw label text
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return result


def create_detector_from_config() -> ObjectDetector:
    """
    Create an object detector with settings from the configuration.

    Returns:
        ObjectDetector: Configured object detector
    """
    model_name = config.get('detection.model', 'yolov5s')
    confidence_threshold = config.get('detection.confidence_threshold', 0.5)
    classes_of_interest = config.get('detection.classes_of_interest', None)

    # Enable algorithm detectors by default if the algorithm is specified
    try:
        if not config.get('detection.use_algorithm_detectors', False):
            config.set('detection.use_algorithm_detectors', True)
    except AttributeError:
        # If config.set is not available, just continue with the default value
        logger.warning("Could not set 'detection.use_algorithm_detectors' in config (method not available)")

    return ObjectDetector(model_name, confidence_threshold, classes_of_interest)
