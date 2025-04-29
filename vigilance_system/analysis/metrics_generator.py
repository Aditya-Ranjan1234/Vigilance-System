"""
Metrics generator for the Vigilance System.

This module generates sample metrics for testing and demonstration purposes.
"""

import time
import random
import threading
from typing import Dict, Any, Optional

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.analysis.metrics_collector import metrics_collector

logger = get_logger(__name__)


class MetricsGenerator:
    """
    Generates sample metrics for testing and demonstration purposes.

    This class is used to generate realistic metrics for different algorithms
    when real metrics are not available.
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern to ensure only one metrics generator exists."""
        if cls._instance is None:
            cls._instance = super(MetricsGenerator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the metrics generator."""
        if self._initialized:
            return

        self._running = False
        self._thread = None
        self._stop_event = threading.Event()
        self._initialized = True

        # Define base metrics for each algorithm
        self._base_metrics = {
            'detection': {
                'background_subtraction': {'fps': 45.0, 'precision': 0.70, 'recall': 0.65, 'map': 0.60},
                'mog2': {'fps': 40.0, 'precision': 0.75, 'recall': 0.70, 'map': 0.65},
                'knn': {'fps': 38.0, 'precision': 0.72, 'recall': 0.68, 'map': 0.62},
                'svm_classifier': {'fps': 35.0, 'precision': 0.78, 'recall': 0.72, 'map': 0.68}
            },
            'tracking': {
                'klt_tracker': {'fps': 42.0, 'id_switches': 8.0, 'mota': 0.65, 'motp': 0.68},
                'kalman_filter': {'fps': 38.0, 'id_switches': 6.0, 'mota': 0.70, 'motp': 0.72},
                'optical_flow': {'fps': 35.0, 'id_switches': 7.0, 'mota': 0.68, 'motp': 0.70}
            },
            'loitering': {
                'rule_based': {'true_positives': 7.0, 'false_positives': 4.0, 'false_negatives': 3.0, 'precision': 0.65, 'recall': 0.70},
                'timer_threshold': {'true_positives': 8.0, 'false_positives': 3.5, 'false_negatives': 2.5, 'precision': 0.70, 'recall': 0.75},
                'decision_tree': {'true_positives': 7.5, 'false_positives': 3.0, 'false_negatives': 2.8, 'precision': 0.72, 'recall': 0.73}
            },
            'crowd': {
                'blob_counting': {'mae': 3.0, 'mse': 10.0, 'accuracy': 0.68, 'precision': 0.65, 'recall': 0.62, 'event_count': 2.5},
                'contour_counting': {'mae': 2.8, 'mse': 9.0, 'accuracy': 0.70, 'precision': 0.68, 'recall': 0.65, 'event_count': 3.0},
                'kmeans_clustering': {'mae': 2.5, 'mse': 8.0, 'accuracy': 0.72, 'precision': 0.70, 'recall': 0.68, 'event_count': 3.2}
            },
            'preprocessing': {
                'feature_matching': {'processing_time': 10.0, 'stability_score': 0.72},
                'orb': {'processing_time': 8.0, 'stability_score': 0.70},
                'sift': {'processing_time': 12.0, 'stability_score': 0.75},
                'affine_transform': {'processing_time': 9.0, 'stability_score': 0.78}
            }
        }

        # Current algorithms
        self._current_algorithms = {
            'detection': config.get('detection.algorithm', 'background_subtraction'),
            'tracking': config.get('tracking.algorithm', 'klt_tracker'),
            'loitering': config.get('alerts.loitering.algorithm', 'rule_based'),
            'crowd': config.get('alerts.crowd.algorithm', 'blob_counting'),
            'preprocessing': config.get('preprocessing.algorithm', 'feature_matching')
        }

        logger.info("Metrics generator initialized")

    def start(self):
        """Start generating metrics."""
        if self._running:
            logger.warning("Metrics generator is already running")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._generate_metrics_loop)
        self._thread.daemon = True
        self._thread.start()

        logger.info("Started metrics generator")

    def stop(self):
        """Stop generating metrics."""
        if not self._running:
            logger.warning("Metrics generator is not running")
            return

        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

        logger.info("Stopped metrics generator")

    def update_current_algorithm(self, component: str, algorithm: str):
        """
        Update the current algorithm for a component.

        Args:
            component: The component name (e.g., 'detection', 'tracking')
            algorithm: The algorithm name
        """
        if component in self._current_algorithms:
            self._current_algorithms[component] = algorithm
            logger.info(f"Updated current algorithm for {component} to {algorithm}")

    def _generate_metrics_loop(self):
        """Generate metrics in a loop."""
        while self._running and not self._stop_event.is_set():
            # Update current algorithms from config
            self._update_current_algorithms()

            # Generate metrics for each component and algorithm
            self._generate_detection_metrics()
            self._generate_tracking_metrics()
            self._generate_loitering_metrics()
            self._generate_crowd_metrics()
            self._generate_preprocessing_metrics()

            # Sleep for a short time
            time.sleep(1.0)

    def _update_current_algorithms(self):
        """Update current algorithms from config."""
        self._current_algorithms = {
            'detection': config.get('detection.algorithm', 'background_subtraction'),
            'tracking': config.get('tracking.algorithm', 'klt_tracker'),
            'loitering': config.get('alerts.loitering.algorithm', 'rule_based'),
            'crowd': config.get('alerts.crowd.algorithm', 'blob_counting'),
            'preprocessing': config.get('preprocessing.algorithm', 'feature_matching')
        }

    def _generate_detection_metrics(self):
        """Generate detection metrics."""
        component = 'detection'
        algorithm = self._current_algorithms[component]

        # Get base metrics for the current algorithm
        base_metrics = self._base_metrics[component].get(algorithm, {})

        # Generate metrics with some random variation
        for metric_name, base_value in base_metrics.items():
            # Add some random variation
            variation = base_value * 0.1  # 10% variation
            value = base_value + random.uniform(-variation, variation)

            # Add metric for each camera
            for camera_name in ['Camera 1', 'Camera 2', 'Camera 3']:
                # Add some camera-specific variation
                camera_variation = base_value * 0.05  # 5% variation
                camera_value = value + random.uniform(-camera_variation, camera_variation)

                # Add the metric
                metrics_collector.add_metric(component, metric_name, camera_value, camera_name)

            # Add overall metric (without camera name)
            metrics_collector.add_metric(component, metric_name, value)

    def _generate_tracking_metrics(self):
        """Generate tracking metrics."""
        component = 'tracking'
        algorithm = self._current_algorithms[component]

        # Get base metrics for the current algorithm
        base_metrics = self._base_metrics[component].get(algorithm, {})

        # Generate metrics with some random variation
        for metric_name, base_value in base_metrics.items():
            # Add some random variation
            variation = base_value * 0.1  # 10% variation
            value = base_value + random.uniform(-variation, variation)

            # Add metric for each camera
            for camera_name in ['Camera 1', 'Camera 2', 'Camera 3']:
                # Add some camera-specific variation
                camera_variation = base_value * 0.05  # 5% variation
                camera_value = value + random.uniform(-camera_variation, camera_variation)

                # Add the metric
                metrics_collector.add_metric(component, metric_name, camera_value, camera_name)

            # Add overall metric (without camera name)
            metrics_collector.add_metric(component, metric_name, value)

    def _generate_loitering_metrics(self):
        """Generate loitering metrics."""
        component = 'loitering'
        algorithm = self._current_algorithms[component]

        # Get base metrics for the current algorithm
        base_metrics = self._base_metrics[component].get(algorithm, {})

        # Generate metrics with some random variation
        for metric_name, base_value in base_metrics.items():
            # Add some random variation
            variation = base_value * 0.1  # 10% variation
            value = base_value + random.uniform(-variation, variation)

            # Add metric for each camera
            for camera_name in ['Camera 1', 'Camera 2', 'Camera 3']:
                # Add some camera-specific variation
                camera_variation = base_value * 0.05  # 5% variation
                camera_value = value + random.uniform(-camera_variation, camera_variation)

                # Add the metric
                metrics_collector.add_metric(component, metric_name, camera_value, camera_name)

            # Add overall metric (without camera name)
            metrics_collector.add_metric(component, metric_name, value)

        # Add event count metric
        event_count = random.randint(0, 5)
        metrics_collector.add_metric(component, 'event_count', event_count)

    def _generate_crowd_metrics(self):
        """Generate crowd metrics."""
        component = 'crowd'
        algorithm = self._current_algorithms[component]

        # Get base metrics for the current algorithm
        base_metrics = self._base_metrics[component].get(algorithm, {})

        # Generate metrics with some random variation
        for metric_name, base_value in base_metrics.items():
            # Add some random variation
            variation = base_value * 0.1  # 10% variation
            value = base_value + random.uniform(-variation, variation)

            # Add metric for each camera
            for camera_name in ['Camera 1', 'Camera 2', 'Camera 3']:
                # Add some camera-specific variation
                camera_variation = base_value * 0.05  # 5% variation
                camera_value = value + random.uniform(-camera_variation, camera_variation)

                # Add the metric
                metrics_collector.add_metric(component, metric_name, camera_value, camera_name)

            # Add overall metric (without camera name)
            metrics_collector.add_metric(component, metric_name, value)

    def _generate_preprocessing_metrics(self):
        """Generate preprocessing metrics."""
        component = 'preprocessing'
        algorithm = self._current_algorithms[component]

        # Get base metrics for the current algorithm
        base_metrics = self._base_metrics[component].get(algorithm, {})

        # Generate metrics with some random variation
        for metric_name, base_value in base_metrics.items():
            # Add some random variation
            variation = base_value * 0.1  # 10% variation
            value = base_value + random.uniform(-variation, variation)

            # Add metric for each camera
            for camera_name in ['Camera 1', 'Camera 2', 'Camera 3']:
                # Add some camera-specific variation
                camera_variation = base_value * 0.05  # 5% variation
                camera_value = value + random.uniform(-camera_variation, camera_variation)

                # Add the metric
                metrics_collector.add_metric(component, metric_name, camera_value, camera_name)

            # Add overall metric (without camera name)
            metrics_collector.add_metric(component, metric_name, value)


# Create a global instance
metrics_generator = MetricsGenerator()
