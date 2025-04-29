"""
Base stabilizer class for all video stabilization algorithms.

This module provides a base class that all video stabilization algorithms should inherit from.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.analysis.metrics_collector import metrics_collector

logger = get_logger(__name__)


class BaseStabilizer(ABC):
    """
    Base class for all video stabilization algorithms.
    
    This class defines the interface that all video stabilization algorithms
    should implement.
    """
    
    def __init__(self, config_prefix: str = 'preprocessing'):
        """
        Initialize the stabilizer.
        
        Args:
            config_prefix: Prefix for configuration keys
        """
        self.config_prefix = config_prefix
        self.algorithm_name = self.get_name()
        self.algorithm_config = f'{config_prefix}.algorithms.{self.algorithm_name}'
        self.enabled = config.get(f'{config_prefix}.stabilization.enabled', True)
        self.smoothing_radius = config.get(f'{config_prefix}.stabilization.smoothing_radius', 15)
        self.prev_frame = None
        self.transforms = []
        self.frame_count = 0
        self.camera_name = None
        
        logger.info(f"Initializing {self.__class__.__name__} with "
                   f"enabled={self.enabled}, smoothing_radius={self.smoothing_radius}")
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the stabilizer.
        
        Returns:
            Name of the stabilizer
        """
        pass
    
    @abstractmethod
    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        Stabilize a frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Stabilized frame
        """
        pass
    
    def set_camera_name(self, camera_name: str) -> None:
        """
        Set the camera name for metrics collection.
        
        Args:
            camera_name: Name of the camera
        """
        self.camera_name = camera_name
    
    def _record_metrics(self, start_time: float, stability_score: float) -> None:
        """
        Record metrics for the stabilization.
        
        Args:
            start_time: Start time of the stabilization
            stability_score: Stability score (0-1, higher is better)
        """
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Record metrics
        metrics_collector.add_metric('preprocessing', 'processing_time', processing_time * 1000, self.camera_name)
        metrics_collector.add_metric('preprocessing', 'stability_score', stability_score, self.camera_name)
    
    def _calculate_stability_score(self, transform: np.ndarray) -> float:
        """
        Calculate a stability score based on the transformation.
        
        Args:
            transform: Transformation matrix
        
        Returns:
            Stability score (0-1, higher is better)
        """
        # Extract translation and rotation from transformation matrix
        dx = transform[0, 2]
        dy = transform[1, 2]
        angle = np.arctan2(transform[1, 0], transform[0, 0]) * 180 / np.pi
        
        # Calculate stability score
        translation_score = 1.0 - min(1.0, (abs(dx) + abs(dy)) / 100.0)
        rotation_score = 1.0 - min(1.0, abs(angle) / 10.0)
        
        # Combine scores
        stability_score = 0.7 * translation_score + 0.3 * rotation_score
        
        return stability_score
