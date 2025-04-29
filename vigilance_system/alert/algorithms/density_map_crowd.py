"""
Density map-based crowd detector implementation.

This module provides a crowd detector based on density maps.
"""

import time
import os
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import Detection
from vigilance_system.alert.algorithms.base_crowd_detector import BaseCrowdDetector, CrowdEvent

logger = get_logger(__name__)


class DensityMapCrowdDetector(BaseCrowdDetector):
    """
    Density map-based crowd detector implementation.
    
    This class implements a crowd detector that uses density maps to estimate
    the number of people in an area and detect crowds.
    """
    
    def __init__(self):
        """Initialize the density map crowd detector."""
        super().__init__()
        
        # Get configuration
        self.model_name = config.get(f'{self.algorithm_config}.model', 'CSRNet')
        self.density_threshold = config.get(f'{self.algorithm_config}.density_threshold', 0.7)
        self.model_path = config.get(f'{self.algorithm_config}.model_path', 'models/crowd_density.h5')
        
        # Initialize model
        self.model = None
        self._load_model()
        
        logger.info(f"Initialized density map crowd detector with "
                   f"model={self.model_name}, density_threshold={self.density_threshold}")
    
    def get_name(self) -> str:
        """
        Get the name of the crowd detector.
        
        Returns:
            Name of the crowd detector
        """
        return 'density_map'
    
    def _load_model(self) -> None:
        """Load the density map model."""
        try:
            # Try to import TensorFlow
            import tensorflow as tf
            
            # Check if model file exists
            if os.path.exists(self.model_path):
                # Load model
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded density map model from {self.model_path}")
            else:
                # Create a simple model for demonstration
                logger.warning(f"Model file not found: {self.model_path}, using detection-based fallback")
                self.model = None
        
        except ImportError:
            logger.warning("TensorFlow not found, using detection-based fallback")
            self.model = None
        
        except Exception as e:
            logger.error(f"Error loading density map model: {str(e)}")
            self.model = None
    
    def update(self, detections: List[Detection], frame: np.ndarray, frame_id: int) -> List[CrowdEvent]:
        """
        Update the crowd detector with new detections.
        
        Args:
            detections: List of detections
            frame: Current frame
            frame_id: ID of the current frame
        
        Returns:
            List of active crowd events
        """
        # Record start time for metrics
        start_time = time.time()
        
        # Increment frame count
        self.frame_count += 1
        
        # Check if we have a model
        if self.model is None:
            # Fall back to detection-based crowd detection
            return self._detection_based_update(detections, frame, frame_id, start_time)
        
        try:
            # Generate density map
            density_map = self._generate_density_map(frame)
            
            # Find crowd regions
            crowd_regions = self._find_crowd_regions(density_map, frame.shape)
            
            # Update crowd events
            self._update_crowd_events(crowd_regions, frame_id)
            
            # Get active events
            active_events = [e for e in self.crowd_events if e.is_active]
            
            # Record metrics
            self._record_metrics(start_time, active_events)
            
            return active_events
        
        except Exception as e:
            logger.error(f"Error in density map crowd detection: {str(e)}")
            return self._detection_based_update(detections, frame, frame_id, start_time)
    
    def _detection_based_update(self, detections: List[Detection], frame: np.ndarray, 
                              frame_id: int, start_time: float) -> List[CrowdEvent]:
        """
        Fall back to detection-based crowd detection.
        
        Args:
            detections: List of detections
            frame: Current frame
            frame_id: ID of the current frame
            start_time: Start time for metrics
        
        Returns:
            List of active crowd events
        """
        # Filter person detections
        person_detections = [d for d in detections if d.class_name.lower() == 'person']
        
        # Create a simple density map from detections
        density_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        
        for detection in person_detections:
            # Get bounding box
            x1, y1, x2, y2 = map(int, detection.bbox)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)
            
            # Skip if bounding box is invalid
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Add to density map
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Create a Gaussian blob
            sigma = 20
            for i in range(max(0, center_y - 3 * sigma), min(frame.shape[0], center_y + 3 * sigma)):
                for j in range(max(0, center_x - 3 * sigma), min(frame.shape[1], center_x + 3 * sigma)):
                    density_map[i, j] += np.exp(-((i - center_y) ** 2 + (j - center_x) ** 2) / (2 * sigma ** 2))
        
        # Normalize density map
        if np.max(density_map) > 0:
            density_map = density_map / np.max(density_map)
        
        # Find crowd regions
        crowd_regions = self._find_crowd_regions(density_map, frame.shape)
        
        # Update crowd events
        self._update_crowd_events(crowd_regions, frame_id)
        
        # Get active events
        active_events = [e for e in self.crowd_events if e.is_active]
        
        # Record metrics
        self._record_metrics(start_time, active_events)
        
        return active_events
    
    def _generate_density_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate a density map from a frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Density map
        """
        try:
            import tensorflow as tf
            
            # Preprocess frame
            input_frame = cv2.resize(frame, (640, 480))
            input_frame = input_frame.astype(np.float32) / 255.0
            input_frame = np.expand_dims(input_frame, axis=0)
            
            # Generate density map
            density_map = self.model.predict(input_frame, verbose=0)[0, :, :, 0]
            
            # Resize to original frame size
            density_map = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))
            
            # Normalize
            if np.max(density_map) > 0:
                density_map = density_map / np.max(density_map)
            
            return density_map
        
        except Exception as e:
            logger.error(f"Error generating density map: {str(e)}")
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    
    def _find_crowd_regions(self, density_map: np.ndarray, frame_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """
        Find crowd regions in a density map.
        
        Args:
            density_map: Density map
            frame_shape: Shape of the original frame
        
        Returns:
            List of crowd regions
        """
        # Threshold density map
        binary_map = (density_map > self.density_threshold).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
        
        # Create crowd regions
        crowd_regions = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            # Get region properties
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            centroid = centroids[i]
            
            # Skip small regions
            if area < 100:
                continue
            
            # Calculate average density
            region_mask = (labels == i).astype(np.uint8)
            avg_density = np.mean(density_map[region_mask > 0])
            
            # Estimate count based on density
            count = int(area * avg_density * 0.01) + 1
            
            # Create region
            region = {
                'bbox': (float(x), float(y), float(x + width), float(y + height)),
                'center': (float(centroid[0]), float(centroid[1])),
                'area': float(area),
                'avg_density': float(avg_density),
                'count': count
            }
            
            crowd_regions.append(region)
        
        return crowd_regions
    
    def _update_crowd_events(self, crowd_regions: List[Dict[str, Any]], frame_id: int) -> None:
        """
        Update crowd events based on crowd regions.
        
        Args:
            crowd_regions: List of crowd regions
            frame_id: ID of the current frame
        """
        # Mark all events as inactive initially
        for event in self.crowd_events:
            if event.is_active:
                event.is_active = False
        
        # Process each crowd region
        for region in crowd_regions:
            # Check if region matches an existing event
            matched_event = None
            
            for event in self.crowd_events:
                if not event.is_active:
                    # Calculate IoU between region and event
                    iou = self._calculate_iou(region['bbox'], event.bbox)
                    
                    if iou > 0.5:
                        # Match found
                        matched_event = event
                        break
            
            if matched_event is not None:
                # Update existing event
                matched_event.update(
                    location=region['center'],
                    bbox=region['bbox'],
                    frame_id=frame_id,
                    count=region['count'],
                    confidence=region['avg_density']
                )
                matched_event.is_active = True
            else:
                # Create new event
                event = CrowdEvent(
                    location=region['center'],
                    bbox=region['bbox'],
                    frame_id=frame_id,
                    count=region['count'],
                    confidence=region['avg_density']
                )
                event.id = self.next_event_id
                self.next_event_id += 1
                
                # Add to events list
                self.crowd_events.append(event)
                event.is_active = True
        
        # End events that are still marked as inactive
        for event in self.crowd_events:
            if not event.is_active:
                event.end()
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
        
        Returns:
            IoU value
        """
        # Calculate intersection area
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def draw_density_map(self, frame: np.ndarray, density_map: np.ndarray) -> np.ndarray:
        """
        Draw a density map on a frame.
        
        Args:
            frame: Input frame
            density_map: Density map
        
        Returns:
            Frame with density map drawn
        """
        # Create colored density map
        colored_map = cv2.applyColorMap((density_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original frame
        alpha = 0.5
        result = cv2.addWeighted(frame, 1 - alpha, colored_map, alpha, 0)
        
        return result
