"""
Clustering-based crowd detector implementation.

This module provides a crowd detector based on clustering algorithms.
"""

import time
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import Detection
from vigilance_system.alert.algorithms.base_crowd_detector import BaseCrowdDetector, CrowdEvent

logger = get_logger(__name__)


class ClusteringCrowdDetector(BaseCrowdDetector):
    """
    Clustering-based crowd detector implementation.
    
    This class implements a crowd detector that uses clustering algorithms
    to group people and detect crowds.
    """
    
    def __init__(self):
        """Initialize the clustering crowd detector."""
        super().__init__()
        
        # Get configuration
        self.algorithm = config.get(f'{self.algorithm_config}.algorithm', 'DBSCAN')
        self.eps = config.get(f'{self.algorithm_config}.eps', 100)
        self.min_samples = config.get(f'{self.algorithm_config}.min_samples', 3)
        self.cluster_threshold = config.get(f'{self.algorithm_config}.cluster_threshold', 2)
        
        logger.info(f"Initialized clustering crowd detector with "
                   f"algorithm={self.algorithm}, eps={self.eps}, "
                   f"min_samples={self.min_samples}, cluster_threshold={self.cluster_threshold}")
    
    def get_name(self) -> str:
        """
        Get the name of the crowd detector.
        
        Returns:
            Name of the crowd detector
        """
        return 'clustering'
    
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
        
        # Filter person detections
        person_detections = [d for d in detections if d.class_name.lower() == 'person']
        
        # Check if we have enough people
        if len(person_detections) < self.min_samples:
            # Not enough people for clustering
            # End all active events
            for event in self.crowd_events:
                if event.is_active:
                    event.end()
            
            # Record metrics
            self._record_metrics(start_time, [])
            
            return []
        
        # Extract person centers
        centers = []
        for detection in person_detections:
            center_x = (detection.bbox[0] + detection.bbox[2]) / 2
            center_y = (detection.bbox[1] + detection.bbox[3]) / 2
            centers.append([center_x, center_y])
        
        # Convert to numpy array
        centers = np.array(centers)
        
        # Perform clustering
        clusters = self._cluster_points(centers)
        
        # Process clusters
        crowd_clusters = []
        
        for cluster_id, cluster_points in clusters.items():
            if cluster_id == -1:
                # Noise points
                continue
            
            # Check if cluster has enough points
            if len(cluster_points) >= self.min_samples:
                # Calculate cluster properties
                cluster_center = np.mean(cluster_points, axis=0)
                min_x = np.min(cluster_points[:, 0])
                min_y = np.min(cluster_points[:, 1])
                max_x = np.max(cluster_points[:, 0])
                max_y = np.max(cluster_points[:, 1])
                
                # Add padding to bounding box
                padding = 20
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(frame.shape[1], max_x + padding)
                max_y = min(frame.shape[0], max_y + padding)
                
                # Create cluster info
                cluster_info = {
                    'id': cluster_id,
                    'center': (float(cluster_center[0]), float(cluster_center[1])),
                    'bbox': (float(min_x), float(min_y), float(max_x), float(max_y)),
                    'count': len(cluster_points),
                    'points': cluster_points
                }
                
                crowd_clusters.append(cluster_info)
        
        # Update crowd events
        self._update_crowd_events(crowd_clusters, frame_id)
        
        # Get active events
        active_events = [e for e in self.crowd_events if e.is_active]
        
        # Record metrics
        self._record_metrics(start_time, active_events)
        
        return active_events
    
    def _cluster_points(self, points: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Cluster points using the specified algorithm.
        
        Args:
            points: Array of points to cluster
        
        Returns:
            Dictionary mapping cluster IDs to arrays of points
        """
        if self.algorithm == 'DBSCAN':
            return self._dbscan_clustering(points)
        else:
            logger.warning(f"Unknown clustering algorithm: {self.algorithm}, falling back to DBSCAN")
            return self._dbscan_clustering(points)
    
    def _dbscan_clustering(self, points: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Cluster points using DBSCAN.
        
        Args:
            points: Array of points to cluster
        
        Returns:
            Dictionary mapping cluster IDs to arrays of points
        """
        try:
            from sklearn.cluster import DBSCAN
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
            
            # Get cluster labels
            labels = clustering.labels_
            
            # Group points by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(points[i])
            
            # Convert lists to numpy arrays
            for label in clusters:
                clusters[label] = np.array(clusters[label])
            
            return clusters
        
        except ImportError:
            logger.warning("scikit-learn not found, using simple distance-based clustering")
            return self._simple_clustering(points)
        
        except Exception as e:
            logger.error(f"Error in DBSCAN clustering: {str(e)}")
            return self._simple_clustering(points)
    
    def _simple_clustering(self, points: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Simple distance-based clustering.
        
        Args:
            points: Array of points to cluster
        
        Returns:
            Dictionary mapping cluster IDs to arrays of points
        """
        # Initialize clusters
        clusters = {}
        cluster_id = 0
        
        # Create a copy of points
        remaining_points = points.copy()
        
        while len(remaining_points) > 0:
            # Take the first point as a seed
            seed = remaining_points[0]
            cluster_points = [seed]
            remaining_points = remaining_points[1:]
            
            # Find all points within eps distance
            i = 0
            while i < len(remaining_points):
                point = remaining_points[i]
                
                # Calculate distance
                distance = np.sqrt(np.sum((seed - point) ** 2))
                
                if distance <= self.eps:
                    # Add to cluster
                    cluster_points.append(point)
                    remaining_points = np.delete(remaining_points, i, axis=0)
                else:
                    i += 1
            
            # Check if cluster has enough points
            if len(cluster_points) >= self.min_samples:
                clusters[cluster_id] = np.array(cluster_points)
                cluster_id += 1
            else:
                # Add to noise
                if -1 not in clusters:
                    clusters[-1] = np.array(cluster_points)
                else:
                    clusters[-1] = np.vstack([clusters[-1], cluster_points])
        
        return clusters
    
    def _update_crowd_events(self, clusters: List[Dict[str, Any]], frame_id: int) -> None:
        """
        Update crowd events based on clusters.
        
        Args:
            clusters: List of cluster information
            frame_id: ID of the current frame
        """
        # Mark all events as inactive initially
        for event in self.crowd_events:
            if event.is_active:
                event.is_active = False
        
        # Process each cluster
        for cluster in clusters:
            # Check if cluster count exceeds threshold
            if cluster['count'] < self.cluster_threshold:
                continue
            
            # Calculate confidence based on count
            confidence = min(1.0, (cluster['count'] - self.cluster_threshold + 1) / self.cluster_threshold)
            
            # Check if cluster matches an existing event
            matched_event = None
            
            for event in self.crowd_events:
                if not event.is_active:
                    # Calculate IoU between cluster and event
                    iou = self._calculate_iou(cluster['bbox'], event.bbox)
                    
                    if iou > 0.5:
                        # Match found
                        matched_event = event
                        break
            
            if matched_event is not None:
                # Update existing event
                matched_event.update(
                    location=cluster['center'],
                    bbox=cluster['bbox'],
                    frame_id=frame_id,
                    count=cluster['count'],
                    confidence=confidence
                )
                matched_event.is_active = True
            else:
                # Create new event
                event = CrowdEvent(
                    location=cluster['center'],
                    bbox=cluster['bbox'],
                    frame_id=frame_id,
                    count=cluster['count'],
                    confidence=confidence
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
    
    def draw_clusters(self, frame: np.ndarray, clusters: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw clusters on a frame.
        
        Args:
            frame: Input frame
            clusters: List of cluster information
        
        Returns:
            Frame with clusters drawn
        """
        result = frame.copy()
        
        # Generate random colors for clusters
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8).tolist()
        
        for i, cluster in enumerate(clusters):
            # Get cluster color
            color = colors[i % len(colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, cluster['bbox'])
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw center
            cx, cy = map(int, cluster['center'])
            cv2.circle(result, (cx, cy), 5, color, -1)
            
            # Draw label
            label = f"Cluster {cluster['id']}: {cluster['count']} people"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw points
            for point in cluster['points']:
                px, py = map(int, point)
                cv2.circle(result, (px, py), 3, color, -1)
        
        return result
