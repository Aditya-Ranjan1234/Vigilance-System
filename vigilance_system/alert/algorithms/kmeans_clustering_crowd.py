"""
K-Means Clustering Crowd Detection implementation.

This module provides a K-means clustering approach to detect crowds
without using deep learning.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans

from vigilance_system.alert.algorithms.base_crowd_detector import BaseCrowdDetector
from vigilance_system.utils.logger import get_logger

logger = get_logger(__name__)


class KMeansClusteringCrowdDetector(BaseCrowdDetector):
    """
    K-Means Clustering crowd detector.
    
    Detects crowds by clustering person positions using K-means
    and analyzing the resulting clusters.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the K-means clustering crowd detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "kmeans_clustering"
        
        # Parameters
        self.crowd_threshold = config.get('crowd_threshold', 3)  # Minimum number of people to consider a crowd
        self.max_clusters = config.get('max_clusters', 5)  # Maximum number of clusters to consider
        self.min_cluster_size = config.get('min_cluster_size', 3)  # Minimum number of people in a cluster to be considered a crowd
        self.max_cluster_radius = config.get('max_cluster_radius', 100)  # Maximum radius of a cluster to be considered a crowd
        
        logger.info(f"Initialized {self.name} crowd detector")
    
    def detect(self, tracked_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect crowds in tracked objects.
        
        Args:
            tracked_objects: List of tracked objects from the tracker
            
        Returns:
            List of crowd events
        """
        # Filter for person objects
        persons = [obj for obj in tracked_objects if obj.get('label') == 'person']
        
        # If not enough people, no crowd
        if len(persons) < self.crowd_threshold:
            return []
        
        # Get centroids of all persons
        centroids = np.array([obj['centroid'] for obj in persons])
        
        # Determine optimal number of clusters
        n_clusters = min(self.max_clusters, len(persons) // 2)
        n_clusters = max(1, n_clusters)  # At least 1 cluster
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(centroids)
        
        # Get cluster labels and centers
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        # Analyze clusters
        crowd_events = []
        for i in range(n_clusters):
            # Get indices of persons in this cluster
            cluster_indices = np.where(labels == i)[0]
            
            # If cluster is large enough, consider it a crowd
            if len(cluster_indices) >= self.min_cluster_size:
                # Calculate cluster radius
                cluster_points = centroids[cluster_indices]
                center = centers[i]
                
                # Calculate distances from center
                distances = np.sqrt(np.sum((cluster_points - center) ** 2, axis=1))
                max_distance = np.max(distances)
                
                # If cluster is compact enough, consider it a crowd
                if max_distance <= self.max_cluster_radius:
                    # Get people in this cluster
                    cluster_people = [persons[idx] for idx in cluster_indices]
                    
                    crowd_events.append({
                        'crowd_id': i,
                        'people_count': len(cluster_indices),
                        'centroid': (int(center[0]), int(center[1])),
                        'people': cluster_people
                    })
        
        return crowd_events
