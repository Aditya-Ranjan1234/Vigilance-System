"""
Blob Counting Crowd Detection implementation.

This module provides a simple blob counting approach to detect crowds
without using deep learning.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import cv2

from vigilance_system.alert.algorithms.base_crowd_detector import BaseCrowdDetector
from vigilance_system.utils.logger import get_logger

logger = get_logger(__name__)


class BlobCountingCrowdDetector(BaseCrowdDetector):
    """
    Blob Counting crowd detector.
    
    Detects crowds by counting the number of blobs (people) in different regions
    of the frame.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the blob counting crowd detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "blob_counting"
        
        # Parameters
        self.crowd_threshold = config.get('crowd_threshold', 3)  # Minimum number of people to consider a crowd
        self.proximity_threshold = config.get('proximity_threshold', 100)  # Maximum distance between people to be considered in the same group
        self.min_group_size = config.get('min_group_size', 3)  # Minimum number of people in a group to be considered a crowd
        
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
        centroids = [obj['centroid'] for obj in persons]
        
        # Find groups of people using proximity
        groups = self._find_groups(centroids)
        
        # Filter groups by size
        crowd_groups = [group for group in groups if len(group) >= self.min_group_size]
        
        # Create crowd events
        crowd_events = []
        for i, group in enumerate(crowd_groups):
            # Calculate group centroid
            group_centroids = [centroids[idx] for idx in group]
            group_centroid = self._calculate_group_centroid(group_centroids)
            
            # Get people in this group
            group_people = [persons[idx] for idx in group]
            
            crowd_events.append({
                'crowd_id': i,
                'people_count': len(group),
                'centroid': group_centroid,
                'people': group_people
            })
        
        return crowd_events
    
    def _find_groups(self, centroids: List[Tuple[int, int]]) -> List[List[int]]:
        """
        Find groups of people based on proximity.
        
        Args:
            centroids: List of person centroids
            
        Returns:
            List of groups, where each group is a list of indices into the centroids list
        """
        n = len(centroids)
        if n == 0:
            return []
        
        # Calculate distance matrix
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                distance = self._calculate_distance(centroids[i], centroids[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        # Find connected components (groups)
        visited = [False] * n
        groups = []
        
        for i in range(n):
            if visited[i]:
                continue
                
            # Start a new group
            group = [i]
            visited[i] = True
            
            # Find all connected persons
            self._dfs(i, distance_matrix, visited, group)
            
            groups.append(group)
        
        return groups
    
    def _dfs(self, i: int, distance_matrix: np.ndarray, visited: List[bool], group: List[int]) -> None:
        """
        Depth-first search to find connected components.
        
        Args:
            i: Current person index
            distance_matrix: Matrix of distances between persons
            visited: List of visited flags
            group: Current group being built
        """
        n = len(visited)
        
        for j in range(n):
            if not visited[j] and distance_matrix[i, j] <= self.proximity_threshold:
                group.append(j)
                visited[j] = True
                self._dfs(j, distance_matrix, visited, group)
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _calculate_group_centroid(self, centroids: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate the centroid of a group of points."""
        if not centroids:
            return (0, 0)
            
        x_sum = sum(c[0] for c in centroids)
        y_sum = sum(c[1] for c in centroids)
        
        return (x_sum // len(centroids), y_sum // len(centroids))
