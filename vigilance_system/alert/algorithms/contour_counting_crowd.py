"""
Contour Counting Crowd Detection implementation.

This module provides a contour-based approach to detect crowds
without using deep learning.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import cv2

from vigilance_system.alert.algorithms.base_crowd_detector import BaseCrowdDetector
from vigilance_system.utils.logger import get_logger

logger = get_logger(__name__)


class ContourCountingCrowdDetector(BaseCrowdDetector):
    """
    Contour Counting crowd detector.
    
    Detects crowds by creating a density map from person positions
    and analyzing contours in the density map.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the contour counting crowd detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "contour_counting"
        
        # Parameters
        self.crowd_threshold = config.get('crowd_threshold', 3)  # Minimum number of people to consider a crowd
        self.density_threshold = config.get('density_threshold', 0.5)  # Threshold for density map
        self.min_contour_area = config.get('min_contour_area', 1000)  # Minimum contour area to consider a crowd
        self.kernel_size = config.get('kernel_size', 50)  # Size of Gaussian kernel for density map
        
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
        
        # Get centroids and bounding boxes of all persons
        centroids = [obj['centroid'] for obj in persons]
        bboxes = [obj['bbox'] for obj in persons]
        
        # Create density map
        density_map = self._create_density_map(centroids, bboxes)
        
        # Find contours in density map
        contours = self._find_contours(density_map)
        
        # Filter contours by area and count people in each contour
        crowd_events = []
        for i, contour in enumerate(contours):
            # Calculate contour area
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Create mask for this contour
            mask = np.zeros_like(density_map, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Count people in this contour
            people_in_contour = []
            for j, centroid in enumerate(centroids):
                x, y = centroid
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0:
                    people_in_contour.append(j)
            
            # If enough people in contour, consider it a crowd
            if len(people_in_contour) >= self.crowd_threshold:
                # Calculate contour centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                
                # Get people in this crowd
                crowd_people = [persons[idx] for idx in people_in_contour]
                
                crowd_events.append({
                    'crowd_id': i,
                    'people_count': len(people_in_contour),
                    'centroid': (cx, cy),
                    'people': crowd_people
                })
        
        return crowd_events
    
    def _create_density_map(self, centroids: List[Tuple[int, int]], 
                           bboxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Create a density map from person positions.
        
        Args:
            centroids: List of person centroids
            bboxes: List of person bounding boxes
            
        Returns:
            Density map as a numpy array
        """
        # Find frame dimensions from bounding boxes
        if not bboxes:
            return np.zeros((100, 100), dtype=np.uint8)
            
        x_coords = [x for bbox in bboxes for x in (bbox[0], bbox[2])]
        y_coords = [y for bbox in bboxes for y in (bbox[1], bbox[3])]
        
        max_x = max(x_coords) + 100  # Add margin
        max_y = max(y_coords) + 100  # Add margin
        
        # Create empty density map
        density_map = np.zeros((max_y, max_x), dtype=np.uint8)
        
        # Add Gaussian blobs at each person's position
        for centroid in centroids:
            x, y = centroid
            if 0 <= x < max_x and 0 <= y < max_y:
                # Create a small Gaussian kernel
                kernel = self._create_gaussian_kernel(self.kernel_size)
                
                # Calculate kernel placement
                kh, kw = kernel.shape
                x1, y1 = max(0, x - kw//2), max(0, y - kh//2)
                x2, y2 = min(max_x, x + kw//2 + 1), min(max_y, y + kh//2 + 1)
                
                # Calculate kernel region
                kx1, ky1 = max(0, kw//2 - x), max(0, kh//2 - y)
                kx2, ky2 = kx1 + (x2 - x1), ky1 + (y2 - y1)
                
                # Add kernel to density map
                if x2 > x1 and y2 > y1 and kx2 > kx1 and ky2 > ky1:
                    kernel_roi = kernel[ky1:ky2, kx1:kx2]
                    density_map[y1:y2, x1:x2] = np.maximum(
                        density_map[y1:y2, x1:x2],
                        kernel_roi
                    )
        
        return density_map
    
    def _create_gaussian_kernel(self, size: int) -> np.ndarray:
        """Create a Gaussian kernel for the density map."""
        # Ensure size is odd
        if size % 2 == 0:
            size += 1
            
        # Create Gaussian kernel
        kernel = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        sigma = size / 6.0  # Standard deviation
        
        # Fill kernel with Gaussian values
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                kernel[i, j] = int(255 * np.exp(-(x*x + y*y) / (2 * sigma * sigma)))
        
        return kernel
    
    def _find_contours(self, density_map: np.ndarray) -> List[np.ndarray]:
        """
        Find contours in the density map.
        
        Args:
            density_map: Density map as a numpy array
            
        Returns:
            List of contours
        """
        # Threshold the density map
        _, binary_map = cv2.threshold(
            density_map, 
            int(255 * self.density_threshold), 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Apply morphological operations to clean up the map
        kernel = np.ones((5, 5), np.uint8)
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary_map, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
