"""
Helper functions for trajectory processing.
"""

from typing import List, Tuple, Any, Dict, Union
import numpy as np

def ensure_valid_trajectory_points(trajectory: List[Any]) -> List[Tuple[int, int]]:
    """
    Ensure all trajectory points are valid tuples of integers.
    
    Args:
        trajectory: List of trajectory points in various possible formats
        
    Returns:
        List[Tuple[int, int]]: List of validated trajectory points as tuples of integers
    """
    valid_trajectory = []
    
    for point in trajectory:
        try:
            # Handle different possible formats of trajectory points
            if isinstance(point, (list, tuple)) and len(point) == 2:
                x, y = point
                valid_trajectory.append((int(x), int(y)))
            elif isinstance(point, np.ndarray) and point.size == 2:
                valid_trajectory.append((int(point[0]), int(point[1])))
            else:
                # Skip invalid points
                continue
        except (ValueError, TypeError):
            # Skip points that can't be converted to integers
            continue
    
    return valid_trajectory
