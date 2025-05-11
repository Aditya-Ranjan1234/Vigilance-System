"""
OpenCV utility functions.
"""

import cv2
import numpy as np
from typing import Tuple, Union, Any

def safe_putText(img: np.ndarray, 
                text: str, 
                org: Union[Tuple[int, int], Any], 
                fontFace: int, 
                fontScale: float, 
                color: Tuple[int, int, int], 
                thickness: int = 1,
                lineType: int = cv2.LINE_AA) -> None:
    """
    A safe wrapper for cv2.putText that ensures coordinates are integers.
    
    Args:
        img: Image to draw on
        text: Text to draw
        org: Bottom-left corner of the text
        fontFace: Font type
        fontScale: Font scale
        color: Text color
        thickness: Line thickness
        lineType: Line type
    """
    # Ensure org is a tuple of integers
    try:
        x, y = org
        org_fixed = (int(x), int(y))
        cv2.putText(img, text, org_fixed, fontFace, fontScale, color, thickness, lineType)
    except Exception as e:
        # If there's an error, try to recover
        if isinstance(org, (list, tuple)) and len(org) == 2:
            try:
                # Try to convert each element to int
                org_fixed = (int(org[0]), int(org[1]))
                cv2.putText(img, text, org_fixed, fontFace, fontScale, color, thickness, lineType)
            except Exception:
                # If all else fails, use a default position
                cv2.putText(img, text, (10, 30), fontFace, fontScale, color, thickness, lineType)
        else:
            # Use a default position
            cv2.putText(img, text, (10, 30), fontFace, fontScale, color, thickness, lineType)
