#!/usr/bin/env python
"""
Test script for the new non-deep learning algorithms.

This script tests each of the new algorithms to make sure they work properly.
"""

import os
import sys
import cv2
import numpy as np
import time
import argparse
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vigilance_system.detection.algorithms.background_subtraction_detector import BackgroundSubtractionDetector
from vigilance_system.detection.algorithms.mog2_detector import MOG2Detector
from vigilance_system.detection.algorithms.knn_detector import KNNDetector
from vigilance_system.detection.algorithms.svm_classifier_detector import SVMClassifierDetector


def test_detection_algorithms(image_path: str) -> None:
    """
    Test all detection algorithms on a single image.
    
    Args:
        image_path: Path to the test image
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Create detectors
    detectors = {
        "Background Subtraction": BackgroundSubtractionDetector(),
        "MOG2": MOG2Detector(),
        "KNN": KNNDetector(),
        "SVM Classifier": SVMClassifierDetector()
    }
    
    # Test each detector
    for name, detector in detectors.items():
        print(f"Testing {name} detector...")
        
        # Make sure the detector has the load_model method
        if not hasattr(detector, 'load_model'):
            print(f"Error: {name} detector does not have load_model method")
            continue
        
        # Call load_model to make sure it works
        try:
            detector.load_model()
            print(f"  load_model() successful")
        except Exception as e:
            print(f"  Error in load_model(): {str(e)}")
            continue
        
        # Test detection
        try:
            start_time = time.time()
            detections = detector.detect(image)
            end_time = time.time()
            
            print(f"  detect() successful")
            print(f"  Found {len(detections)} detections")
            print(f"  Processing time: {(end_time - start_time) * 1000:.2f} ms")
            
            # Draw detections on the image
            result_image = image.copy()
            for detection in detections:
                # Get bounding box
                if hasattr(detection, 'bbox'):
                    # Detection object
                    bbox = detection.bbox
                    confidence = detection.confidence
                    class_name = detection.class_name
                elif isinstance(detection, dict):
                    # Dictionary
                    bbox = detection['bbox']
                    confidence = detection['confidence']
                    class_name = detection['label']
                else:
                    print(f"  Unknown detection format: {type(detection)}")
                    continue
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the result image
            output_path = f"test_results_{name.lower().replace(' ', '_')}.jpg"
            cv2.imwrite(output_path, result_image)
            print(f"  Result saved to {output_path}")
            
        except Exception as e:
            print(f"  Error in detect(): {str(e)}")
    
    print("Detection algorithm tests completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test non-deep learning algorithms')
    parser.add_argument('--image', type=str, default='test_image.jpg', help='Path to test image')
    args = parser.parse_args()
    
    # Check if the image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} does not exist")
        return 1
    
    # Test detection algorithms
    test_detection_algorithms(args.image)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
