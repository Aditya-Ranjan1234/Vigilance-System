"""
Simple example of using the vigilance system for object detection.

This script demonstrates how to use the vigilance system to detect objects
in a video file or camera stream.
"""

import os
import sys
import cv2
import time
import argparse
from pathlib import Path

# Add parent directory to path to import vigilance_system
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vigilance_system.utils.logger import setup_logger
from vigilance_system.detection.model_loader import model_loader
from vigilance_system.detection.object_detector import ObjectDetector


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Simple Object Detection Example')
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to video file or camera index (0, 1, etc.)'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='yolov5s',
        help='Model to use for detection'
    )
    
    parser.add_argument(
        '--confidence', 
        type=float, 
        default=0.5,
        help='Confidence threshold for detections'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        help='Path to output video file'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logger
    setup_logger('INFO')
    
    # Initialize detector
    detector = ObjectDetector(args.model, args.confidence)
    
    # Open video source
    try:
        # Check if input is a camera index
        if args.input.isdigit():
            cap = cv2.VideoCapture(int(args.input))
        else:
            cap = cv2.VideoCapture(args.input)
    except Exception as e:
        print(f"Error opening video source: {e}")
        return 1
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.input}")
        return 1
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output is specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Process video
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            start_time = time.time()
            detections = detector.detect(frame)
            elapsed = time.time() - start_time
            
            # Draw detections
            frame_with_detections = detector.draw_detections(frame, detections)
            
            # Add FPS information
            fps_text = f"FPS: {1/elapsed:.2f}" if elapsed > 0 else "FPS: N/A"
            cv2.putText(frame_with_detections, fps_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Object Detection', frame_with_detections)
            
            # Write frame if output is specified
            if writer:
                writer.write(frame_with_detections)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        # Release resources
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
