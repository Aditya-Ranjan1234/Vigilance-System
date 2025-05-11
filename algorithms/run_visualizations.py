"""
Run Algorithm Visualizations

This script runs the visualizations for the algorithms used in the Vigilance System.
It creates educational visualizations that explain how each algorithm works and its time complexity.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Tuple, Any

# Import algorithm visualization modules
from algorithms.preprocessing.video_stabilization import VideoStabilizationVisualizer
from algorithms.tracking.object_tracking import ObjectTrackingVisualizer, Detection
from algorithms.detection.decision_making import DecisionMakingVisualizer

# Create directory for visualizations
os.makedirs('visualizations', exist_ok=True)

def generate_sample_video_frames(num_frames: int = 10, width: int = 640, height: int = 480) -> List[np.ndarray]:
    """
    Generate sample video frames for visualization.

    Args:
        num_frames: Number of frames to generate
        width: Frame width
        height: Frame height

    Returns:
        List of video frames
    """
    frames = []

    # Create a black background
    background = np.zeros((height, width, 3), dtype=np.uint8)

    # Add a grid pattern
    for i in range(0, height, 50):
        cv2.line(background, (0, i), (width, i), (50, 50, 50), 1)
    for i in range(0, width, 50):
        cv2.line(background, (i, 0), (i, height), (50, 50, 50), 1)

    # Generate frames with moving objects
    for i in range(num_frames):
        frame = background.copy()

        # Add frame number
        cv2.putText(frame, f"Frame {i+1}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (200, 200, 200), 2)

        # Add moving rectangle (simulating camera shake)
        offset_x = int(10 * np.sin(i * 0.5))
        offset_y = int(5 * np.cos(i * 0.7))

        cv2.rectangle(frame,
                     (100 + offset_x, 100 + offset_y),
                     (200 + offset_x, 200 + offset_y),
                     (0, 0, 255), 3)

        # Add moving circle (simulating an object)
        circle_x = int(width // 2 + 100 * np.sin(i * 0.2))
        circle_y = int(height // 2 + 50 * np.cos(i * 0.3))

        cv2.circle(frame, (circle_x, circle_y), 30, (0, 255, 0), -1)

        frames.append(frame)

    return frames

def generate_sample_detections(frames: List[np.ndarray]) -> List[List[Detection]]:
    """
    Generate sample object detections for visualization.

    Args:
        frames: List of video frames

    Returns:
        List of detection lists for each frame
    """
    detection_results = []

    # Generate detections for each frame
    for i, frame in enumerate(frames):
        height, width = frame.shape[:2]
        frame_detections = []

        # Add a "person" detection that moves across the frame
        person_x = int(width * 0.2 + width * 0.6 * (i / len(frames)))
        person_y = int(height * 0.5)
        person_width = 80
        person_height = 160

        person_detection = Detection(
            bbox=(person_x - person_width//2, person_y - person_height//2,
                 person_x + person_width//2, person_y + person_height//2),
            class_id=0,
            class_name="person",
            confidence=0.95,
            timestamp=time.time() + i  # Simulate time passing
        )
        frame_detections.append(person_detection)

        # Add another "person" detection that stays in one place (loitering)
        loitering_x = int(width * 0.3)
        loitering_y = int(height * 0.7)

        loitering_detection = Detection(
            bbox=(loitering_x - person_width//2, loitering_y - person_height//2,
                 loitering_x + person_width//2, loitering_y + person_height//2),
            class_id=0,
            class_name="person",
            confidence=0.92,
            timestamp=time.time() + i  # Simulate time passing
        )
        frame_detections.append(loitering_detection)

        # Add more "person" detections in later frames to trigger crowd detection
        if i > len(frames) // 2:
            for j in range(3):
                crowd_x = int(width * (0.5 + j * 0.15))
                crowd_y = int(height * 0.6)

                crowd_detection = Detection(
                    bbox=(crowd_x - person_width//2, crowd_y - person_height//2,
                         crowd_x + person_width//2, crowd_y + person_height//2),
                    class_id=0,
                    class_name="person",
                    confidence=0.9,
                    timestamp=time.time() + i  # Simulate time passing
                )
                frame_detections.append(crowd_detection)

        # Add a "car" detection
        car_x = int(width * 0.7 - i * 10)  # Moving right to left
        car_y = int(height * 0.3)
        car_width = 120
        car_height = 80

        if car_x + car_width//2 > 0:  # Only add if still visible
            car_detection = Detection(
                bbox=(car_x - car_width//2, car_y - car_height//2,
                     car_x + car_width//2, car_y + car_height//2),
                class_id=1,
                class_name="car",
                confidence=0.88,
                timestamp=time.time() + i  # Simulate time passing
            )
            frame_detections.append(car_detection)

        detection_results.append(frame_detections)

    return detection_results

def generate_sample_tracked_objects(frames: List[np.ndarray],
                                   detections: List[List[Detection]]) -> List[Dict[int, Detection]]:
    """
    Generate sample tracked objects for visualization.

    Args:
        frames: List of video frames
        detections: List of detection lists for each frame

    Returns:
        List of tracked objects dictionaries for each frame
    """
    # Initialize object tracking visualizer
    tracker = ObjectTrackingVisualizer(max_disappeared=5, max_distance=100.0)

    tracked_objects_sequence = []

    # Process each frame
    for i, frame in enumerate(frames):
        # Update tracking with detections
        tracked_objects = tracker.update(frame, detections[i])
        tracked_objects_sequence.append(tracked_objects)

    return tracked_objects_sequence

def visualize_all_algorithms():
    """Run visualizations for all algorithms."""
    print("Generating sample data...")

    # Generate sample video frames
    frames = generate_sample_video_frames(num_frames=20)

    # Generate sample detections
    detections = generate_sample_detections(frames)

    # Generate sample tracked objects
    tracked_objects_sequence = generate_sample_tracked_objects(frames, detections)

    print("Running algorithm visualizations...")

    # Visualize video stabilization
    print("\nVisualizing Video Stabilization Algorithm...")
    stabilizer = VideoStabilizationVisualizer(method='optical_flow')

    for frame in frames:
        stabilized = stabilizer.process_frame(frame)

    stabilizer.save_animation("video_stabilization")
    stabilizer.create_explanation()

    # Visualize object tracking
    print("\nVisualizing Object Tracking Algorithm...")
    tracker = ObjectTrackingVisualizer()

    for i, frame in enumerate(frames):
        tracked_objects = tracker.update(frame, detections[i])

    tracker.save_animation("object_tracking")
    tracker.create_explanation()

    # Visualize decision making
    print("\nVisualizing Decision Making Algorithms...")
    decision_maker = DecisionMakingVisualizer(person_loitering_time=5.0, crowd_threshold=3)

    for i, frame in enumerate(frames):
        if i < len(tracked_objects_sequence):
            alerts = decision_maker.update(frame, tracked_objects_sequence[i])

    decision_maker.save_animation("decision_making")
    decision_maker.create_explanation()

    print("\nAll visualizations completed. Results saved in 'visualizations' directory.")

if __name__ == "__main__":
    visualize_all_algorithms()
