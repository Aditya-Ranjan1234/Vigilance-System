"""
Object Tracking Algorithm Visualization.

This module visualizes the object tracking algorithm used in the Vigilance System.
The algorithm uses distance-based metrics to associate objects across frames and track
objects over time.

Time Complexity: O(n*m) where n is the number of tracked objects and m is the number of new detections
Space Complexity: O(n) for storing the tracked objects
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Tuple, Any, Optional
import os
import time
from collections import defaultdict

# Create a directory for saving visualizations
os.makedirs('visualizations', exist_ok=True)

class Detection:
    """Class representing a detection from an object detector."""
    
    def __init__(self, bbox: Tuple[float, float, float, float], class_id: int, 
                class_name: str, confidence: float, timestamp: float = None):
        """
        Initialize a detection.
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            class_id: Class ID of the detected object
            class_name: Class name of the detected object
            confidence: Confidence score of the detection
            timestamp: Timestamp of the detection
        """
        self.bbox = bbox
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.timestamp = timestamp or time.time()
        
        # Calculate center coordinates
        self.center_x = (bbox[0] + bbox[2]) / 2
        self.center_y = (bbox[1] + bbox[3]) / 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the detection to a dictionary."""
        return {
            'bbox': self.bbox,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'center_x': self.center_x,
            'center_y': self.center_y
        }


class ObjectTrackingVisualizer:
    """Visualizer for the object tracking algorithm."""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0):
        """
        Initialize the object tracking visualizer.
        
        Args:
            max_disappeared: Maximum number of frames an object can disappear before being removed
            max_distance: Maximum distance between detections to be considered the same object
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_object_id = 0
        self.objects = {}  # Map from object ID to centroid
        self.disappeared = defaultdict(int)  # Map from object ID to number of frames disappeared
        self.object_history = defaultdict(list)  # Map from object ID to list of detections
        self.colors = {}  # Map from object ID to color
        
        # Visualization setup
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle("Object Tracking Algorithm", fontsize=16)
        self.frames = []
        
        # Frame counter
        self.frame_count = 0
        
        # Tracking metrics
        self.matches_history = []
        self.new_objects_history = []
        self.disappeared_objects_history = []
    
    def update(self, frame: np.ndarray, detections: List[Detection]) -> Dict[int, Detection]:
        """
        Update object tracking with new detections and visualize the process.
        
        Args:
            frame: Current video frame
            detections: List of new detections
            
        Returns:
            Dictionary mapping object IDs to their current detections
        """
        self.frame_count += 1
        
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # If no objects are being tracked, register all detections
        if len(self.objects) == 0:
            for detection in detections:
                self._register(detection)
            
            # Visualize the initial registrations
            self._visualize_process(vis_frame, detections, {}, {}, {})
            
            return {obj_id: self.object_history[obj_id][-1] for obj_id in self.objects}
        
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            disappeared_objects = {}
            
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Add to disappeared objects for visualization
                if object_id in self.objects:
                    disappeared_objects[object_id] = self.objects[object_id]
                
                # Remove object if it has disappeared for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            
            # Visualize the disappearances
            self._visualize_process(vis_frame, detections, {}, {}, disappeared_objects)
            
            return {obj_id: self.object_history[obj_id][-1] for obj_id in self.objects}
        
        # Match detections to existing objects
        object_centroids = {obj_id: (obj[0], obj[1]) for obj_id, obj in self.objects.items()}
        detection_centroids = [(d.center_x, d.center_y) for d in detections]
        
        # Calculate distances between all objects and detections
        distances = {}
        for obj_id, obj_centroid in object_centroids.items():
            for i, det_centroid in enumerate(detection_centroids):
                distance = ((obj_centroid[0] - det_centroid[0]) ** 2 + 
                           (obj_centroid[1] - det_centroid[1]) ** 2) ** 0.5
                if distance <= self.max_distance:
                    distances[(obj_id, i)] = distance
        
        # Sort distances and match objects to detections
        matched_objects = set()
        matched_detections = set()
        matches = {}
        
        # Sort by distance and match greedily
        for (obj_id, det_idx), distance in sorted(distances.items(), key=lambda x: x[1]):
            if obj_id not in matched_objects and det_idx not in matched_detections:
                self.objects[obj_id] = (detection_centroids[det_idx][0], detection_centroids[det_idx][1])
                self.object_history[obj_id].append(detections[det_idx])
                self.disappeared[obj_id] = 0
                matched_objects.add(obj_id)
                matched_detections.add(det_idx)
                matches[obj_id] = (det_idx, distance)
        
        # Check for disappeared objects
        disappeared_objects = {}
        for obj_id in set(self.objects.keys()) - matched_objects:
            self.disappeared[obj_id] += 1
            disappeared_objects[obj_id] = self.objects[obj_id]
            
            if self.disappeared[obj_id] > self.max_disappeared:
                self._deregister(obj_id)
        
        # Register new detections
        new_objects = {}
        for i in range(len(detections)):
            if i not in matched_detections:
                obj_id = self._register(detections[i])
                new_objects[obj_id] = (detections[i].center_x, detections[i].center_y)
        
        # Update tracking metrics
        self.matches_history.append(len(matches))
        self.new_objects_history.append(len(new_objects))
        self.disappeared_objects_history.append(len(disappeared_objects))
        
        # Visualize the tracking process
        self._visualize_process(vis_frame, detections, matches, new_objects, disappeared_objects)
        
        # Return current detections for each tracked object
        return {obj_id: self.object_history[obj_id][-1] for obj_id in self.objects}
    
    def _register(self, detection: Detection) -> int:
        """
        Register a new object.
        
        Args:
            detection: Detection to register as a new object
            
        Returns:
            ID of the registered object
        """
        object_id = self.next_object_id
        self.objects[object_id] = (detection.center_x, detection.center_y)
        self.object_history[object_id].append(detection)
        self.disappeared[object_id] = 0
        
        # Assign a random color to the object
        self.colors[object_id] = (
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256)
        )
        
        self.next_object_id += 1
        return object_id
    
    def _deregister(self, object_id: int) -> None:
        """
        Deregister an object.
        
        Args:
            object_id: ID of the object to deregister
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.object_history[object_id]
        if object_id in self.colors:
            del self.colors[object_id]
    
    def _visualize_process(self, frame: np.ndarray, detections: List[Detection],
                          matches: Dict[int, Tuple[int, float]],
                          new_objects: Dict[int, Tuple[float, float]],
                          disappeared_objects: Dict[int, Tuple[float, float]]) -> None:
        """
        Visualize the tracking process.
        
        Args:
            frame: Current video frame
            detections: List of new detections
            matches: Dictionary mapping object IDs to (detection index, distance) tuples
            new_objects: Dictionary mapping new object IDs to their centroids
            disappeared_objects: Dictionary mapping disappeared object IDs to their centroids
        """
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame with tracking information
        self.axes[0, 0].imshow(frame_rgb)
        self.axes[0, 0].set_title(f"Frame {self.frame_count}")
        self.axes[0, 0].axis('off')
        
        # Draw detections
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                linewidth=2, edgecolor='yellow', facecolor='none')
            self.axes[0, 0].add_patch(rect)
            
            # Draw detection label
            self.axes[0, 0].text(x1, y1 - 10, f"{detection.class_name} ({detection.confidence:.2f})", 
                                color='yellow', fontsize=8)
            
            # Draw centroid
            self.axes[0, 0].plot(detection.center_x, detection.center_y, 'yo', markersize=5)
        
        # Draw tracked objects
        for obj_id, centroid in self.objects.items():
            color = self.colors.get(obj_id, (0, 255, 0))
            color_rgb = (color[0]/255, color[1]/255, color[2]/255)
            
            # Get the latest detection for this object
            if obj_id in self.object_history and self.object_history[obj_id]:
                latest_detection = self.object_history[obj_id][-1]
                x1, y1, x2, y2 = latest_detection.bbox
                
                # Draw bounding box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                    linewidth=2, edgecolor=color_rgb, facecolor='none')
                self.axes[0, 0].add_patch(rect)
                
                # Draw object ID and class
                self.axes[0, 0].text(x1, y1 - 10, f"ID: {obj_id} ({latest_detection.class_name})", 
                                    color=color_rgb, fontsize=8, weight='bold')
            
            # Draw centroid
            self.axes[0, 0].plot(centroid[0], centroid[1], 'o', color=color_rgb, markersize=5)
            
            # Draw trajectory
            if len(self.object_history[obj_id]) > 1:
                trajectory_x = [d.center_x for d in self.object_history[obj_id][-10:]]
                trajectory_y = [d.center_y for d in self.object_history[obj_id][-10:]]
                self.axes[0, 0].plot(trajectory_x, trajectory_y, '-', color=color_rgb, linewidth=1, alpha=0.5)
        
        # Visualize matches
        for obj_id, (det_idx, distance) in matches.items():
            obj_centroid = self.objects[obj_id]
            det_centroid = (detections[det_idx].center_x, detections[det_idx].center_y)
            
            # Draw a line between the centroids
            self.axes[0, 0].plot([obj_centroid[0], det_centroid[0]], 
                                [obj_centroid[1], det_centroid[1]], 
                                '--', color='green', linewidth=1)
            
            # Draw the distance
            mid_x = (obj_centroid[0] + det_centroid[0]) / 2
            mid_y = (obj_centroid[1] + det_centroid[1]) / 2
            self.axes[0, 0].text(mid_x, mid_y, f"{distance:.1f}", 
                                color='green', fontsize=8)
        
        # Visualize tracking metrics
        self.axes[0, 1].set_title("Tracking Metrics")
        
        # Plot tracking metrics over time
        if len(self.matches_history) > 1:
            frames = list(range(1, len(self.matches_history) + 1))
            self.axes[0, 1].plot(frames, self.matches_history, 'g-', label='Matches')
            self.axes[0, 1].plot(frames, self.new_objects_history, 'b-', label='New Objects')
            self.axes[0, 1].plot(frames, self.disappeared_objects_history, 'r-', label='Disappeared')
            self.axes[0, 1].set_xlabel('Frame')
            self.axes[0, 1].set_ylabel('Count')
            self.axes[0, 1].legend(loc='upper right')
            self.axes[0, 1].grid(True, alpha=0.3)
        else:
            self.axes[0, 1].text(0.5, 0.5, "Insufficient data", 
                                ha='center', va='center', 
                                transform=self.axes[0, 1].transAxes)
        
        # Visualize current frame metrics
        self.axes[1, 0].set_title("Current Frame Analysis")
        self.axes[1, 0].axis('off')
        
        # Create a table with current frame metrics
        metrics_text = f"""
        Frame: {self.frame_count}
        
        Detections: {len(detections)}
        Tracked Objects: {len(self.objects)}
        
        Matches: {len(matches)}
        New Objects: {len(new_objects)}
        Disappeared Objects: {len(disappeared_objects)}
        
        Max Distance Threshold: {self.max_distance}
        Max Disappeared Frames: {self.max_disappeared}
        """
        
        self.axes[1, 0].text(0.1, 0.9, metrics_text, fontsize=10, 
                            verticalalignment='top', horizontalalignment='left',
                            transform=self.axes[1, 0].transAxes)
        
        # Visualize algorithm explanation
        self.axes[1, 1].set_title("Algorithm Explanation")
        self.axes[1, 1].axis('off')
        
        explanation_text = """
        Object Tracking Algorithm:
        
        1. For each new frame, get object detections
        2. If no objects are being tracked, register all detections
        3. If no detections in current frame, mark all objects as disappeared
        4. Calculate distances between all tracked objects and new detections
        5. Match objects to detections greedily by distance (closest first)
        6. Update positions of matched objects
        7. Mark unmatched objects as disappeared
        8. Register unmatched detections as new objects
        9. Remove objects that have been disappeared for too long
        
        Time Complexity: O(n*m) where n is the number of tracked objects 
                        and m is the number of new detections
        Space Complexity: O(n) for storing the tracked objects
        """
        
        self.axes[1, 1].text(0.05, 0.95, explanation_text, fontsize=10, 
                            verticalalignment='top', horizontalalignment='left',
                            transform=self.axes[1, 1].transAxes)
        
        # Capture the current state for animation
        self.fig.tight_layout()
        self.fig.canvas.draw()
        frame = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.frames.append(frame)
    
    def save_animation(self, filename: str = "object_tracking", fps: int = 5) -> None:
        """
        Save the animation to a file.
        
        Args:
            filename: Name of the file to save (without extension)
            fps: Frames per second for the animation
        """
        if not self.frames:
            print("No frames to save")
            return
            
        path = os.path.join('visualizations', f"{filename}.gif")
        
        # Create animation
        ani = animation.ArtistAnimation(self.fig, 
                                       [[plt.imshow(frame)] for frame in self.frames], 
                                       interval=1000//fps, blit=True)
        
        # Save animation
        ani.save(path, writer='pillow', fps=fps)
        print(f"Animation saved to {path}")
    
    def show(self) -> None:
        """Display the current visualization."""
        plt.tight_layout()
        plt.show()
    
    def create_explanation(self) -> None:
        """Create an educational figure explaining the object tracking algorithm."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Title
        fig.suptitle("Object Tracking Algorithm", fontsize=16, fontweight='bold')
        
        # Example visualization
        if len(self.frames) > 0:
            # Use the last frame as an example
            example_img = self.frames[-1]
            ax1.imshow(example_img)
            ax1.set_title("Example Visualization")
            ax1.axis('off')
        else:
            ax1.set_title("No Example Available")
            ax1.axis('off')
        
        # Algorithm explanation
        explanation = """
        Object Tracking Algorithm
        
        The object tracking algorithm associates detections across frames to track objects over time.
        
        Algorithm Steps:
        1. For each new frame, get object detections from the detector
        2. If no objects are being tracked, register all detections as new objects
        3. If no detections in current frame, mark all objects as disappeared
        4. Calculate distances between all tracked objects and new detections
        5. Match objects to detections greedily by distance (closest first)
        6. Update positions of matched objects
        7. Mark unmatched objects as disappeared
        8. Register unmatched detections as new objects
        9. Remove objects that have been disappeared for too long
        
        Time Complexity: O(n*m) where n is the number of tracked objects and m is the number of new detections
        Space Complexity: O(n) for storing the tracked objects
        
        Applications:
        • Person tracking for loitering detection
        • Vehicle tracking for traffic analysis
        • Object counting and trajectory analysis
        • Multi-object tracking for security systems
        • Crowd monitoring and management
        """
        
        ax2.text(0.05, 0.95, explanation, fontsize=12, 
                verticalalignment='top', horizontalalignment='left',
                transform=ax2.transAxes)
        ax2.axis('off')
        
        # Save the explanation
        plt.tight_layout()
        plt.savefig(os.path.join('visualizations', 'object_tracking_explanation.png'))
        plt.close()
        print("Explanation saved to visualizations/object_tracking_explanation.png")


def visualize_object_tracking(video_path: str, detection_results: List[List[Detection]], 
                             max_disappeared: int = 30, max_distance: float = 50.0,
                             max_frames: int = 100) -> None:
    """
    Visualize the object tracking algorithm on a video file with pre-computed detections.
    
    Args:
        video_path: Path to the video file
        detection_results: List of detection lists for each frame
        max_disappeared: Maximum number of frames an object can disappear before being removed
        max_distance: Maximum distance between detections to be considered the same object
        max_frames: Maximum number of frames to process
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Create the visualizer
    visualizer = ObjectTrackingVisualizer(max_disappeared, max_distance)
    
    # Process frames
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames and frame_count < len(detection_results):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections for this frame
        detections = detection_results[frame_count]
        
        # Update tracking
        tracked_objects = visualizer.update(frame, detections)
        
        frame_count += 1
    
    # Release the video capture
    cap.release()
    
    # Save the animation
    visualizer.save_animation("object_tracking")
    
    # Create explanation
    visualizer.create_explanation()
    
    # Show the visualization
    visualizer.show()


if __name__ == "__main__":
    # Example usage
    # This would typically be called with actual video and detection data
    print("To use this module, provide a video file and detection results.")
