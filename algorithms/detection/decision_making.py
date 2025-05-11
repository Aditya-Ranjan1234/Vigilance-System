"""
Decision Making Algorithms Visualization.

This module visualizes the decision making algorithms used in the Vigilance System:
1. Loitering Detection: Tracks object duration and triggers alerts based on thresholds
2. Crowd Detection: Counts objects of specific classes and triggers alerts based on thresholds

Time Complexity: O(n) where n is the number of tracked objects
Space Complexity: O(n) for storing the tracked objects and their histories
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


class DecisionMakingVisualizer:
    """Visualizer for decision making algorithms."""
    
    def __init__(self, person_loitering_time: float = 30.0, crowd_threshold: int = 3):
        """
        Initialize the decision making visualizer.
        
        Args:
            person_loitering_time: Seconds a person must be present to trigger loitering alert
            crowd_threshold: Number of people to trigger a crowd alert
        """
        self.person_loitering_time = person_loitering_time
        self.crowd_threshold = crowd_threshold
        
        # Tracking data
        self.tracked_objects = {}  # Map from object ID to current detection
        self.object_history = defaultdict(list)  # Map from object ID to list of detections
        self.object_durations = {}  # Map from object ID to duration in seconds
        self.colors = {}  # Map from object ID to color
        
        # Alert history
        self.loitering_alerts = []
        self.crowd_alerts = []
        
        # Visualization setup
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle("Decision Making Algorithms", fontsize=16)
        self.frames = []
        
        # Frame counter and timestamp
        self.frame_count = 0
        self.start_time = time.time()
    
    def update(self, frame: np.ndarray, tracked_objects: Dict[int, Detection]) -> List[Dict[str, Any]]:
        """
        Update decision making with tracked objects and visualize the process.
        
        Args:
            frame: Current video frame
            tracked_objects: Dictionary mapping object IDs to their current detections
            
        Returns:
            List of alerts generated
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Update tracked objects
        self.tracked_objects = tracked_objects
        
        # Update object histories and durations
        for obj_id, detection in tracked_objects.items():
            # Add to history
            self.object_history[obj_id].append(detection)
            
            # Calculate duration
            if len(self.object_history[obj_id]) >= 2:
                first_detection = self.object_history[obj_id][0]
                self.object_durations[obj_id] = detection.timestamp - first_detection.timestamp
            else:
                self.object_durations[obj_id] = 0.0
            
            # Assign a color if not already assigned
            if obj_id not in self.colors:
                self.colors[obj_id] = (
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                    np.random.randint(0, 256)
                )
        
        # Check for loitering
        loitering_alerts = self._check_person_loitering()
        
        # Check for crowds
        crowd_alerts = self._check_crowd_detection()
        
        # Combine alerts
        alerts = loitering_alerts + crowd_alerts
        
        # Visualize the decision making process
        self._visualize_process(frame, loitering_alerts, crowd_alerts)
        
        return alerts
    
    def _check_person_loitering(self) -> List[Dict[str, Any]]:
        """
        Check for people loitering in the scene.
        
        Returns:
            List of loitering alerts
        """
        alerts = []
        
        for obj_id, detection in self.tracked_objects.items():
            # Check if object is a person
            if detection.class_name.lower() != 'person':
                continue
            
            # Check if person has been present for too long
            duration = self.object_durations.get(obj_id, 0.0)
            
            if duration >= self.person_loitering_time:
                alert = {
                    'type': 'loitering',
                    'object_id': obj_id,
                    'duration': duration,
                    'detection': detection.to_dict(),
                    'timestamp': time.time(),
                    'message': f"Person loitering detected for {duration:.1f} seconds"
                }
                
                alerts.append(alert)
                self.loitering_alerts.append(alert)
        
        return alerts
    
    def _check_crowd_detection(self) -> List[Dict[str, Any]]:
        """
        Check for crowds (multiple people) in the scene.
        
        Returns:
            List of crowd alerts
        """
        alerts = []
        
        # Count people
        people_count = sum(1 for d in self.tracked_objects.values() if d.class_name.lower() == 'person')
        
        if people_count >= self.crowd_threshold:
            alert = {
                'type': 'crowd',
                'people_count': people_count,
                'timestamp': time.time(),
                'message': f"Crowd detected: {people_count} people"
            }
            
            alerts.append(alert)
            self.crowd_alerts.append(alert)
        
        return alerts
    
    def _visualize_process(self, frame: np.ndarray, loitering_alerts: List[Dict[str, Any]],
                          crowd_alerts: List[Dict[str, Any]]) -> None:
        """
        Visualize the decision making process.
        
        Args:
            frame: Current video frame
            loitering_alerts: List of loitering alerts for the current frame
            crowd_alerts: List of crowd alerts for the current frame
        """
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame with decision making information
        self.axes[0, 0].imshow(frame_rgb)
        self.axes[0, 0].set_title(f"Frame {self.frame_count}")
        self.axes[0, 0].axis('off')
        
        # Draw tracked objects
        for obj_id, detection in self.tracked_objects.items():
            color = self.colors.get(obj_id, (0, 255, 0))
            color_rgb = (color[0]/255, color[1]/255, color[2]/255)
            
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                linewidth=2, edgecolor=color_rgb, facecolor='none')
            self.axes[0, 0].add_patch(rect)
            
            # Draw object ID, class, and duration
            duration = self.object_durations.get(obj_id, 0.0)
            self.axes[0, 0].text(x1, y1 - 10, 
                                f"ID: {obj_id} ({detection.class_name}) - {duration:.1f}s", 
                                color=color_rgb, fontsize=8, weight='bold')
            
            # Highlight loitering objects
            if any(alert['object_id'] == obj_id for alert in loitering_alerts):
                # Draw a red highlight around the object
                rect = plt.Rectangle((x1-5, y1-5), x2-x1+10, y2-y1+10, 
                                    linewidth=3, edgecolor='red', facecolor='none')
                self.axes[0, 0].add_patch(rect)
                
                # Add loitering label
                self.axes[0, 0].text(x1, y2 + 15, "LOITERING", 
                                    color='red', fontsize=10, weight='bold')
        
        # Add crowd alert if present
        if crowd_alerts:
            people_count = crowd_alerts[0]['people_count']
            self.axes[0, 0].text(10, 30, f"CROWD DETECTED: {people_count} people", 
                                color='red', fontsize=12, weight='bold',
                                bbox=dict(facecolor='white', alpha=0.7))
        
        # Visualize object durations
        self.axes[0, 1].set_title("Object Durations")
        
        # Sort objects by duration
        sorted_durations = sorted(self.object_durations.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_durations:
            obj_ids = [f"ID: {obj_id}" for obj_id, _ in sorted_durations]
            durations = [duration for _, duration in sorted_durations]
            
            # Create horizontal bar chart
            bars = self.axes[0, 1].barh(obj_ids, durations)
            
            # Color bars based on object class and loitering threshold
            for i, (obj_id, duration) in enumerate(sorted_durations):
                if obj_id in self.tracked_objects:
                    detection = self.tracked_objects[obj_id]
                    if detection.class_name.lower() == 'person':
                        if duration >= self.person_loitering_time:
                            bars[i].set_color('red')
                        else:
                            bars[i].set_color('orange')
                    else:
                        bars[i].set_color('blue')
            
            # Add loitering threshold line
            self.axes[0, 1].axvline(x=self.person_loitering_time, color='red', linestyle='--', 
                                   label=f'Loitering Threshold ({self.person_loitering_time}s)')
            
            self.axes[0, 1].set_xlabel('Duration (seconds)')
            self.axes[0, 1].set_ylabel('Object ID')
            self.axes[0, 1].legend()
            self.axes[0, 1].grid(True, alpha=0.3)
        else:
            self.axes[0, 1].text(0.5, 0.5, "No tracked objects", 
                                ha='center', va='center', 
                                transform=self.axes[0, 1].transAxes)
        
        # Visualize alert history
        self.axes[1, 0].set_title("Alert History")
        
        # Create time series of alerts
        if self.loitering_alerts or self.crowd_alerts:
            # Create time points for all alerts
            loitering_times = [alert['timestamp'] - self.start_time for alert in self.loitering_alerts]
            crowd_times = [alert['timestamp'] - self.start_time for alert in self.crowd_alerts]
            
            # Create event plot
            if loitering_times:
                self.axes[1, 0].eventplot(loitering_times, colors='red', lineoffsets=1, 
                                         linelengths=0.5, label='Loitering Alerts')
            
            if crowd_times:
                self.axes[1, 0].eventplot(crowd_times, colors='blue', lineoffsets=2, 
                                         linelengths=0.5, label='Crowd Alerts')
            
            self.axes[1, 0].set_xlabel('Time (seconds)')
            self.axes[1, 0].set_yticks([1, 2])
            self.axes[1, 0].set_yticklabels(['Loitering', 'Crowd'])
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True, alpha=0.3)
        else:
            self.axes[1, 0].text(0.5, 0.5, "No alerts generated", 
                                ha='center', va='center', 
                                transform=self.axes[1, 0].transAxes)
        
        # Visualize algorithm explanation
        self.axes[1, 1].set_title("Decision Making Algorithms")
        self.axes[1, 1].axis('off')
        
        explanation_text = """
        Decision Making Algorithms:
        
        1. Loitering Detection:
           - Track each person's duration in the scene
           - If duration exceeds threshold (%.1fs), trigger alert
           - Used for security monitoring of restricted areas
        
        2. Crowd Detection:
           - Count number of people in the scene
           - If count exceeds threshold (%d), trigger alert
           - Used for occupancy monitoring and crowd management
        
        Time Complexity: O(n) where n is the number of tracked objects
        Space Complexity: O(n) for storing object histories
        
        Current Status:
        - Tracked Objects: %d
        - People Count: %d
        - Loitering Alerts: %d
        - Crowd Alerts: %d
        """ % (
            self.person_loitering_time,
            self.crowd_threshold,
            len(self.tracked_objects),
            sum(1 for d in self.tracked_objects.values() if d.class_name.lower() == 'person'),
            len(self.loitering_alerts),
            len(self.crowd_alerts)
        )
        
        self.axes[1, 1].text(0.05, 0.95, explanation_text, fontsize=10, 
                            verticalalignment='top', horizontalalignment='left',
                            transform=self.axes[1, 1].transAxes)
        
        # Capture the current state for animation
        self.fig.tight_layout()
        self.fig.canvas.draw()
        frame = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.frames.append(frame)
    
    def save_animation(self, filename: str = "decision_making", fps: int = 5) -> None:
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
        """Create educational figures explaining the decision making algorithms."""
        # Create loitering detection explanation
        self._create_loitering_explanation()
        
        # Create crowd detection explanation
        self._create_crowd_explanation()
    
    def _create_loitering_explanation(self) -> None:
        """Create an educational figure explaining the loitering detection algorithm."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Title
        fig.suptitle("Loitering Detection Algorithm", fontsize=16, fontweight='bold')
        
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
        Loitering Detection Algorithm
        
        The loitering detection algorithm tracks how long each person has been present in the scene
        and triggers alerts when they exceed a time threshold.
        
        Algorithm Steps:
        1. Track each person across frames using object tracking
        2. Calculate duration from first detection to current frame
        3. If duration exceeds threshold, trigger loitering alert
        4. Continue monitoring until person leaves the scene
        
        Parameters:
        - Loitering Time Threshold: %.1f seconds
        
        Time Complexity: O(n) where n is the number of tracked persons
        Space Complexity: O(n) for storing person histories
        
        Applications:
        • Security monitoring of restricted areas
        • Suspicious behavior detection
        • Retail store analytics
        • Public space management
        • Prevention of unauthorized lingering
        """ % self.person_loitering_time
        
        ax2.text(0.05, 0.95, explanation, fontsize=12, 
                verticalalignment='top', horizontalalignment='left',
                transform=ax2.transAxes)
        ax2.axis('off')
        
        # Save the explanation
        plt.tight_layout()
        plt.savefig(os.path.join('visualizations', 'loitering_detection_explanation.png'))
        plt.close()
        print("Explanation saved to visualizations/loitering_detection_explanation.png")
    
    def _create_crowd_explanation(self) -> None:
        """Create an educational figure explaining the crowd detection algorithm."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Title
        fig.suptitle("Crowd Detection Algorithm", fontsize=16, fontweight='bold')
        
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
        Crowd Detection Algorithm
        
        The crowd detection algorithm counts the number of people in the scene and
        triggers alerts when the count exceeds a threshold.
        
        Algorithm Steps:
        1. Count the number of objects classified as 'person' in the current frame
        2. If count exceeds threshold, trigger crowd alert
        3. Continue monitoring in subsequent frames
        
        Parameters:
        - Crowd Threshold: %d people
        
        Time Complexity: O(n) where n is the number of tracked objects
        Space Complexity: O(1) for storing the count
        
        Applications:
        • Occupancy monitoring for safety regulations
        • Crowd management in public spaces
        • Queue monitoring in retail and services
        • Social distancing enforcement
        • Event management and planning
        """ % self.crowd_threshold
        
        ax2.text(0.05, 0.95, explanation, fontsize=12, 
                verticalalignment='top', horizontalalignment='left',
                transform=ax2.transAxes)
        ax2.axis('off')
        
        # Save the explanation
        plt.tight_layout()
        plt.savefig(os.path.join('visualizations', 'crowd_detection_explanation.png'))
        plt.close()
        print("Explanation saved to visualizations/crowd_detection_explanation.png")


def visualize_decision_making(video_path: str, tracked_objects_sequence: List[Dict[int, Detection]],
                             person_loitering_time: float = 30.0, crowd_threshold: int = 3,
                             max_frames: int = 100) -> None:
    """
    Visualize the decision making algorithms on a video file with pre-computed tracked objects.
    
    Args:
        video_path: Path to the video file
        tracked_objects_sequence: List of tracked objects dictionaries for each frame
        person_loitering_time: Seconds a person must be present to trigger loitering alert
        crowd_threshold: Number of people to trigger a crowd alert
        max_frames: Maximum number of frames to process
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Create the visualizer
    visualizer = DecisionMakingVisualizer(person_loitering_time, crowd_threshold)
    
    # Process frames
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames and frame_count < len(tracked_objects_sequence):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get tracked objects for this frame
        tracked_objects = tracked_objects_sequence[frame_count]
        
        # Update decision making
        alerts = visualizer.update(frame, tracked_objects)
        
        frame_count += 1
    
    # Release the video capture
    cap.release()
    
    # Save the animation
    visualizer.save_animation("decision_making")
    
    # Create explanations
    visualizer.create_explanation()
    
    # Show the visualization
    visualizer.show()


if __name__ == "__main__":
    # Example usage
    # This would typically be called with actual video and tracking data
    print("To use this module, provide a video file and tracked objects sequence.")
