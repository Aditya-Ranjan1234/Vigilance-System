import cv2
import numpy as np
import time
import logging
from collections import defaultdict

logger = logging.getLogger('algorithm_demo.kalman_tracker')

class KalmanTracker:
    """
    Kalman filter-based object tracking implementation.
    
    This class demonstrates the steps of Kalman filter tracking:
    1. Detect objects using background subtraction
    2. Initialize Kalman filters for new objects
    3. Predict new positions using Kalman filters
    4. Update Kalman filters with new measurements
    5. Track objects across frames
    6. Calculate metrics
    """
    
    def __init__(self):
        """Initialize Kalman tracker."""
        # Initialize background subtractor for detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
        
        # Parameters
        self.min_contour_area = 500  # Minimum contour area to be considered an object
        self.max_tracking_age = 10   # Maximum number of frames to keep tracking without detection
        self.min_detection_count = 3 # Minimum number of detections to start tracking
        
        # Tracking state
        self.tracks = {}             # Dictionary of tracked objects
        self.next_track_id = 0       # Next track ID to assign
        
        # Metrics
        self.total_objects = 0
        self.frame_count = 0
        self.track_lengths = []
        
        logger.info("Kalman tracker initialized")
    
    def create_kalman_filter(self):
        """Create and initialize a Kalman filter for tracking."""
        kalman = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurement variables
        
        # State transition matrix (x, y, dx, dy)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + dx
            [0, 1, 0, 1],  # y = y + dy
            [0, 0, 1, 0],  # dx = dx
            [0, 0, 0, 1]   # dy = dy
        ], np.float32)
        
        # Measurement matrix (we only measure x, y)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Process noise covariance
        kalman.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32) * 0.03
        
        # Measurement noise covariance
        kalman.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1]
        ], np.float32) * 0.1
        
        return kalman
    
    def detect_objects(self, frame):
        """Detect objects in the frame using background subtraction."""
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(blurred_frame, learningRate=0.01)
        
        # Apply threshold to get binary mask
        _, binary_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and extract object information
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            detections.append({
                'center': (center_x, center_y),
                'bbox': (x, y, w, h),
                'area': cv2.contourArea(contour)
            })
        
        return detections, {
            'fg_mask': fg_mask,
            'binary_mask': binary_mask,
            'morphology': closing
        }
    
    def process(self, frame):
        """
        Process a frame with Kalman filter tracking.
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (result_frame, step_frames, metrics)
                - result_frame: Frame with tracking visualization
                - step_frames: Dictionary of intermediate frames for visualization
                - metrics: Dictionary of metrics
        """
        # Start timing
        start_time = time.time()
        
        # Create a copy of the frame
        original_frame = frame.copy()
        result_frame = original_frame.copy()
        
        # Step 1: Detect objects
        detections, mask_frames = self.detect_objects(frame)
        
        # Step 2: Predict new positions for existing tracks
        for track_id, track in list(self.tracks.items()):
            # Predict new position
            prediction = track['kalman'].predict()
            
            # Get predicted position
            pred_x = int(prediction[0])
            pred_y = int(prediction[1])
            
            # Update track with prediction
            track['predicted'] = (pred_x, pred_y)
            track['age'] += 1
            
            # Remove old tracks
            if track['age'] > self.max_tracking_age:
                # Record track length for metrics
                self.track_lengths.append(track['age'])
                del self.tracks[track_id]
        
        # Step 3: Associate detections with existing tracks
        assigned_tracks = set()
        assigned_detections = set()
        
        # Calculate distance matrix
        distance_matrix = {}
        for i, detection in enumerate(detections):
            for track_id, track in self.tracks.items():
                # Calculate distance between detection and track prediction
                det_x, det_y = detection['center']
                pred_x, pred_y = track['predicted']
                
                distance = np.sqrt((det_x - pred_x) ** 2 + (det_y - pred_y) ** 2)
                distance_matrix[(i, track_id)] = distance
        
        # Assign detections to tracks (greedy algorithm)
        if distance_matrix:
            # Sort distances
            sorted_distances = sorted(distance_matrix.items(), key=lambda x: x[1])
            
            # Assign detections to tracks
            for (det_idx, track_id), distance in sorted_distances:
                # Only assign if both detection and track are not already assigned
                if det_idx not in assigned_detections and track_id not in assigned_tracks:
                    # Only assign if distance is reasonable
                    if distance < 100:  # Maximum distance for assignment
                        assigned_detections.add(det_idx)
                        assigned_tracks.add(track_id)
                        
                        # Update track with new measurement
                        detection = detections[det_idx]
                        track = self.tracks[track_id]
                        
                        # Convert center to measurement
                        measurement = np.array([[np.float32(detection['center'][0])], 
                                              [np.float32(detection['center'][1])]])
                        
                        # Correct Kalman filter with measurement
                        track['kalman'].correct(measurement)
                        
                        # Update track information
                        track['center'] = detection['center']
                        track['bbox'] = detection['bbox']
                        track['age'] = 0  # Reset age
                        track['detection_count'] += 1
        
        # Step 4: Create new tracks for unassigned detections
        for i, detection in enumerate(detections):
            if i not in assigned_detections:
                # Create new track
                kalman = self.create_kalman_filter()
                
                # Initialize Kalman filter with first measurement
                measurement = np.array([[np.float32(detection['center'][0])], 
                                      [np.float32(detection['center'][1])]])
                kalman.statePost = np.array([[measurement[0, 0]], 
                                           [measurement[1, 0]], 
                                           [0], 
                                           [0]], dtype=np.float32)
                
                # Add new track
                self.tracks[self.next_track_id] = {
                    'kalman': kalman,
                    'center': detection['center'],
                    'bbox': detection['bbox'],
                    'predicted': detection['center'],
                    'age': 0,
                    'detection_count': 1,
                    'trajectory': [detection['center']]
                }
                
                self.next_track_id += 1
        
        # Step 5: Draw tracking results
        objects_tracked = 0
        
        for track_id, track in self.tracks.items():
            # Only draw tracks with enough detections
            if track['detection_count'] >= self.min_detection_count:
                # Get track information
                x, y, w, h = track['bbox']
                center_x, center_y = track['center']
                pred_x, pred_y = track['predicted']
                
                # Draw bounding box
                confidence = min(1.0, track['detection_count'] / 10)
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw ID
                cv2.putText(result_frame, f"ID: {track_id}", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw center point
                cv2.circle(result_frame, (center_x, center_y), 4, (0, 0, 255), -1)
                
                # Draw predicted point
                cv2.circle(result_frame, (pred_x, pred_y), 4, (255, 0, 0), -1)
                
                # Draw line from center to prediction
                cv2.line(result_frame, (center_x, center_y), (pred_x, pred_y), (255, 0, 255), 2)
                
                # Update trajectory
                track['trajectory'].append((center_x, center_y))
                
                # Draw trajectory
                if len(track['trajectory']) > 1:
                    for i in range(1, len(track['trajectory'])):
                        cv2.line(result_frame, track['trajectory'][i-1], track['trajectory'][i], 
                                (0, 255, 255), 1)
                
                objects_tracked += 1
        
        # Update metrics
        self.total_objects = max(self.total_objects, objects_tracked)
        self.frame_count += 1
        
        # Create colored versions of masks for visualization
        fg_mask_colored = cv2.cvtColor(mask_frames['fg_mask'], cv2.COLOR_GRAY2BGR)
        binary_mask_colored = cv2.cvtColor(mask_frames['binary_mask'], cv2.COLOR_GRAY2BGR)
        morphology_colored = cv2.cvtColor(mask_frames['morphology'], cv2.COLOR_GRAY2BGR)
        
        # Create prediction visualization
        prediction_frame = original_frame.copy()
        for track_id, track in self.tracks.items():
            if track['detection_count'] >= self.min_detection_count:
                # Draw predicted position
                pred_x, pred_y = track['predicted']
                cv2.circle(prediction_frame, (pred_x, pred_y), 6, (255, 0, 0), -1)
                cv2.putText(prediction_frame, f"ID: {track_id}", (pred_x, pred_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add step frames for visualization
        step_frames = {
            'original': original_frame,
            'fg_mask': fg_mask_colored,
            'binary_mask': binary_mask_colored,
            'morphology': morphology_colored,
            'prediction': prediction_frame,
            'result': result_frame
        }
        
        # Calculate metrics
        processing_time = time.time() - start_time
        avg_track_length = np.mean(self.track_lengths) if self.track_lengths else 0
        
        metrics = {
            "objects_tracked": objects_tracked,
            "total_objects": self.total_objects,
            "average_track_length": avg_track_length,
            "processing_time": processing_time
        }
        
        return result_frame, step_frames, metrics
