import cv2
import numpy as np
import time
import logging

logger = logging.getLogger('algorithm_demo.bg_subtraction')

class BackgroundSubtractor:
    """
    Background subtraction algorithm implementation.
    
    This class demonstrates the steps of background subtraction:
    1. Convert frame to grayscale
    2. Apply Gaussian blur to reduce noise
    3. Apply background subtraction
    4. Apply threshold to get binary mask
    5. Apply morphological operations to remove noise
    6. Find contours and draw bounding boxes
    7. Calculate metrics
    """
    
    def __init__(self):
        """Initialize background subtractor."""
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
        
        # Parameters
        self.min_contour_area = 500  # Minimum contour area to be considered an object
        self.learning_rate = 0.01    # Learning rate for background subtraction
        self.threshold_value = 250   # Threshold value for binary mask
        self.kernel_size = 5         # Kernel size for morphological operations
        
        # Metrics
        self.total_objects = 0
        self.frame_count = 0
        self.confidence_sum = 0
        
        logger.info("Background subtractor initialized")
    
    def process(self, frame):
        """
        Process a frame with background subtraction.
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (result_frame, step_frames, metrics)
                - result_frame: Frame with bounding boxes
                - step_frames: Dictionary of intermediate frames for visualization
                - metrics: Dictionary of metrics
        """
        # Start timing
        start_time = time.time()
        
        # Create a copy of the frame
        original_frame = frame.copy()
        
        # Step 1: Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Apply Gaussian blur
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Step 3: Apply background subtraction
        fg_mask = self.bg_subtractor.apply(blurred_frame, learningRate=self.learning_rate)
        
        # Step 4: Apply threshold to get binary mask
        _, binary_mask = cv2.threshold(fg_mask, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # Step 5: Apply morphological operations
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        # Step 6: Find contours
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 7: Draw bounding boxes and calculate metrics
        result_frame = original_frame.copy()
        objects_detected = 0
        distances = []
        
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate confidence (based on contour area)
            confidence = min(1.0, cv2.contourArea(contour) / 10000)
            
            # Draw bounding box
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence
            cv2.putText(result_frame, f"{confidence:.2f}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Calculate center point
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Draw center point
            cv2.circle(result_frame, (center_x, center_y), 4, (0, 0, 255), -1)
            
            # Calculate distances to other objects
            for other_contour in contours:
                if contour is other_contour:
                    continue
                
                if cv2.contourArea(other_contour) < self.min_contour_area:
                    continue
                
                other_x, other_y, other_w, other_h = cv2.boundingRect(other_contour)
                other_center_x = other_x + other_w // 2
                other_center_y = other_y + other_h // 2
                
                # Calculate Euclidean distance
                distance = np.sqrt((center_x - other_center_x) ** 2 + (center_y - other_center_y) ** 2)
                distances.append(distance)
                
                # Draw line between centers if close enough
                if distance < 200:
                    cv2.line(result_frame, (center_x, center_y), 
                            (other_center_x, other_center_y), (255, 0, 0), 1)
                    
                    # Draw distance
                    mid_x = (center_x + other_center_x) // 2
                    mid_y = (center_y + other_center_y) // 2
                    cv2.putText(result_frame, f"{distance:.1f}px", (mid_x, mid_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            objects_detected += 1
            self.confidence_sum += confidence
        
        # Update metrics
        self.total_objects += objects_detected
        self.frame_count += 1
        
        # Calculate average distance
        avg_distance = np.mean(distances) if distances else 0
        
        # Create colored versions of masks for visualization
        fg_mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        binary_mask_colored = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        morphology_colored = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
        
        # Add step frames for visualization
        step_frames = {
            'original': original_frame,
            'grayscale': cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR),
            'blurred': cv2.cvtColor(blurred_frame, cv2.COLOR_GRAY2BGR),
            'fg_mask': fg_mask_colored,
            'binary_mask': binary_mask_colored,
            'morphology': morphology_colored,
            'result': result_frame
        }
        
        # Calculate metrics
        processing_time = time.time() - start_time
        avg_confidence = self.confidence_sum / max(1, self.total_objects)
        
        metrics = {
            "objects_detected": objects_detected,
            "total_objects": self.total_objects,
            "average_confidence": avg_confidence,
            "average_distance": avg_distance,
            "processing_time": processing_time
        }
        
        return result_frame, step_frames, metrics
