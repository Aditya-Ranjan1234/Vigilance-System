"""
Video Stabilization Algorithm Visualization.

This module visualizes the video stabilization algorithm used in the Vigilance System.
The algorithm uses optical flow or feature matching to estimate and correct motion between frames.

Time Complexity: O(n) where n is the number of pixels in the frame
Space Complexity: O(n) for storing the optical flow or feature points
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Optional, Dict, Any
import os
import time

# Create a directory for saving visualizations
os.makedirs('visualizations', exist_ok=True)

class VideoStabilizationVisualizer:
    """Visualizer for the video stabilization algorithm."""
    
    def __init__(self, method: str = 'optical_flow', smoothing_radius: int = 15):
        """
        Initialize the video stabilization visualizer.
        
        Args:
            method: Stabilization method ('optical_flow' or 'feature_matching')
            smoothing_radius: Number of frames to consider for smoothing motion
        """
        self.method = method
        self.smoothing_radius = smoothing_radius
        self.prev_gray = None
        self.transforms = []
        self.smoothed_transforms = []
        
        # Parameters for feature matching
        self.feature_detector = cv2.SIFT_create()
        self.feature_matcher = cv2.BFMatcher()
        
        # Visualization setup
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle(f"Video Stabilization ({method.replace('_', ' ').title()})", fontsize=16)
        self.frames = []
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame through the stabilization algorithm and visualize the steps.
        
        Args:
            frame: Input frame to stabilize
            
        Returns:
            Stabilized frame
        """
        if frame is None:
            return None
        
        # Convert to grayscale for motion estimation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize with first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame
        
        # Estimate transform between frames
        if self.method == 'optical_flow':
            transform, visualization = self._estimate_transform_optical_flow(gray)
        else:  # feature_matching
            transform, visualization = self._estimate_transform_feature_matching(gray)
        
        # If transform estimation failed, return original frame
        if transform is None:
            return frame
        
        # Add to transforms list
        self.transforms.append(transform)
        
        # Calculate smoothed transform
        if len(self.transforms) < 2:
            smoothed_transform = transform
        else:
            smoothed_transform = self._smooth_transform()
        
        self.smoothed_transforms.append(smoothed_transform)
        
        # Apply smoothed transform to frame
        stabilized_frame = self._apply_transform(frame, smoothed_transform)
        
        # Visualize the process
        self._visualize_process(frame, stabilized_frame, visualization, transform, smoothed_transform)
        
        # Update previous frame
        self.prev_gray = gray
        
        return stabilized_frame
    
    def _estimate_transform_optical_flow(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Estimate transform between frames using optical flow.
        
        Args:
            gray: Current frame in grayscale
            
        Returns:
            Tuple of transformation matrix and visualization image
        """
        try:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Create visualization of optical flow
            hsv = np.zeros_like(gray, dtype=np.uint8)
            hsv = cv2.cvtColor(hsv, cv2.COLOR_GRAY2BGR)
            
            # Convert flow to polar coordinates
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Set hue according to the angle of optical flow
            hsv[..., 0] = ang * 180 / np.pi / 2
            
            # Set value according to the magnitude of optical flow
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert HSV to BGR for visualization
            flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Calculate translation
            h, w = flow.shape[:2]
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            dx = np.median(flow_x)
            dy = np.median(flow_y)
            
            # Create transformation matrix
            transform = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            return transform, flow_vis
            
        except Exception as e:
            print(f"Error estimating transform with optical flow: {str(e)}")
            return None, self.prev_gray
    
    def _estimate_transform_feature_matching(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Estimate transform between frames using feature matching.
        
        Args:
            gray: Current frame in grayscale
            
        Returns:
            Tuple of transformation matrix and visualization image
        """
        try:
            # Detect features
            prev_keypoints, prev_descriptors = self.feature_detector.detectAndCompute(self.prev_gray, None)
            curr_keypoints, curr_descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            if prev_descriptors is None or curr_descriptors is None or len(prev_descriptors) < 2 or len(curr_descriptors) < 2:
                print("Not enough features detected for matching")
                return None, self.prev_gray
            
            # Match features
            matches = self.feature_matcher.knnMatch(prev_descriptors, curr_descriptors, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) < 4:
                print("Not enough good matches for homography")
                return None, self.prev_gray
            
            # Create visualization of feature matches
            matches_vis = cv2.drawMatches(self.prev_gray, prev_keypoints, gray, curr_keypoints, 
                                         good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            # Extract matched keypoints
            prev_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            curr_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            transform, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
            return transform, matches_vis
            
        except Exception as e:
            print(f"Error estimating transform with feature matching: {str(e)}")
            return None, self.prev_gray
    
    def _smooth_transform(self) -> np.ndarray:
        """
        Smooth the transformation to reduce jitter.
        
        Returns:
            Smoothed transformation matrix
        """
        # Get the transforms to consider for smoothing
        n = min(len(self.transforms), self.smoothing_radius)
        recent_transforms = self.transforms[-n:]
        
        # Calculate mean of recent transforms
        mean_dx = np.mean([t[0, 2] for t in recent_transforms])
        mean_dy = np.mean([t[1, 2] for t in recent_transforms])
        
        # Get the latest transform
        latest_transform = self.transforms[-1].copy()
        
        # Apply smoothing to translation components
        latest_transform[0, 2] = mean_dx
        latest_transform[1, 2] = mean_dy
        
        return latest_transform
    
    def _apply_transform(self, frame: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Apply transformation to frame.
        
        Args:
            frame: Input frame
            transform: Transformation matrix to apply
            
        Returns:
            Transformed frame
        """
        h, w = frame.shape[:2]
        stabilized_frame = cv2.warpAffine(frame, transform, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return stabilized_frame
    
    def _visualize_process(self, original: np.ndarray, stabilized: np.ndarray, 
                          motion_vis: np.ndarray, transform: np.ndarray, 
                          smoothed_transform: np.ndarray) -> None:
        """
        Visualize the stabilization process.
        
        Args:
            original: Original frame
            stabilized: Stabilized frame
            motion_vis: Visualization of motion estimation
            transform: Original transformation matrix
            smoothed_transform: Smoothed transformation matrix
        """
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Convert BGR to RGB for matplotlib
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        stabilized_rgb = cv2.cvtColor(stabilized, cv2.COLOR_BGR2RGB)
        
        if motion_vis.ndim == 2:
            motion_vis_rgb = cv2.cvtColor(motion_vis, cv2.COLOR_GRAY2RGB)
        else:
            motion_vis_rgb = cv2.cvtColor(motion_vis, cv2.COLOR_BGR2RGB)
        
        # Display original and stabilized frames
        self.axes[0, 0].imshow(original_rgb)
        self.axes[0, 0].set_title("Original Frame")
        self.axes[0, 0].axis('off')
        
        self.axes[0, 1].imshow(stabilized_rgb)
        self.axes[0, 1].set_title("Stabilized Frame")
        self.axes[0, 1].axis('off')
        
        # Display motion estimation visualization
        self.axes[1, 0].imshow(motion_vis_rgb)
        self.axes[1, 0].set_title(f"Motion Estimation ({self.method.replace('_', ' ').title()})")
        self.axes[1, 0].axis('off')
        
        # Display transformation data
        self.axes[1, 1].set_title("Transformation Data")
        self.axes[1, 1].axis('off')
        
        # Plot the transformation values over time
        if len(self.transforms) > 1:
            # Extract dx and dy values
            dx_values = [t[0, 2] for t in self.transforms]
            dy_values = [t[1, 2] for t in self.transforms]
            smoothed_dx = [t[0, 2] for t in self.smoothed_transforms]
            smoothed_dy = [t[1, 2] for t in self.smoothed_transforms]
            
            # Create a subplot for the transformation values
            trans_ax = self.axes[1, 1].inset_axes([0.1, 0.1, 0.8, 0.8])
            trans_ax.plot(dx_values, label='Original dx', color='red', alpha=0.5)
            trans_ax.plot(dy_values, label='Original dy', color='blue', alpha=0.5)
            trans_ax.plot(smoothed_dx, label='Smoothed dx', color='red', linestyle='--')
            trans_ax.plot(smoothed_dy, label='Smoothed dy', color='blue', linestyle='--')
            trans_ax.set_xlabel('Frame')
            trans_ax.set_ylabel('Translation')
            trans_ax.legend(loc='upper right', fontsize='small')
            trans_ax.grid(True, alpha=0.3)
        
        # Add text with current transformation values
        transform_text = f"Original Transform:\ndx={transform[0, 2]:.2f}, dy={transform[1, 2]:.2f}\n\n"
        transform_text += f"Smoothed Transform:\ndx={smoothed_transform[0, 2]:.2f}, dy={smoothed_transform[1, 2]:.2f}"
        self.axes[1, 1].text(0.5, 0.5, transform_text, 
                            ha='center', va='center', 
                            transform=self.axes[1, 1].transAxes)
        
        # Capture the current state for animation
        self.fig.tight_layout()
        self.fig.canvas.draw()
        frame = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.frames.append(frame)
    
    def save_animation(self, filename: str = "video_stabilization", fps: int = 5) -> None:
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
        """Create an educational figure explaining the video stabilization algorithm."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Title
        fig.suptitle("Video Stabilization Algorithm", fontsize=16, fontweight='bold')
        
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
        Video Stabilization Algorithm
        
        The video stabilization algorithm reduces camera shake by estimating and correcting motion between frames.
        
        Two methods are supported:
        
        1. Optical Flow Method:
           - Calculates dense optical flow between consecutive frames
           - Estimates global motion using median flow values
           - Creates transformation matrix to counteract the motion
           - Smooths transformations over time to reduce jitter
        
        2. Feature Matching Method:
           - Detects key features in consecutive frames
           - Matches features using a ratio test to find correspondences
           - Estimates affine transformation between matched points
           - Smooths transformations over time to reduce jitter
        
        Time Complexity: O(n) where n is the number of pixels in the frame
        Space Complexity: O(n) for storing the optical flow or feature points
        
        Applications:
        • Surveillance video stabilization
        • Handheld camera shake reduction
        • Drone footage stabilization
        • Improving object detection accuracy on shaky video
        """
        
        ax2.text(0.05, 0.95, explanation, fontsize=12, 
                verticalalignment='top', horizontalalignment='left',
                transform=ax2.transAxes)
        ax2.axis('off')
        
        # Save the explanation
        plt.tight_layout()
        plt.savefig(os.path.join('visualizations', 'video_stabilization_explanation.png'))
        plt.close()
        print("Explanation saved to visualizations/video_stabilization_explanation.png")


def visualize_video_stabilization(video_path: str, method: str = 'optical_flow', 
                                 smoothing_radius: int = 15, max_frames: int = 100) -> None:
    """
    Visualize the video stabilization algorithm on a video file.
    
    Args:
        video_path: Path to the video file
        method: Stabilization method ('optical_flow' or 'feature_matching')
        smoothing_radius: Number of frames to consider for smoothing motion
        max_frames: Maximum number of frames to process
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Create the visualizer
    visualizer = VideoStabilizationVisualizer(method, smoothing_radius)
    
    # Process frames
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        stabilized = visualizer.process_frame(frame)
        
        frame_count += 1
    
    # Release the video capture
    cap.release()
    
    # Save the animation
    visualizer.save_animation(f"video_stabilization_{method}")
    
    # Create explanation
    visualizer.create_explanation()
    
    # Show the visualization
    visualizer.show()


if __name__ == "__main__":
    # Example usage
    video_path = "videos/sample_video.mp4"  # Replace with an actual video path
    
    # Visualize optical flow method
    visualize_video_stabilization(video_path, method='optical_flow', max_frames=50)
    
    # Visualize feature matching method
    visualize_video_stabilization(video_path, method='feature_matching', max_frames=50)
