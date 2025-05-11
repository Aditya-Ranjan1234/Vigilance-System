"""
Script to generate test videos for the vigilance system.

This script creates synthetic test videos with moving objects
for use with the vigilance system.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path to allow importing from vigilance_system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vigilance_system.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def create_moving_rectangle_video(output_path, width=640, height=480, duration=10, fps=30):
    """
    Create a video with a moving rectangle.
    
    Args:
        output_path: Path to save the video
        width: Frame width
        height: Frame height
        duration: Video duration in seconds
        fps: Frames per second
    """
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate total frames
    total_frames = duration * fps
    
    # Rectangle properties
    rect_width, rect_height = 60, 40
    rect_color = (0, 0, 255)  # Red
    
    # Initial position
    x, y = 0, height // 2
    
    # Speed
    speed_x = 5
    
    logger.info(f"Generating video with moving rectangle: {output_path}")
    
    # Generate frames
    for i in range(total_frames):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update position
        x += speed_x
        
        # Bounce off walls
        if x <= 0 or x + rect_width >= width:
            speed_x = -speed_x
            x += speed_x
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + rect_width, y + rect_height), rect_color, -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    # Release resources
    out.release()
    logger.info(f"Video generated: {output_path}")

def create_moving_circles_video(output_path, width=640, height=480, duration=10, fps=30, num_circles=5):
    """
    Create a video with multiple moving circles.
    
    Args:
        output_path: Path to save the video
        width: Frame width
        height: Frame height
        duration: Video duration in seconds
        fps: Frames per second
        num_circles: Number of circles to draw
    """
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate total frames
    total_frames = duration * fps
    
    # Circle properties
    circles = []
    colors = []
    
    # Generate random circles
    for _ in range(num_circles):
        # Random position
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        
        # Random radius
        radius = np.random.randint(20, 40)
        
        # Random speed
        speed_x = np.random.randint(-5, 5)
        speed_y = np.random.randint(-5, 5)
        
        # Ensure non-zero speed
        if speed_x == 0:
            speed_x = 3
        if speed_y == 0:
            speed_y = 2
        
        circles.append([x, y, radius, speed_x, speed_y])
        
        # Random color
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        )
        colors.append(color)
    
    logger.info(f"Generating video with {num_circles} moving circles: {output_path}")
    
    # Generate frames
    for i in range(total_frames):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update and draw circles
        for j, (x, y, radius, speed_x, speed_y) in enumerate(circles):
            # Update position
            x += speed_x
            y += speed_y
            
            # Bounce off walls
            if x - radius <= 0 or x + radius >= width:
                speed_x = -speed_x
                x += speed_x
            
            if y - radius <= 0 or y + radius >= height:
                speed_y = -speed_y
                y += speed_y
            
            # Update circle
            circles[j] = [x, y, radius, speed_x, speed_y]
            
            # Draw circle
            cv2.circle(frame, (x, y), radius, colors[j], -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    # Release resources
    out.release()
    logger.info(f"Video generated: {output_path}")

def create_grid_pattern_video(output_path, width=640, height=480, duration=10, fps=30):
    """
    Create a video with a moving grid pattern.
    
    Args:
        output_path: Path to save the video
        width: Frame width
        height: Frame height
        duration: Video duration in seconds
        fps: Frames per second
    """
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate total frames
    total_frames = duration * fps
    
    # Grid properties
    grid_size = 40
    offset_x, offset_y = 0, 0
    speed_x, speed_y = 1, 1
    
    logger.info(f"Generating video with moving grid pattern: {output_path}")
    
    # Generate frames
    for i in range(total_frames):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update offset
        offset_x = (offset_x + speed_x) % grid_size
        offset_y = (offset_y + speed_y) % grid_size
        
        # Draw grid
        for x in range(-grid_size + offset_x, width + grid_size, grid_size):
            cv2.line(frame, (x, 0), (x, height), (100, 100, 100), 1)
        
        for y in range(-grid_size + offset_y, height + grid_size, grid_size):
            cv2.line(frame, (0, y), (width, y), (100, 100, 100), 1)
        
        # Draw moving object
        obj_x = int(width/2 + width/4 * np.sin(i * 0.05))
        obj_y = int(height/2 + height/4 * np.cos(i * 0.05))
        
        cv2.rectangle(frame, (obj_x-30, obj_y-30), (obj_x+30, obj_y+30), (0, 255, 0), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    # Release resources
    out.release()
    logger.info(f"Video generated: {output_path}")

def main():
    """
    Main function to generate test videos.
    """
    # Create videos directory if it doesn't exist
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)
    
    logger.info(f"Generating test videos in {videos_dir.absolute()}")
    
    # Generate videos
    create_moving_rectangle_video(str(videos_dir / "Camera 1.mp4"), duration=20)
    create_moving_circles_video(str(videos_dir / "Camera 2.mp4"), duration=20, num_circles=8)
    create_grid_pattern_video(str(videos_dir / "Camera 3.mp4"), duration=20)
    
    logger.info("Test videos generated successfully")
    logger.info("Restart the system to use the new videos")

if __name__ == "__main__":
    main()
