"""
Script to generate more realistic test videos for the vigilance system.

This script creates test videos with human silhouettes that can be detected
by the YOLOv5 model for testing the tracking functionality.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import random

# Add parent directory to path to allow importing from vigilance_system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vigilance_system.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Human silhouette template
def create_human_silhouette(height=120):
    """Create a human-like silhouette that YOLOv5 might detect as a person."""
    # Calculate width based on typical human proportions
    width = int(height * 0.4)
    
    # Create blank silhouette
    silhouette = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Head (circle)
    head_radius = int(width * 0.4)
    head_center = (width // 2, head_radius + 5)
    cv2.circle(silhouette, head_center, head_radius, (200, 200, 200), -1)
    
    # Body (rectangle)
    body_top = head_center[1] + head_radius - 5
    body_height = int(height * 0.6)
    body_width = int(width * 0.8)
    body_left = (width - body_width) // 2
    cv2.rectangle(silhouette, 
                 (body_left, body_top), 
                 (body_left + body_width, body_top + body_height), 
                 (200, 200, 200), -1)
    
    # Legs (two rectangles)
    leg_top = body_top + body_height
    leg_height = height - leg_top - 5
    leg_width = int(body_width * 0.3)
    
    # Left leg
    left_leg_left = body_left + int(body_width * 0.1)
    cv2.rectangle(silhouette,
                 (left_leg_left, leg_top),
                 (left_leg_left + leg_width, leg_top + leg_height),
                 (200, 200, 200), -1)
    
    # Right leg
    right_leg_left = body_left + body_width - leg_width - int(body_width * 0.1)
    cv2.rectangle(silhouette,
                 (right_leg_left, leg_top),
                 (right_leg_left + leg_width, leg_top + leg_height),
                 (200, 200, 200), -1)
    
    # Arms (two rectangles)
    arm_top = body_top + int(body_height * 0.1)
    arm_height = int(body_height * 0.4)
    arm_width = int(width * 0.2)
    
    # Left arm
    cv2.rectangle(silhouette,
                 (body_left - arm_width + 5, arm_top),
                 (body_left + 5, arm_top + arm_height),
                 (200, 200, 200), -1)
    
    # Right arm
    cv2.rectangle(silhouette,
                 (body_left + body_width - 5, arm_top),
                 (body_left + body_width + arm_width - 5, arm_top + arm_height),
                 (200, 200, 200), -1)
    
    return silhouette

def create_people_walking_video(output_path, width=640, height=480, duration=20, fps=30, num_people=3):
    """
    Create a video with multiple people walking around.
    
    Args:
        output_path: Path to save the video
        width: Frame width
        height: Frame height
        duration: Video duration in seconds
        fps: Frames per second
        num_people: Number of people to simulate
    """
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate total frames
    total_frames = duration * fps
    
    # Create background (simple gradient)
    background = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # Create a gradient background
            background[y, x] = [
                int(100 + (y / height) * 50),
                int(100 + (x / width) * 50),
                int(100 + ((x+y) / (width+height)) * 50)
            ]
    
    # Add some horizontal lines to simulate a floor
    for y in range(height//2, height, 40):
        cv2.line(background, (0, y), (width, y), (70, 70, 70), 1)
    
    # Create people
    people = []
    for i in range(num_people):
        # Random size (height)
        person_height = random.randint(80, 150)
        
        # Create silhouette
        silhouette = create_human_silhouette(person_height)
        
        # Random position
        x = random.randint(0, width - silhouette.shape[1])
        y = random.randint(0, height - silhouette.shape[0])
        
        # Random speed
        speed_x = random.randint(-3, 3)
        if speed_x == 0:
            speed_x = 1
        speed_y = random.randint(-2, 2)
        
        people.append({
            'silhouette': silhouette,
            'x': x,
            'y': y,
            'speed_x': speed_x,
            'speed_y': speed_y,
            'width': silhouette.shape[1],
            'height': silhouette.shape[0]
        })
    
    logger.info(f"Generating video with {num_people} people walking: {output_path}")
    
    # Generate frames
    for i in range(total_frames):
        # Start with background
        frame = background.copy()
        
        # Update and draw people
        for person in people:
            # Update position
            person['x'] += person['speed_x']
            person['y'] += person['speed_y']
            
            # Bounce off walls
            if person['x'] <= 0 or person['x'] + person['width'] >= width:
                person['speed_x'] = -person['speed_x']
                person['x'] += person['speed_x']
            
            if person['y'] <= 0 or person['y'] + person['height'] >= height:
                person['speed_y'] = -person['speed_y']
                person['y'] += person['speed_y']
            
            # Draw person on frame
            x, y = person['x'], person['y']
            silhouette = person['silhouette']
            
            # Create a mask for the silhouette
            mask = silhouette.sum(axis=2) > 0
            
            # Place silhouette on frame
            for c in range(3):  # RGB channels
                frame[y:y+silhouette.shape[0], x:x+silhouette.shape[1], c][mask] = silhouette[:,:,c][mask]
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    # Release resources
    out.release()
    logger.info(f"Video generated: {output_path}")

def create_loitering_video(output_path, width=640, height=480, duration=40, fps=30):
    """
    Create a video with people loitering in specific areas.
    
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
    
    # Create background (simple gradient with some structure)
    background = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # Create a gradient background
            background[y, x] = [
                int(80 + (y / height) * 70),
                int(80 + (x / width) * 70),
                int(80 + ((x+y) / (width+height)) * 70)
            ]
    
    # Add some structure to background (building-like)
    cv2.rectangle(background, (50, 50), (width-50, height-50), (120, 120, 120), 2)
    cv2.rectangle(background, (150, 100), (300, 380), (100, 100, 100), 2)
    cv2.rectangle(background, (350, 100), (500, 380), (100, 100, 100), 2)
    
    # Create loitering person (stays in one area)
    loiterer_height = 120
    loiterer = create_human_silhouette(loiterer_height)
    loiterer_x = width // 2 - loiterer.shape[1] // 2
    loiterer_y = height // 2 - loiterer.shape[0] // 2
    
    # Create walking people
    walkers = []
    for i in range(2):
        # Random size (height)
        person_height = random.randint(80, 150)
        
        # Create silhouette
        silhouette = create_human_silhouette(person_height)
        
        # Random position
        x = random.randint(0, width - silhouette.shape[1])
        y = random.randint(0, height - silhouette.shape[0])
        
        # Random speed
        speed_x = random.randint(1, 3) * (1 if i % 2 == 0 else -1)
        speed_y = random.randint(-1, 1)
        
        walkers.append({
            'silhouette': silhouette,
            'x': x,
            'y': y,
            'speed_x': speed_x,
            'speed_y': speed_y,
            'width': silhouette.shape[1],
            'height': silhouette.shape[0]
        })
    
    logger.info(f"Generating video with loitering person: {output_path}")
    
    # Generate frames
    for i in range(total_frames):
        # Start with background
        frame = background.copy()
        
        # Draw loitering person (with small random movement)
        if i % 30 == 0:  # Every second, make a small movement
            loiterer_x += random.randint(-5, 5)
            loiterer_y += random.randint(-5, 5)
            
            # Keep within bounds
            loiterer_x = max(0, min(width - loiterer.shape[1], loiterer_x))
            loiterer_y = max(0, min(height - loiterer.shape[0], loiterer_y))
        
        # Draw loiterer on frame
        mask = loiterer.sum(axis=2) > 0
        for c in range(3):
            frame[loiterer_y:loiterer_y+loiterer.shape[0], 
                  loiterer_x:loiterer_x+loiterer.shape[1], c][mask] = loiterer[:,:,c][mask]
        
        # Update and draw walking people
        for person in walkers:
            # Update position
            person['x'] += person['speed_x']
            person['y'] += person['speed_y']
            
            # Bounce off walls or wrap around
            if person['x'] <= 0:
                person['x'] = width - person['width']
            elif person['x'] + person['width'] >= width:
                person['x'] = 0
            
            if person['y'] <= 0 or person['y'] + person['height'] >= height:
                person['speed_y'] = -person['speed_y']
                person['y'] += person['speed_y']
            
            # Draw person on frame
            x, y = person['x'], person['y']
            silhouette = person['silhouette']
            
            # Create a mask for the silhouette
            mask = silhouette.sum(axis=2) > 0
            
            # Place silhouette on frame
            for c in range(3):  # RGB channels
                frame[y:y+silhouette.shape[0], x:x+silhouette.shape[1], c][mask] = silhouette[:,:,c][mask]
        
        # Add frame number and timer
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        seconds = i / fps
        cv2.putText(frame, f"Time: {seconds:.1f}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    # Release resources
    out.release()
    logger.info(f"Video generated: {output_path}")

def main():
    """
    Main function to generate realistic test videos.
    """
    # Create videos directory if it doesn't exist
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)
    
    logger.info(f"Generating realistic test videos in {videos_dir.absolute()}")
    
    # Generate videos
    create_people_walking_video(str(videos_dir / "Camera 1.mp4"), duration=30, num_people=5)
    create_loitering_video(str(videos_dir / "Camera 2.mp4"), duration=40)
    create_people_walking_video(str(videos_dir / "Camera 3.mp4"), duration=30, num_people=8)
    
    logger.info("Realistic test videos generated successfully")
    logger.info("Restart the system to use the new videos")

if __name__ == "__main__":
    main()
