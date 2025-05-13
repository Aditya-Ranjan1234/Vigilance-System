import os
import sys
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('algorithm_demo.run')

def create_placeholder():
    """Create placeholder image for the video display."""
    try:
        from utils.create_placeholder import create_placeholder
        placeholder = create_placeholder()
        
        # Save the image
        output_dir = os.path.join(os.path.dirname(__file__), "static")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "placeholder.jpg")
        import cv2
        cv2.imwrite(output_path, placeholder)
        
        logger.info(f"Placeholder image created at {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating placeholder image: {str(e)}")
        return False

def check_videos():
    """Check for available videos."""
    # Check vigilance system videos directory
    vigilance_videos_path = os.path.join('..', 'vigilance_system', 'videos')
    if not os.path.exists(vigilance_videos_path):
        # Try alternative path
        vigilance_videos_path = os.path.join('..', 'videos')
        if not os.path.exists(vigilance_videos_path):
            # Try D:\Main EL\videos
            vigilance_videos_path = r'D:\Main EL\videos'
            if not os.path.exists(vigilance_videos_path):
                logger.warning("No videos directory found. Please add videos to the static/videos directory.")
                os.makedirs(os.path.join('static', 'videos'), exist_ok=True)
                return False
    
    # Find videos in vigilance system directory
    video_files = []
    for root, dirs, files in os.walk(vigilance_videos_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_files.append(file)
    
    if not video_files:
        logger.warning("No video files found in videos directory.")
        return False
    
    logger.info(f"Found {len(video_files)} video files: {', '.join(video_files)}")
    return True

def run_app():
    """Run the Flask application."""
    try:
        # Import and run the app
        from app import app, socketio
        
        logger.info("Starting Algorithm Visualization Demo...")
        socketio.run(app, host='0.0.0.0', port=5050, debug=True)
        
        return True
    except Exception as e:
        logger.error(f"Error running application: {str(e)}")
        return False

if __name__ == "__main__":
    # Create placeholder image
    create_placeholder()
    
    # Check for videos
    check_videos()
    
    # Run the app
    run_app()
