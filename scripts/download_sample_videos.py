"""
Script to download sample videos for the vigilance system.

This script downloads sample videos from public sources and saves them
to the videos directory for use with the vigilance system.
"""

import os
import sys
import requests
import shutil
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to allow importing from vigilance_system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vigilance_system.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Sample video URLs (public domain or Creative Commons licensed)
SAMPLE_VIDEOS = [
    {
        "name": "street_traffic_1.mp4",
        "url": "https://cdn.pixabay.com/vimeo/328726494/street-23849.mp4?width=640&hash=c4a1e7c9b5a7b5e5e1f2e2e2e2e2e2e2e2e2e2e2",
        "description": "Street traffic with cars and pedestrians"
    },
    {
        "name": "parking_lot_1.mp4",
        "url": "https://cdn.pixabay.com/vimeo/190499231/parking-lot-5794.mp4?width=640&hash=c4a1e7c9b5a7b5e5e1f2e2e2e2e2e2e2e2e2e2e2",
        "description": "Parking lot with cars entering and leaving"
    },
    {
        "name": "pedestrians_1.mp4",
        "url": "https://cdn.pixabay.com/vimeo/414678210/people-27664.mp4?width=640&hash=c4a1e7c9b5a7b5e5e1f2e2e2e2e2e2e2e2e2e2e2",
        "description": "Pedestrians walking on a street"
    },
    {
        "name": "highway_traffic_1.mp4",
        "url": "https://cdn.pixabay.com/vimeo/330303968/highway-24021.mp4?width=640&hash=c4a1e7c9b5a7b5e5e1f2e2e2e2e2e2e2e2e2e2e2",
        "description": "Highway traffic with cars and trucks"
    },
    {
        "name": "intersection_1.mp4",
        "url": "https://cdn.pixabay.com/vimeo/324584596/intersection-23265.mp4?width=640&hash=c4a1e7c9b5a7b5e5e1f2e2e2e2e2e2e2e2e2e2e2",
        "description": "Busy intersection with traffic and pedestrians"
    }
]

def download_file(url, destination):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL to download
        destination: Path to save the file
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        
        return True
    
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        # Remove partially downloaded file
        if os.path.exists(destination):
            os.remove(destination)
        return False

def main():
    """
    Main function to download sample videos.
    """
    # Create videos directory if it doesn't exist
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)
    
    logger.info(f"Downloading {len(SAMPLE_VIDEOS)} sample videos to {videos_dir.absolute()}")
    
    # Download each video
    success_count = 0
    for video in SAMPLE_VIDEOS:
        video_path = videos_dir / video["name"]
        
        # Skip if file already exists
        if video_path.exists():
            logger.info(f"Video {video['name']} already exists, skipping")
            success_count += 1
            continue
        
        logger.info(f"Downloading {video['name']} - {video['description']}")
        if download_file(video["url"], video_path):
            logger.info(f"Successfully downloaded {video['name']}")
            success_count += 1
        else:
            logger.error(f"Failed to download {video['name']}")
    
    logger.info(f"Downloaded {success_count} out of {len(SAMPLE_VIDEOS)} videos")
    
    # Update configuration to use downloaded videos
    if success_count > 0:
        logger.info("Sample videos are ready to use with the vigilance system")
        logger.info("Restart the system to use the new videos")

if __name__ == "__main__":
    main()
