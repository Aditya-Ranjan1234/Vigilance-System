#!/usr/bin/env python
"""
Script to download sample videos for the Vigilance System.

This script downloads sample surveillance videos from public sources
and places them in the videos directory.
"""

import os
import sys
import shutil
import argparse
import requests
# from pathlib import Path  # Uncomment if needed for future enhancements
from tqdm import tqdm

# Sample videos URLs (public domain or Creative Commons)
SAMPLE_VIDEOS = {
    "store_surveillance.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4",
    "parking_lot.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/parking-lot-detection.mp4",
    "pedestrians.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
}

def download_file(url, destination):
    """
    Download a file with progress bar.

    Args:
        url: URL to download
        destination: Destination file path

    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        with open(destination, 'wb') as f, tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)

        return True

    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        # Remove partially downloaded file
        if os.path.exists(destination):
            os.remove(destination)
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download sample videos for the Vigilance System")
    parser.add_argument('--all', action='store_true', help="Download all sample videos")
    parser.add_argument('--video', choices=list(SAMPLE_VIDEOS.keys()), help="Download a specific video")

    args = parser.parse_args()

    # Determine which videos to download
    videos_to_download = {}
    if args.all:
        videos_to_download = SAMPLE_VIDEOS
    elif args.video:
        videos_to_download = {args.video: SAMPLE_VIDEOS[args.video]}
    else:
        # Default: download the first video
        key = next(iter(SAMPLE_VIDEOS))
        videos_to_download = {key: SAMPLE_VIDEOS[key]}

    # Create videos directory if it doesn't exist
    videos_dir = os.path.join('vigilance_system', 'videos')
    os.makedirs(videos_dir, exist_ok=True)

    # Also create the root videos directory if it doesn't exist
    root_videos_dir = 'videos'
    os.makedirs(root_videos_dir, exist_ok=True)

    # Download videos
    success_count = 0
    for filename, url in videos_to_download.items():
        # Download to vigilance_system/videos directory
        destination = os.path.join(videos_dir, filename)

        print(f"Downloading {filename}...")
        if download_file(url, destination):
            success_count += 1

            # Copy to root videos directory
            root_destination = os.path.join(root_videos_dir, filename)
            try:
                shutil.copy2(destination, root_destination)
                print(f"Copied {filename} to {root_videos_dir}")
            except Exception as e:
                print(f"Error copying to root videos directory: {str(e)}")

    # Print summary
    print(f"\nDownloaded {success_count} of {len(videos_to_download)} videos to {videos_dir}")
    print(f"Videos are available in both {videos_dir} and {root_videos_dir} directories")

    return 0

if __name__ == '__main__':
    sys.exit(main())
