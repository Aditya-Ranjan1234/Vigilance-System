"""
Script to reorganize the Vigilance System directory structure.

This script fixes the nested directory structure and creates a clean,
properly organized project.
"""

import os
import shutil
import glob
from pathlib import Path

# Define the root directory
ROOT_DIR = Path('.')

# Define the target structure
STRUCTURE = {
    'vigilance_system': {
        '__init__.py': '',
        'alert': {
            '__init__.py': '',
        },
        'dashboard': {
            '__init__.py': '',
            'templates': {},
            'static': {
                'css': {},
                'js': {},
                'img': {},
                'audio': {},
            },
        },
        'detection': {
            '__init__.py': '',
        },
        'preprocessing': {
            '__init__.py': '',
        },
        'utils': {
            '__init__.py': '',
        },
        'video_acquisition': {
            '__init__.py': '',
        },
        'videos': {},
    },
    'tests': {},
    'examples': {},
}

# Search patterns for finding files in the current structure
FILE_PATTERNS = [
    # Main package files
    ('vigilance_system/__main__.py', 'vigilance_system/__main__.py'),

    # Alert module
    ('vigilance_system/alert/decision_maker.py', 'vigilance_system/alert/decision_maker.py'),
    ('vigilance_system/alert/notifier.py', 'vigilance_system/alert/notifier.py'),

    # Dashboard module
    ('vigilance_system/dashboard/app.py', 'vigilance_system/dashboard/app.py'),
    ('vigilance_system/dashboard/templates/index.html', 'vigilance_system/dashboard/templates/index.html'),
    ('vigilance_system/dashboard/templates/login.html', 'vigilance_system/dashboard/templates/login.html'),
    ('vigilance_system/dashboard/static/css/dashboard.css', 'vigilance_system/dashboard/static/css/dashboard.css'),
    ('vigilance_system/dashboard/static/js/dashboard.js', 'vigilance_system/dashboard/static/js/dashboard.js'),

    # Detection module
    ('vigilance_system/detection/model_loader.py', 'vigilance_system/detection/model_loader.py'),
    ('vigilance_system/detection/object_detector.py', 'vigilance_system/detection/object_detector.py'),

    # Preprocessing module
    ('vigilance_system/preprocessing/frame_extractor.py', 'vigilance_system/preprocessing/frame_extractor.py'),
    ('vigilance_system/preprocessing/video_stabilizer.py', 'vigilance_system/preprocessing/video_stabilizer.py'),

    # Utils module
    ('vigilance_system/utils/config.py', 'vigilance_system/utils/config.py'),
    ('vigilance_system/utils/logger.py', 'vigilance_system/utils/logger.py'),

    # Video acquisition module
    ('vigilance_system/video_acquisition/camera.py', 'vigilance_system/video_acquisition/camera.py'),
    ('vigilance_system/video_acquisition/stream_manager.py', 'vigilance_system/video_acquisition/stream_manager.py'),

    # Videos directory
    ('vigilance_system/videos/README.md', 'vigilance_system/videos/README.md'),

    # Tests
    ('tests/test_config.py', 'tests/test_config.py'),
    ('tests/test_camera.py', 'tests/test_camera.py'),
    ('tests/README.md', 'tests/README.md'),

    # Examples
    ('examples/simple_detection.py', 'examples/simple_detection.py'),
    ('examples/README.md', 'examples/README.md'),

    # Root files
    ('config.yaml', 'config.yaml'),
    ('requirements.txt', 'requirements.txt'),
    ('setup.py', 'setup.py'),
    ('setup.sh', 'setup.sh'),
    ('setup.bat', 'setup.bat'),
    ('download_sample_videos.py', 'download_sample_videos.py'),
    ('README.md', 'README.md'),
    ('COMPONENTS_GUIDE.md', 'COMPONENTS_GUIDE.md'),
    ('DEPENDENCIES.md', 'DEPENDENCIES.md'),
]

# Function to find files in the current structure
def find_files():
    """Find files in the current structure and map them to the target structure."""
    files_to_copy = []

    # Find video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
        video_files.extend(glob.glob(f'**/*{ext}', recursive=True))

    for video_file in video_files:
        filename = os.path.basename(video_file)
        files_to_copy.append((video_file, f'vigilance_system/videos/{filename}'))

    # Find other files using patterns
    for pattern, target in FILE_PATTERNS:
        # Try exact match first
        if os.path.exists(pattern):
            files_to_copy.append((pattern, target))
            continue

        # Try to find the file in any subdirectory
        matches = glob.glob(f'**/{os.path.basename(pattern)}', recursive=True)
        if matches:
            # Use the first match
            files_to_copy.append((matches[0], target))

    return files_to_copy

def create_directory_structure(base_dir, structure, current_path=None):
    """Create the directory structure."""
    if current_path is None:
        current_path = Path(base_dir)

    for name, content in structure.items():
        path = current_path / name

        if isinstance(content, dict):
            # Create directory
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

            # Recursively create subdirectories
            create_directory_structure(base_dir, content, path)
        else:
            # Create file with content
            with open(path, 'w') as f:
                f.write(content)
            print(f"Created file: {path}")

def copy_files(base_dir, files_to_copy):
    """Copy files from source to destination."""
    for src, dst in files_to_copy:
        src_path = Path(base_dir) / src
        dst_path = Path(base_dir) / dst

        # Skip if source doesn't exist
        if not os.path.exists(src_path):
            print(f"Warning: Source file not found: {src_path}")
            continue

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Copy the file
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {src} -> {dst}")

def clean_up(backup_dir):
    """Clean up the old directory structure."""
    # Move the old vigilance_system directory to backup
    if os.path.exists('vigilance_system/vigilance_system'):
        print("Moving nested vigilance_system directory to backup...")
        shutil.move('vigilance_system/vigilance_system', backup_dir)

    # Remove empty directories
    for root, dirs, files in os.walk('.', topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_path.startswith('./backup') or dir_path == './vigilance_system':
                continue

            try:
                if not os.listdir(dir_path):  # Check if directory is empty
                    os.rmdir(dir_path)
                    print(f"Removed empty directory: {dir_path}")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")

def main():
    """Main function."""
    # Create a backup directory
    backup_dir = ROOT_DIR / 'backup'
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Created backup directory: {backup_dir}")

    # Find files to copy
    files_to_copy = find_files()
    print(f"Found {len(files_to_copy)} files to copy")

    # Create the new directory structure
    create_directory_structure(ROOT_DIR, STRUCTURE)

    # Copy files
    copy_files(ROOT_DIR, files_to_copy)

    # Clean up
    clean_up(backup_dir)

    print("Reorganization complete!")

if __name__ == '__main__':
    main()
