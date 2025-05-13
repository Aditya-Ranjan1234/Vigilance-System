# Vigilance System Reorganization Summary

## Overview

The Vigilance System project has been successfully reorganized to improve the directory structure and create a clean, properly organized project.

## Latest Changes (2023)

The directory structure has been further improved with the following changes:

1. **Organized Files by Type**
   - Created dedicated directories for different types of files
   - Moved files to their appropriate directories
   - Reduced clutter in the root directory

2. **Current Project Structure**
   ```
   Main EL/
   ├── alerts/                     # Alert images and data
   │   ├── crowd/                  # Crowd-related alerts
   │   └── loitering/              # Loitering-related alerts
   ├── config/                     # Configuration files
   │   ├── config.yaml             # Main configuration
   │   ├── current_algorithm.txt   # Current algorithm setting
   │   └── nodes.json              # Network node configuration
   ├── docs/                       # Documentation files
   │   ├── COMPONENTS_GUIDE.md
   │   ├── DEPENDENCIES.md
   │   ├── QUICK_START.md
   │   ├── REORGANIZATION_SUMMARY.md
   │   ├── SYSTEM_OVERVIEW.md
   │   └── Vigilance_System_Project_Report.txt
   ├── logs/                       # Log files
   ├── models/                     # Model files
   │   └── yolov5s.pt              # YOLOv5 model
   ├── scripts/                    # Script files
   ├── tests/                      # Test files
   │   └── test_*.py               # Test scripts
   ├── utils/                      # Utility scripts
   │   ├── check_sklearn.py
   │   ├── download_sample_videos.py
   │   └── reorganize.py
   ├── videos/                     # Video files
   │   ├── samples/                # Sample videos
   │   ├── surveillance/           # Surveillance videos
   │   └── test/                   # Test videos
   ├── vigilance_system/           # Main package
   │   ├── __init__.py
   │   ├── __main__.py
   │   ├── algorithms/             # Algorithm implementations
   │   ├── dashboard/              # Dashboard module
   │   │   ├── __init__.py
   │   │   ├── app.py
   │   │   ├── templates/
   │   │   └── static/
   │   ├── detection/              # Detection module
   │   │   ├── __init__.py
   │   │   ├── model_loader.py
   │   │   └── object_detector.py
   │   ├── network/                # Network module
   │   ├── preprocessing/          # Preprocessing module
   │   │   ├── __init__.py
   │   │   ├── frame_extractor.py
   │   │   └── video_stabilizer.py
   │   ├── tracking/               # Tracking module
   │   ├── utils/                  # Utilities
   │   │   ├── __init__.py
   │   │   ├── config.py
   │   │   └── logger.py
   │   └── videos -> ../videos     # Symbolic link to videos directory
   ├── requirements.txt            # Dependencies
   ├── setup.py                    # Package setup
   └── README.md                   # Main documentation
   ```

## How to Run the System

The system can now be run with the following commands:

1. **Activate the virtual environment**:
   ```bash
   # On Windows
   venv\Scripts\activate

   # On Linux/Mac
   source venv/bin/activate
   ```

2. **Start the system**:
   ```bash
   python -m vigilance_system
   ```

3. **Access the dashboard**:
   Open your browser and navigate to `http://localhost:5000`

## Video File Support

The system now properly supports video files in the `videos` directory. You can:

1. Download sample videos:
   ```bash
   python utils/download_sample_videos.py --all
   ```

2. Or place your own video files in the appropriate subdirectory:
   - `videos/samples/` - For sample videos
   - `videos/surveillance/` - For surveillance videos
   - `videos/test/` - For test videos

The system will automatically detect and use these videos as camera sources if no IP cameras are configured.

## Configuration

Configuration files are now stored in the `config` directory:

1. `config/config.yaml` - Main configuration file
2. `config/nodes.json` - Network node configuration
3. `config/current_algorithm.txt` - Current algorithm setting

## Logs

Log files are now stored in the `logs` directory. This makes it easier to find and manage log files.

## Models

Model files are now stored in the `models` directory. The system will automatically load models from this directory when needed.
