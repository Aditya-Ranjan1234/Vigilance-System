# Reorganization Plan for Vigilance System

## Current Issues
- Multiple nested vigilance_system directories
- Redundant directories and files
- Confusing structure

## Target Structure
```
Main EL/
├── vigilance_system/           # Main package
│   ├── __init__.py
│   ├── __main__.py
│   ├── alert/
│   │   ├── __init__.py
│   │   ├── decision_maker.py
│   │   └── notifier.py
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── templates/
│   │   └── static/
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   └── object_detector.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── frame_extractor.py
│   │   └── video_stabilizer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logger.py
│   ├── video_acquisition/
│   │   ├── __init__.py
│   │   ├── camera.py
│   │   └── stream_manager.py
│   └── videos/                 # Directory for video files
│       └── README.md
├── tests/                      # Test directory
│   ├── test_config.py
│   ├── test_camera.py
│   └── README.md
├── examples/                   # Example scripts
│   ├── simple_detection.py
│   └── README.md
├── config.yaml                 # Configuration file
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
├── setup.sh                    # Linux/Mac setup script
├── setup.bat                   # Windows setup script
├── download_sample_videos.py   # Script to download sample videos
└── README.md                   # Main documentation
```

## Steps to Reorganize
1. Create missing __init__.py files
2. Move files to their correct locations
3. Remove redundant directories
4. Update imports if necessary
5. Test the reorganized structure
