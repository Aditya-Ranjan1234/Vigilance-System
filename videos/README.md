# Video Files Directory

This directory contains video files used by the Vigilance System for testing and demonstration purposes.

## Adding Videos

To add new videos to the system:

1. Simply place your video files in this directory or any subdirectory
2. The system will automatically detect and load all video files with supported extensions
3. Videos will appear in the dashboard with their filename as the camera name

## Supported Video Formats

The following video formats are supported:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)

## Organizing Videos

You can organize your videos into subdirectories for better management:

```
videos/
├── traffic/
│   ├── highway.mp4
│   └── intersection.mp4
├── surveillance/
│   ├── parking_lot.mp4
│   └── building_entrance.mp4
└── test_videos/
    ├── test1.mp4
    └── test2.mp4
```

The system will recursively scan all subdirectories and load all video files.

## Video Properties

For optimal performance:
- Resolution: 640x480 or higher
- Frame rate: 25-30 FPS
- Codec: H.264 or H.265
- Duration: At least 10 seconds (videos will loop automatically)
