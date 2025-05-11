# Vigilance System Examples

This directory contains example scripts demonstrating how to use the Vigilance System.

## Simple Object Detection

The `simple_detection.py` script demonstrates how to use the object detection module to detect objects in a video file or camera stream.

### Usage

```bash
# Activate virtual environment first
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows

# Run with webcam (camera index 0)
python examples/simple_detection.py --input 0

# Run with video file
python examples/simple_detection.py --input path/to/video.mp4

# Run with custom model and confidence threshold
python examples/simple_detection.py --input 0 --model yolov5m --confidence 0.6

# Save output to video file
python examples/simple_detection.py --input path/to/video.mp4 --output output.mp4
```

Press 'q' to exit the script.

## Adding More Examples

Feel free to add more example scripts to demonstrate other features of the Vigilance System, such as:

- Video stabilization
- Alert generation
- Custom detection rules
- Integration with other systems
