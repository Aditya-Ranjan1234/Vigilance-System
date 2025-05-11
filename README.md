# Camera-Only Vigilance System

A modular Python-based security monitoring system that uses only cameras and software analytics to detect, analyze, and alert on security threats in real-time.

## Features

- **Real-time Object Detection**: Detect people, vehicles, and other objects using state-of-the-art deep learning models
- **Video Preprocessing**: Extract frames, reduce noise, and stabilize video for improved detection accuracy
- **Intelligent Alerts**: Generate alerts based on customizable rules such as loitering detection and crowd monitoring
- **Multi-Camera Support**: Monitor multiple camera streams simultaneously
- **Web Dashboard**: View live camera feeds, detections, and alerts through a user-friendly web interface
- **Notification System**: Receive alerts via email or SMS
- **Modular Architecture**: Easily extend or customize any component of the system
- **Algorithm Visualizations**: Educational visualizations showing how each algorithm works and its time complexity
- **Multiple ML Models**: Support for various machine learning models including YOLOv5, SVM, KNN, and Naive Bayes
- **Network Simulation**: Visualization of routing algorithms and network performance metrics

## System Architecture

The system consists of the following components:

1. **Video Acquisition Module**: Handles camera streams and video ingestion
2. **Preprocessing Module**: Extracts frames, reduces noise, and stabilizes video
3. **Detection Module**: Uses deep learning models to detect objects and anomalies
4. **Alert Module**: Makes decisions based on detections and sends alerts
5. **Dashboard Module**: Provides a web interface for monitoring and configuration
6. **Algorithm Visualization Module**: Shows how different algorithms work with educational visualizations
7. **Network Simulation Module**: Simulates different routing algorithms and network configurations
8. **Machine Learning Module**: Implements various ML models for classification and analysis
9. **Utilities**: Configuration, logging, and other helper functions

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for real-time processing)
- OpenCV dependencies (on Linux, you may need to install `libgl1-mesa-glx`)

### Quick Setup

#### Windows

```bash
# Run the setup script
setup.bat

# Activate the virtual environment
venv\Scripts\activate

# Start the system
python -m vigilance_system
```

#### Linux/Mac

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh

# Activate the virtual environment
source venv/bin/activate

# Start the system
python -m vigilance_system
```

### Manual Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Configuration

The system is configured using a `config.yaml` file in the root directory. A default configuration file is provided, but you should modify it to match your setup.

### Camera Configuration

The system supports multiple camera types:

```yaml
cameras:
  # IP camera using RTSP protocol
  - name: front_door
    url: rtsp://username:password@192.168.1.100:554/stream1
    type: rtsp
    fps: 10  # Target frames per second to process
    resolution: [1280, 720]  # Width, Height

  # IP camera using HTTP/MJPEG protocol
  - name: back_yard
    url: http://192.168.1.101:8080/video
    type: http
    fps: 5
    resolution: [1920, 1080]

  # Local webcam
  - name: webcam
    url: 0  # Device ID (0 is usually the default camera)
    type: webcam
    fps: 30
    resolution: [640, 480]

  # Video file
  - name: store_entrance
    url: videos/store_entrance.mp4
    type: video
    fps: null  # null means use the video's native FPS
    loop: true  # Loop the video when it reaches the end
```

### Video File Support

If no cameras are configured, the system will automatically use video files from the `vigilance_system/videos` directory. This is useful for:

1. Testing the system without actual cameras
2. Analyzing pre-recorded surveillance footage
3. Demonstrations and presentations

To use this feature:

1. Download sample videos:
   ```bash
   # Download a single sample video
   python download_sample_videos.py

   # Download all sample videos
   python download_sample_videos.py --all
   ```

2. Or place your own video files in the `vigilance_system/videos` directory

The system will automatically detect and use these videos as camera sources. The system supports various video formats including MP4, AVI, MOV, MKV, WMV, and FLV.

**Note**: The system currently includes sample WhatsApp video files that demonstrate different scenarios like loitering and crowd detection.

### Detection Configuration

```yaml
detection:
  model: yolov5s  # Options: yolov5s, yolov5m, yolov5l, yolov5x
  device: cuda:0  # Use 'cpu' if no GPU is available
  confidence_threshold: 0.5
  nms_threshold: 0.45  # Non-maximum suppression threshold
  classes_of_interest: [0, 1, 2]  # person, bicycle, car in COCO dataset
```

### Alert Configuration

```yaml
alerts:
  motion_threshold: 0.2  # Minimum motion percentage to trigger alert
  person_loitering_time: 30  # Seconds a person must be present to trigger loitering alert
  notification:
    email:
      enabled: true
      recipients: [admin@example.com]
      smtp_server: smtp.gmail.com
      smtp_port: 587
      smtp_username: your_email@gmail.com
      smtp_password: your_app_password
```

## Usage

### Starting the System

```bash
# Start with default configuration
python -m vigilance_system

# Start with custom configuration file
python -m vigilance_system --config path/to/config.yaml

# Start with custom log level
python -m vigilance_system --log-level DEBUG

# Start with custom host and port
python -m vigilance_system --host 127.0.0.1 --port 8080
```

### Accessing the Dashboard

Open your browser and navigate to `http://localhost:5000` (or the custom host/port you specified).

Default login credentials (change these in the config file):
- Username: `admin`
- Password: `change_me_immediately`

The dashboard provides the following features:

- **Live Camera Feeds**: View all camera streams with detection overlays
- **Algorithm Selection**: Choose from different detection, tracking, classification, and analysis algorithms
- **Algorithm Visualization**: See how each algorithm works with educational visualizations
- **Network Configuration**: Adjust frame rate, resolution, and routing algorithms
- **Alert History**: View past alerts with images and details
- **System Status**: Monitor system performance and resource usage
- **View Modes**: Switch between grid view and single camera view

## Examples

Check out the `examples` directory for sample scripts demonstrating how to use different components of the system:

- `simple_detection.py`: Demonstrates object detection on a video file or camera stream

## Development

### Project Structure

```
Main EL/
├── vigilance_system/         # Main package
│   ├── __init__.py
│   ├── __main__.py
│   ├── video_acquisition/    # Camera and stream handling
│   │   ├── __init__.py
│   │   ├── camera.py
│   │   └── stream_manager.py
│   ├── preprocessing/        # Video preprocessing
│   │   ├── __init__.py
│   │   ├── frame_extractor.py
│   │   └── video_stabilizer.py
│   ├── detection/            # Object detection
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   └── object_detector.py
│   ├── alert/                # Alert generation
│   │   ├── __init__.py
│   │   ├── decision_maker.py
│   │   └── notifier.py
│   ├── alerts/               # Stored alert images and metadata
│   │   └── [alert files]
│   ├── dashboard/            # Web interface
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── templates/
│   │   └── static/
│   │       ├── css/          # Stylesheets
│   │       ├── js/           # JavaScript files
│   │       └── img/          # Images and icons
│   ├── algorithms/           # Algorithm implementations and visualizations
│   │   ├── __init__.py
│   │   ├── tracking/         # Tracking algorithms (centroid, kalman, etc.)
│   │   ├── classification/   # Classification algorithms (SVM, KNN, Naive Bayes)
│   │   └── analysis/         # Analysis algorithms (basic, weighted, fuzzy)
│   ├── network/              # Network simulation components
│   │   ├── __init__.py
│   │   ├── routing.py        # Routing algorithm implementations
│   │   └── metrics.py        # Network performance metrics
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logger.py
│   └── videos/               # Video files for testing
│       ├── README.md
│       └── [video files]
├── alerts/                   # Generated alert images and metadata
│   └── [alert files]
├── logs/                     # Log files
│   └── vigilance.log
├── models/                   # Downloaded ML models
│   └── [model files]
├── tests/                    # Test files
│   ├── test_config.py
│   ├── test_camera.py
│   └── README.md
├── algorithms/               # Algorithm visualizations
│   ├── README.md
│   ├── visualization.py
│   └── run_visualizations.py
├── examples/                 # Example scripts
│   ├── simple_detection.py
│   └── README.md
├── videos/                   # Additional video files
│   └── [video files]
├── config.yaml               # Configuration file
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
├── setup.sh                  # Linux/Mac setup script
├── setup.bat                 # Windows setup script
├── download_sample_videos.py # Script to download sample videos
└── README.md                 # This file
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=vigilance_system
```

### Adding New Features

1. **Adding a new camera type**:
   - Extend the `Camera` base class in `video_acquisition/camera.py`
   - Implement the required methods: `connect()`, `disconnect()`, and `read_frame()`
   - Update the `create_camera()` factory function to support the new camera type

2. **Adding a new detection model**:
   - Update the `ModelLoader` class in `detection/model_loader.py`
   - Implement a new loading method (e.g., `_load_new_model()`)
   - Update the `load_model()` method to support the new model type

3. **Adding a new alert type**:
   - Update the `DecisionMaker` class in `alert/decision_maker.py`
   - Implement a new check method (e.g., `_check_new_condition()`)
   - Update the `process_detections()` method to call the new check method

4. **Adding a new classification algorithm**:
   - Create a new class in `algorithms/classification/`
   - Implement the required methods: `train()`, `predict()`, and `evaluate()`
   - Update the `set_classifier_algorithm()` method in `decision_maker.py`
   - Add the new algorithm to the dashboard UI in `templates/index.html`

5. **Adding a new tracking algorithm**:
   - Create a new class in `algorithms/tracking/`
   - Implement the required methods: `update()`, `predict()`, and `get_tracks()`
   - Update the `reset_all_trackers()` method in `decision_maker.py`
   - Add the new algorithm to the dashboard UI in `templates/index.html`

6. **Adding a new routing algorithm**:
   - Create a new function in `network/routing.py`
   - Implement the routing logic and performance metrics
   - Update the routing visualization in `app.py`
   - Add the new algorithm to the dashboard UI in `templates/index.html`

## Troubleshooting

### Common Issues

1. **Camera connection failures**:
   - Check that the camera URL is correct
   - Ensure the camera is accessible from your network
   - Check username/password if authentication is required
   - If using IP cameras, try using video files instead for testing

2. **Video file issues**:
   - Make sure video files are in a supported format (MP4, AVI, MOV, MKV, WMV, FLV)
   - Check that the video files are in the correct directory (`vigilance_system/videos`)
   - Try downloading sample videos using the provided script: `python download_sample_videos.py --all`
   - If a video file won't play, try converting it to MP4 using a tool like FFmpeg

3. **Slow detection performance**:
   - Use a smaller model (e.g., yolov5s instead of yolov5x)
   - Reduce the resolution or frame rate in the camera configuration
   - Ensure you're using a GPU if available
   - If using video files, try reducing their resolution

4. **Dashboard not loading**:
   - Check that the Flask server is running
   - Verify the host and port settings
   - Check for any errors in the console output

5. **Algorithm changes not persisting**:
   - Make sure the configuration file is writable
   - Check the console for any error messages when changing algorithms
   - Restart the system if changes are not being applied
   - Verify that the algorithm selection is properly saved in the config file

6. **Visualization issues**:
   - Ensure your browser supports HTML5 Canvas and WebSockets
   - Try a different browser if visualizations are not displaying correctly
   - Check that the frame rate is not set too high for your system
   - Reduce the resolution if visualizations are slow or laggy

## Documentation

- [Quick Start Guide](QUICK_START.md): Get up and running quickly
- [System Overview](SYSTEM_OVERVIEW.md): Comprehensive overview of the system
- [Components Guide](COMPONENTS_GUIDE.md): Detailed information about each component
- [Dependencies](DEPENDENCIES.md): Information about the project dependencies

## License

MIT

## Acknowledgements

- [YOLOv5](https://github.com/ultralytics/yolov5) for object detection
- [OpenCV](https://opencv.org/) for image processing
- [Flask](https://flask.palletsprojects.com/) for the web dashboard
- [Socket.IO](https://socket.io/) for real-time communication
