# Camera-Only Vigilance System

A modular Python-based security monitoring system that uses only cameras and software analytics to detect, analyze, and alert on security threats in real-time.

## Features

- **Multiple Object Detection Algorithms**: Choose from Background Subtraction, MOG2, KNN, or SVM Classifier for efficient performance
- **Advanced Object Tracking**: Implement KLT Tracker, Kalman Filter, or Optical Flow tracking for robust object persistence
- **Intelligent Loitering Detection**: Select from rule-based, timer threshold, or decision tree algorithms
- **Crowd Detection Methods**: Utilize blob counting, contour counting, or K-means clustering approaches for crowd monitoring
- **Video Preprocessing Options**: Apply feature matching, ORB, SIFT, or affine transform stabilization techniques
- **Algorithm Analysis Dashboard**: Compare and analyze the performance of different algorithms in real-time
- **Multi-Camera Support**: Monitor multiple camera streams simultaneously
- **Web Dashboard**: View live camera feeds, detections, and alerts through a user-friendly web interface
- **Notification System**: Receive alerts via email or SMS
- **Modular Architecture**: Easily extend or customize any component of the system

## System Architecture

The system consists of the following components:

1. **Video Acquisition Module**: Handles camera streams and video ingestion
2. **Preprocessing Module**: Extracts frames, reduces noise, and stabilizes video
3. **Detection Module**: Uses deep learning models to detect objects and anomalies
4. **Alert Module**: Makes decisions based on detections and sends alerts
5. **Dashboard Module**: Provides a web interface for monitoring and configuration
6. **Utilities**: Configuration, logging, and other helper functions

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
  algorithm: background_subtraction  # Options: background_subtraction, mog2, knn, svm_classifier
  confidence_threshold: 0.5
  classes_of_interest: [0]  # 0 = person
  algorithms:
    background_subtraction:
      min_contour_area: 500
    mog2:
      history: 500
      var_threshold: 16
      learning_rate: 0.01
    knn:
      history: 500
      dist2_threshold: 400.0
      learning_rate: 0.01
    svm_classifier:
      scale: 1.05
      win_stride: [8, 8]
      padding: [16, 16]
```

### Alert Configuration

```yaml
alerts:
  # Loitering Detection
  loitering:
    algorithm: rule_based  # Options: rule_based, timer_threshold, decision_tree
    algorithms:
      rule_based:
        min_loitering_time: 10.0  # Seconds a person must be present to trigger loitering alert
        max_movement_distance: 50  # Maximum distance to move while still considered loitering
        grid_size: 50  # Size of grid cells for spatial analysis
      timer_threshold:
        min_loitering_time: 10.0  # Seconds a person must be present to trigger loitering alert
        zone_size: 100  # Size of zones for spatial analysis
      decision_tree:
        min_trajectory_length: 20  # Minimum trajectory length to analyze
        max_speed_threshold: 10.0  # Maximum speed to consider loitering
        min_direction_changes: 3  # Minimum number of direction changes
        min_time_in_area: 8.0  # Minimum time in seconds to consider loitering

  # Crowd Detection
  crowd:
    algorithm: blob_counting  # Options: blob_counting, contour_counting, kmeans_clustering
    algorithms:
      blob_counting:
        crowd_threshold: 3  # Minimum number of people to consider a crowd
        proximity_threshold: 100  # Maximum distance between people to be considered in the same group
        min_group_size: 3  # Minimum number of people in a group to be considered a crowd
      contour_counting:
        crowd_threshold: 3  # Minimum number of people to consider a crowd
        density_threshold: 0.5  # Threshold for density map
        min_contour_area: 1000  # Minimum contour area to consider a crowd
        kernel_size: 50  # Size of Gaussian kernel for density map
      kmeans_clustering:
        crowd_threshold: 3  # Minimum number of people to consider a crowd
        max_clusters: 5  # Maximum number of clusters to consider
        min_cluster_size: 3  # Minimum number of people in a cluster to be considered a crowd
        max_cluster_radius: 100  # Maximum radius of a cluster to be considered a crowd

  # Motion Detection
  motion:
    threshold: 0.2  # Minimum motion percentage to trigger alert

  # Notification
  notification:
    email:
      enabled: true
      recipients: [admin@example.com]
      smtp_server: smtp.gmail.com
      smtp_port: 587
      smtp_username: your_email@gmail.com
      smtp_password: your_app_password
    sms:
      enabled: false
      phone_numbers: [+1234567890]
      service: twilio  # Options: twilio, aws_sns
```

### Preprocessing Configuration

```yaml
preprocessing:
  algorithm: feature_matching  # Options: feature_matching, orb, sift, affine_transform
  denoising:
    enabled: true
    strength: 10  # Higher values = more denoising
  stabilization:
    enabled: true
    smoothing_radius: 30  # Frames to consider for stabilization
  algorithms:
    feature_matching:
      max_corners: 200
      quality_level: 0.01
      min_distance: 30
      block_size: 3
    orb:
      max_features: 500
      match_threshold: 0.7
    sift:
      max_features: 500
      match_threshold: 0.7
    affine_transform:
      grid_size: 8
      window_size: 15
```

### Tracking Configuration

```yaml
tracking:
  algorithm: klt_tracker  # Options: klt_tracker, kalman_filter, optical_flow
  algorithms:
    klt_tracker:
      max_corners: 100
      quality_level: 0.3
      min_distance: 7
      block_size: 7
    kalman_filter:
      max_age: 10
      min_hits: 3
      iou_threshold: 0.3
    optical_flow:
      max_corners: 200
      quality_level: 0.01
      min_distance: 30
```

### Analysis Dashboard Configuration

```yaml
analysis_dashboard:
  enabled: true
  port: 5001  # Separate port for the analysis dashboard
  metrics:
    # Object Detection Metrics
    detection:
      enabled: true
      metrics: [fps, map, precision, recall]
      update_interval: 1  # Update interval in seconds

    # Object Tracking Metrics
    tracking:
      enabled: true
      metrics: [id_switches, mota, motp, mostly_tracked, mostly_lost]
      update_interval: 1

    # Loitering Detection Metrics
    loitering:
      enabled: true
      metrics: [true_positives, false_positives, false_negatives, precision, recall]
      update_interval: 5

    # Crowd Detection Metrics
    crowd:
      enabled: true
      metrics: [mae, mse, accuracy, precision, recall]
      update_interval: 5

    # Video Preprocessing Metrics
    preprocessing:
      enabled: true
      metrics: [stability_score, processing_time]
      update_interval: 1

  # Visualization Options
  visualization:
    charts: [line, bar, scatter, heatmap]
    real_time_updates: true
    history_length: 100  # Number of data points to keep in history
    export_formats: [csv, json, png]
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

# Run with specific algorithms
python -m vigilance_system --detection-algorithm background_subtraction --tracking-algorithm klt_tracker --loitering-algorithm rule_based --crowd-algorithm blob_counting --preprocessing-algorithm feature_matching

# Enable the analysis dashboard
python -m vigilance_system --enable-analysis

# Specify ports for both dashboards
python -m vigilance_system --port 5000 --analysis-port 5001 --enable-analysis
```

### Accessing the Dashboards

#### Main Dashboard
Open your browser and navigate to `http://localhost:5000` (or the custom host/port you specified).

Default login credentials (change these in the config file):
- Username: `admin`
- Password: `change_me_immediately`

#### Analysis Dashboard
If you've enabled the analysis dashboard, open your browser and navigate to `http://localhost:5001` (or the custom analysis port you specified).

The analysis dashboard provides:
- Real-time performance metrics for all algorithms
- Comparative analysis between different detection, tracking, and alert algorithms
- Visualization tools for algorithm efficiency
- Export functionality for metrics data

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

2. **Adding a new detection algorithm**:
   - Create a new class in `detection/algorithms/` that inherits from `BaseDetector`
   - Implement the required methods: `get_name()`, `load_model()`, and `detect()`
   - Update the `ObjectDetector` class in `detection/object_detector.py` to support the new algorithm

3. **Adding a new tracking algorithm**:
   - Create a new class in `tracking/algorithms/` that inherits from `BaseTracker`
   - Implement the required methods: `get_name()` and `update()`
   - Update the `ObjectTracker` class in `tracking/object_tracker.py` to support the new algorithm

4. **Adding a new loitering detection algorithm**:
   - Create a new class in `alert/algorithms/` that inherits from `BaseLoiteringDetector`
   - Implement the required methods: `get_name()` and `update()`
   - Update the `LoiteringDetector` class in `alert/loitering_detector.py` to support the new algorithm

5. **Adding a new crowd detection algorithm**:
   - Create a new class in `alert/algorithms/` that inherits from `BaseCrowdDetector`
   - Implement the required methods: `get_name()` and `update()`
   - Update the `CrowdDetector` class in `alert/crowd_detector.py` to support the new algorithm

6. **Adding a new video stabilization algorithm**:
   - Create a new class in `preprocessing/algorithms/` that inherits from `BaseStabilizer`
   - Implement the required methods: `get_name()` and `stabilize()`
   - Update the `VideoStabilizer` class in `preprocessing/video_stabilizer.py` to support the new algorithm

7. **Adding a new metric to the analysis dashboard**:
   - Update the `metrics_collector.py` file in `analysis/` to collect the new metric
   - Add the metric to the configuration in `config.yaml`
   - Update the dashboard UI in `analysis/templates/` and `analysis/static/js/` to display the new metric

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
   - Try different detection algorithms (SSD might be faster than Faster R-CNN)
   - Reduce the resolution or frame rate in the camera configuration
   - Ensure you're using a GPU if available
   - If using video files, try reducing their resolution

4. **Dashboard not loading**:
   - Check that the Flask server is running
   - Verify the host and port settings
   - Check for any errors in the console output

5. **Algorithm-specific issues**:
   - **Deep learning algorithms**: If TensorFlow-based algorithms fail, ensure TensorFlow is installed correctly or try using alternative algorithms
   - **Feature-based stabilization**: If SIFT features are not available, the system will fall back to ORB features
   - **DeepSORT tracking**: Requires additional models for feature extraction; if these fail to load, the system will use simpler appearance features

6. **Analysis dashboard issues**:
   - If charts are not updating, check browser console for errors
   - Ensure WebSocket connections are not blocked by firewalls
   - If metrics are missing, verify that the corresponding algorithm is enabled and running

## Documentation

- [Quick Start Guide](QUICK_START.md): Get up and running quickly
- [System Overview](SYSTEM_OVERVIEW.md): Comprehensive overview of the system
- [Components Guide](COMPONENTS_GUIDE.md): Detailed information about each component
- [Dependencies](DEPENDENCIES.md): Information about the project dependencies

## License

MIT

## Acknowledgements

- [OpenCV](https://opencv.org/) for image processing and computer vision algorithms
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [Flask](https://flask.palletsprojects.com/) for the web dashboard
- [Socket.IO](https://socket.io/) for real-time communication
- [Chart.js](https://www.chartjs.org/) for interactive data visualization
