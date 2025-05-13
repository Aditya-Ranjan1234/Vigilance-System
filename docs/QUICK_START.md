# Vigilance System: Quick Start Guide

This guide will help you get the Vigilance System up and running quickly.

## Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for real-time processing)
- OpenCV dependencies (on Linux, you may need to install `libgl1-mesa-glx`)

## Installation

### Windows

1. Clone or download the repository
2. Open a command prompt in the project directory
3. Run the setup script:
   ```
   setup.bat
   ```
4. Activate the virtual environment:
   ```
   venv\Scripts\activate
   ```

### Linux/Mac

1. Clone or download the repository
2. Open a terminal in the project directory
3. Make the setup script executable:
   ```
   chmod +x setup.sh
   ```
4. Run the setup script:
   ```
   ./setup.sh
   ```
5. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

## Running the System

### Using Sample Videos

The easiest way to test the system is with sample videos:

1. Download sample videos:
   ```
   python utils/download_sample_videos.py --all
   ```
2. Start the system:
   ```
   python -m vigilance_system
   ```
3. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```
4. Login with the default credentials:
   - Username: `admin`
   - Password: `change_me_immediately`

### Using Your Own Cameras

1. Edit the `config/config.yaml` file to add your cameras:
   ```yaml
   cameras:
     - name: front_door
       url: rtsp://username:password@192.168.1.100:554/stream1
       type: rtsp
       fps: 10
       resolution: [1280, 720]
   ```
2. Start the system:
   ```
   python -m vigilance_system
   ```
3. Access the dashboard as described above

## Configuration Options

### Camera Configuration

The system supports multiple camera types:

- **IP Cameras (RTSP)**:
  ```yaml
  - name: front_door
    url: rtsp://username:password@192.168.1.100:554/stream1
    type: rtsp
    fps: 10
    resolution: [1280, 720]
  ```

- **IP Cameras (HTTP/MJPEG)**:
  ```yaml
  - name: back_yard
    url: http://192.168.1.101:8080/video
    type: http
    fps: 5
    resolution: [1920, 1080]
  ```

- **Local Webcams**:
  ```yaml
  - name: webcam
    url: 0  # Device ID (0 is usually the default camera)
    type: webcam
    fps: 30
    resolution: [640, 480]
  ```

- **Video Files**:
  ```yaml
  - name: store_entrance
    url: videos/store_entrance.mp4
    type: video
    fps: null  # null means use the video's native FPS
    loop: true  # Loop the video when it reaches the end
  ```

### Algorithm Configuration

```yaml
detection:
  model: yolov5s  # Options: yolov5s, yolov5m, yolov5l, yolov5x
  device: cuda:0  # Use 'cpu' if no GPU is available
  confidence_threshold: 0.5
  nms_threshold: 0.45
  classes_of_interest: [0, 1, 2]  # person, bicycle, car in COCO dataset

tracking:
  algorithm: centroid  # Options: centroid, kalman, iou, sort
  max_disappeared: 30  # Maximum number of frames an object can disappear before being removed
  max_distance: 50     # Maximum distance (in pixels) for associating detections

classification:
  algorithm: svm  # Options: svm, knn, naive_bayes
  confidence_threshold: 0.7

analysis:
  algorithm: basic  # Options: basic, weighted, fuzzy
  confidence_threshold: 0.7
```

### Network Configuration

```yaml
network:
  frame_rate: 25  # Target frames per second
  resolution: medium  # Options: low, medium, high
  routing_algorithm: direct  # Options: direct, round_robin, least_connection, weighted, ip_hash
```

### Dashboard Configuration

```yaml
dashboard:
  host: 0.0.0.0  # Listen on all interfaces
  port: 5000
  debug: false
  show_algorithm_steps: true  # Show algorithm pipeline visualization
  show_decision_making: true  # Show decision making visualization
  show_network_stats: true    # Show network performance statistics
```

### Alert Configuration

```yaml
alerts:
  motion_threshold: 0.2  # Minimum motion percentage to trigger alert
  person_loitering_time: 30  # Seconds a person must be present to trigger loitering alert
  crowd_threshold: 3  # Number of people to trigger a crowd alert
  notification:
    email:
      enabled: true
      recipients: [admin@example.com]
      smtp_server: smtp.gmail.com
      smtp_port: 587
      smtp_username: your_email@gmail.com
      smtp_password: your_app_password
```

## Troubleshooting

### Camera Connection Issues

- Check that the camera URL is correct
- Ensure the camera is accessible from your network
- Check username/password if authentication is required
- Try using video files instead for testing

### Performance Issues

- Use a smaller model (e.g., yolov5s instead of yolov5x)
- Reduce the resolution or frame rate in the camera configuration
- Ensure you're using a GPU if available
- If using video files, try reducing their resolution

### Dashboard Not Loading

- Check that the Flask server is running
- Verify the host and port settings
- Check for any errors in the console output

### Algorithm Changes Not Persisting

- Make sure the configuration file is writable
- Check the console for any error messages when changing algorithms
- Restart the system if changes are not being applied
- Verify that the algorithm selection is properly saved in the config file

### Visualization Issues

- Ensure your browser supports HTML5 Canvas and WebSockets
- Try a different browser if visualizations are not displaying correctly
- Check that the frame rate is not set too high for your system
- Reduce the resolution if visualizations are slow or laggy

### Camera Feed Display Issues

- Try switching between grid view and single camera view
- Adjust the browser window size if feeds appear cut off
- Check the network configuration if feeds are loading slowly
- Verify that the resolution setting matches your camera capabilities

## Next Steps

- Read the [System Overview](SYSTEM_OVERVIEW.md) for a comprehensive understanding of the system
- Explore the [Components Guide](COMPONENTS_GUIDE.md) for details on each component
- Check the [README.md](README.md) for complete documentation
- Explore the [Algorithm Visualizations](algorithms/README.md) to understand how each algorithm works
- Try different algorithms and compare their performance
- Experiment with different network routing algorithms and observe the simulated performance
- Use the decision making visualization to understand how alerts are generated
