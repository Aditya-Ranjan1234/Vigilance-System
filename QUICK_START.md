# Vigilance System: Quick Start Guide

This guide will help you get the Vigilance System up and running quickly.

## Prerequisites

- Python 3.10 or higher
- OpenCV dependencies (on Linux, you may need to install `libgl1-mesa-glx`)
- scikit-learn (for K-means clustering and other ML algorithms)

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
   python download_sample_videos.py --all
   ```
2. Start the system:
   ```
   # Start with default settings
   python -m vigilance_system

   # Start with specific algorithms
   python -m vigilance_system --detection-algorithm background_subtraction --tracking-algorithm klt_tracker

   # Enable the analysis dashboard
   python -m vigilance_system --enable-analysis

   # Specify ports for both dashboards
   python -m vigilance_system --port 5000 --analysis-port 5001 --enable-analysis
   ```
3. Open your browser and navigate to:
   ```
   # Main dashboard
   http://localhost:5000

   # Analysis dashboard (if enabled)
   http://localhost:5001
   ```
4. Login with the default credentials:
   - Username: `admin`
   - Password: `change_me_immediately`

### Using Your Own Cameras

1. Edit the `config.yaml` file to add your cameras:
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

### Detection Configuration

```yaml
detection:
  algorithm: background_subtraction  # Options: background_subtraction, mog2, knn, svm_classifier
  use_algorithm_detectors: true
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

### Alert Configuration

```yaml
alerts:
  cooldown: 60  # Seconds between alerts of the same type

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
    threshold: 3  # Number of people to trigger a crowd alert
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
      metrics: [id_switches, mota, motp]
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

## Troubleshooting

### Camera Connection Issues

- Check that the camera URL is correct
- Ensure the camera is accessible from your network
- Check username/password if authentication is required
- Try using video files instead for testing

### Performance Issues

- Try different detection algorithms (background_subtraction is faster than svm_classifier)
- Use simpler tracking algorithms (klt_tracker is generally faster than kalman_filter)
- Disable video stabilization if not needed
- Reduce the resolution or frame rate in the camera configuration
- If using video files, try reducing their resolution
- Adjust parameters like min_contour_area or max_corners to reduce computational load

### Dashboard Not Loading

- Check that the Flask server is running
- Verify the host and port settings
- Check for any errors in the console output
- Try a different browser

### Analysis Dashboard Issues

- If the analysis dashboard is not loading, make sure it's enabled with `--enable-analysis`
- Check that the analysis port (default: 5001) is not being used by another application
- If charts are not updating, check browser console for errors
- Ensure WebSocket connections are not blocked by firewalls
- If metrics are missing, verify that the corresponding algorithm is enabled and running

## Next Steps

- Read the [System Overview](SYSTEM_OVERVIEW.md) for a comprehensive understanding of the system
- Explore the [Components Guide](COMPONENTS_GUIDE.md) for details on each component
- Check the [README.md](README.md) for complete documentation
