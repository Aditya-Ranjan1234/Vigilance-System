# Vigilance System Components Guide

This guide explains how to run and understand the different components of the Vigilance System, with a focus on Artificial Intelligence/Machine Learning (AI/ML), Data Analysis and Algorithms (DAA), and Computer Networks (CN) aspects.

## Table of Contents
- [AI/ML Components](#aiml-components)
- [Data Analysis and Algorithms (DAA)](#data-analysis-and-algorithms-daa)
- [Computer Networks (CN)](#computer-networks-cn)
- [Running Individual Components](#running-individual-components)
- [Component Integration](#component-integration)

## AI/ML Components

The Vigilance System uses several AI/ML components for object detection, tracking, and anomaly detection.

### Object Detection Models

The system uses YOLOv5 (You Only Look Once) models for real-time object detection.

#### Available Models:
- **YOLOv5s**: Small model, fastest but less accurate
- **YOLOv5m**: Medium model, balanced speed and accuracy
- **YOLOv5l**: Large model, more accurate but slower
- **YOLOv5x**: Extra large model, most accurate but slowest

#### How to Change the Model:

1. **In the configuration file**:
   ```yaml
   detection:
     model: yolov5s  # Change to yolov5m, yolov5l, or yolov5x
     confidence_threshold: 0.5
   ```

2. **Using the command line**:
   ```bash
   # Run the example script with a different model
   python examples/simple_detection.py --input 0 --model yolov5m
   ```

### Object Tracking

The system uses a custom tracking algorithm to track objects across frames.

#### Key Features:
- Object persistence across frames
- Loitering detection based on time thresholds
- Crowd detection based on object counts

#### How to Adjust Tracking Parameters:

```yaml
alerts:
  person_loitering_time: 30  # Seconds a person must be present to trigger loitering alert
  crowd_threshold: 3  # Number of people to trigger a crowd alert
```

### Running AI/ML Components Independently:

```bash
# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run object detection on a video file
python examples/simple_detection.py --input path/to/video.mp4

# Run object detection on a webcam
python examples/simple_detection.py --input 0

# Save detection results to a video file
python examples/simple_detection.py --input path/to/video.mp4 --output results.mp4
```

## Data Analysis and Algorithms (DAA)

The Vigilance System implements several algorithms for data processing, analysis, and decision making.

### Video Preprocessing Algorithms

#### Frame Extraction:
The system extracts frames at a configurable rate to balance performance and accuracy.

```python
# Example of using the frame extractor
from vigilance_system.preprocessing.frame_extractor import FrameExtractor

extractor = FrameExtractor(target_fps=10)
processed_frame = extractor.extract_frame(frame)
```

#### Video Stabilization:
The system uses optical flow or feature matching to stabilize shaky video.

```python
# Example of using the video stabilizer
from vigilance_system.preprocessing.video_stabilizer import VideoStabilizer

stabilizer = VideoStabilizer(smoothing_radius=15, method='optical_flow')
stabilized_frame = stabilizer.stabilize_frame(frame)
```

### Decision Making Algorithms

The system uses several algorithms to make decisions based on detections:

1. **Object Association**: Matches objects across frames using distance-based metrics
2. **Loitering Detection**: Tracks object duration and triggers alerts based on thresholds
3. **Crowd Detection**: Counts objects of specific classes and triggers alerts based on thresholds

```python
# Example of using the decision maker
from vigilance_system.alert.decision_maker import decision_maker

alerts = decision_maker.process_detections(camera_name, detections)
```

### Running Data Analysis Components Independently:

You can create custom scripts to test and analyze the data processing components:

```python
# Example script to test video stabilization
import cv2
from vigilance_system.preprocessing.video_stabilizer import VideoStabilizer

# Open video source
cap = cv2.VideoCapture(0)  # Use webcam

# Create stabilizer
stabilizer = VideoStabilizer(smoothing_radius=15)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Stabilize frame
    stabilized = stabilizer.stabilize_frame(frame)

    # Display original and stabilized frames
    cv2.imshow('Original', frame)
    cv2.imshow('Stabilized', stabilized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Computer Networks (CN)

The Vigilance System relies on computer networks for camera streaming, dashboard communication, and alert notifications.

### Camera Streaming Protocols

The system supports multiple streaming protocols:

1. **RTSP (Real-Time Streaming Protocol)**: Used for IP cameras
2. **HTTP/MJPEG**: Used for web cameras and some IP cameras
3. **Local Device**: Used for directly connected webcams

#### How to Configure Camera Streams:

```yaml
cameras:
  # IP camera using RTSP protocol
  - name: front_door
    url: rtsp://username:password@192.168.1.100:554/stream1
    type: rtsp
    fps: 10
    resolution: [1280, 720]

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

If no cameras are configured (all entries are commented out), the system will automatically use video files from the `vigilance_system/videos` directory.

### Web Dashboard Communication

The dashboard uses:
1. **HTTP/HTTPS**: For serving the web interface
2. **WebSockets (Socket.IO)**: For real-time updates of video frames and alerts

#### How to Configure the Dashboard Network:

```yaml
dashboard:
  host: 0.0.0.0  # Listen on all interfaces
  port: 5000
  debug: false
```

### Alert Notification Networks

The system can send alerts through:
1. **SMTP**: For email notifications
2. **SMS Gateways**: For text message notifications

#### How to Configure Alert Networks:

```yaml
alerts:
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
      service: twilio
      twilio_account_sid: your_account_sid
      twilio_auth_token: your_auth_token
      twilio_from_number: +1987654321
```

### Testing Network Components:

```bash
# Test RTSP connection
python -c "import cv2; cap = cv2.VideoCapture('rtsp://username:password@192.168.1.100:554/stream1'); print('Connected:', cap.isOpened())"

# Test dashboard connectivity
curl http://localhost:5000

# Test email notification
python -c "import smtplib; server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls(); server.login('your_email@gmail.com', 'your_app_password'); server.sendmail('your_email@gmail.com', 'recipient@example.com', 'Subject: Test\n\nTest message'); server.quit(); print('Email sent')"
```

## Running Individual Components

You can run individual components of the system for testing or development purposes.

### 1. Video Acquisition Component

```bash
# Create a test script
cat > test_camera.py << 'EOF'
from vigilance_system.video_acquisition.camera import RTSPCamera, HTTPCamera, WebcamCamera
import cv2
import time

# Create a camera (choose one)
# camera = RTSPCamera("test", "rtsp://username:password@192.168.1.100:554/stream1")
# camera = HTTPCamera("test", "http://192.168.1.101:8080/video")
camera = WebcamCamera("test", 0)  # Use default webcam

# Connect and start
camera.connect()
camera.start()

# Display frames for 30 seconds
start_time = time.time()
while time.time() - start_time < 30:
    success, frame = camera.get_latest_frame()
    if success:
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    time.sleep(0.03)  # ~30 FPS

# Clean up
camera.stop()
cv2.destroyAllWindows()
EOF

# Run the test script
python test_camera.py
```

### 2. Object Detection Component

```bash
# Create a test script
cat > test_detection.py << 'EOF'
from vigilance_system.detection.object_detector import ObjectDetector
import cv2

# Create detector
detector = ObjectDetector(model_name="yolov5s", confidence_threshold=0.5)

# Open video source (webcam or file)
cap = cv2.VideoCapture(0)  # Use webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    detections = detector.detect(frame)

    # Draw detections
    frame_with_detections = detector.draw_detections(frame, detections)

    # Display
    cv2.imshow('Detections', frame_with_detections)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
EOF

# Run the test script
python test_detection.py
```

### 3. Alert Component

```bash
# Create a test script
cat > test_alerts.py << 'EOF'
from vigilance_system.alert.notifier import notifier
import time
import cv2
import numpy as np

# Create a test image
image = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(image, "TEST ALERT", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

# Create a test alert
alert = {
    'type': 'test',
    'camera': 'test_camera',
    'timestamp': time.time(),
    'message': 'This is a test alert'
}

# Send the alert
success = notifier.send_alert(alert, image)
print(f"Alert sent: {success}")
EOF

# Run the test script
python test_alerts.py
```

### 4. Dashboard Component

```bash
# Run just the dashboard
python -c "from vigilance_system.dashboard.app import create_app; app, socketio, host, port, debug = create_app(); socketio.run(app, host=host, port=port, debug=True)"
```

## Component Integration

The full system integrates all components. Here's how they work together:

1. **Video Acquisition** captures frames from cameras or video files
2. **Preprocessing** extracts and stabilizes frames
3. **Detection** identifies objects in the frames
4. **Decision Making** analyzes detections for alert conditions
5. **Alert** sends notifications when conditions are met and saves alert images
6. **Dashboard** displays everything to the user

To run the full integrated system:

```bash
# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the system
python -m vigilance_system
```

### Directory Structure

The system is organized into the following directories:

- **vigilance_system/**: Main package
  - **alert/**: Alert generation components
  - **alerts/**: Stored alert images and metadata
  - **dashboard/**: Web interface components
  - **detection/**: Object detection components
  - **preprocessing/**: Video preprocessing components
  - **utils/**: Utility functions
  - **video_acquisition/**: Camera and stream handling
  - **videos/**: Video files for testing

- **alerts/**: Generated alert images and metadata
- **logs/**: Log files
- **models/**: Downloaded ML models
- **tests/**: Test files
- **examples/**: Example scripts
- **videos/**: Additional video files

## Advanced Configuration

### Custom Detection Classes

You can focus on specific object classes:

```yaml
detection:
  classes_of_interest: [0, 1, 2]  # 0=person, 1=bicycle, 2=car in COCO dataset
```

### Custom Alert Rules

You can create custom alert conditions by modifying the `decision_maker.py` file:

```python
def _check_custom_condition(self, camera_name, tracked_objects):
    """
    Check for a custom alert condition.
    """
    alerts = []
    # Your custom logic here
    return alerts
```

Then add your method to the `process_detections` method:

```python
def process_detections(self, camera_name, detections):
    # ...existing code...

    # Check for custom condition
    custom_alerts = self._check_custom_condition(camera_name, tracked_objects)
    alerts.extend(custom_alerts)

    return alerts
```

### Performance Optimization

For better performance:

1. **Reduce resolution**:
   ```yaml
   cameras:
     - name: front_door
       resolution: [640, 360]  # Lower resolution
   ```

2. **Lower frame rate**:
   ```yaml
   cameras:
     - name: front_door
       fps: 5  # Process fewer frames per second
   ```

3. **Use a smaller model**:
   ```yaml
   detection:
     model: yolov5s  # Smallest, fastest model
   ```

4. **Limit detection area**:
   Modify the `object_detector.py` file to add a region of interest:
   ```python
   def detect(self, frame):
       # Define region of interest (x, y, width, height)
       roi = (100, 100, 400, 300)
       x, y, w, h = roi

       # Crop frame to ROI
       roi_frame = frame[y:y+h, x:x+w]

       # Detect in ROI only
       detections = self._detect_in_frame(roi_frame)

       # Adjust coordinates back to original frame
       for detection in detections:
           detection.bbox = (
               detection.bbox[0] + x,
               detection.bbox[1] + y,
               detection.bbox[2] + x,
               detection.bbox[3] + y
           )

       return detections
   ```
