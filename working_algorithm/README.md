# Algorithm Visualization Demo

This is a standalone application that demonstrates computer vision algorithms with detailed visualizations of each processing step.

## Features

- **Background Subtraction**: Demonstrates foreground-background separation with visualization of each step
- **Kalman Filter Tracking**: Shows object tracking with prediction and correction steps
- **Real-time Metrics**: Displays processing time, FPS, object counts, and algorithm-specific metrics
- **Step-by-Step Visualization**: Shows each processing stage for educational purposes

## Installation

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create the placeholder image:
   ```
   python utils/create_placeholder.py
   ```

4. Add sample videos to the `static/videos` directory:
   - Sample videos should be named:
     - `sample_video.mp4`
     - `people_walking.mp4`
     - `traffic.mp4`
   - Or update the dropdown options in `templates/index.html`

## Running the Application

1. Start the application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5050
   ```

3. Select an algorithm and video source, then click "Start Processing"

## How It Works

### Background Subtraction

The background subtraction algorithm works by:
1. Converting the frame to grayscale
2. Applying Gaussian blur to reduce noise
3. Using MOG2 background subtractor to separate foreground from background
4. Applying threshold to get a binary mask
5. Using morphological operations to clean up the mask
6. Finding contours and drawing bounding boxes around detected objects
7. Calculating distances between objects

### Kalman Filter Tracking

The Kalman filter tracking algorithm works by:
1. Detecting objects using background subtraction
2. Initializing Kalman filters for new objects
3. Predicting object positions in the next frame
4. Matching detections with predictions
5. Updating Kalman filters with new measurements
6. Tracking objects across frames with unique IDs
7. Visualizing trajectories and predictions

## Customization

- Add new algorithms by creating new classes in the `algorithms` directory
- Modify the UI by editing `templates/index.html`
- Add new metrics by updating the metrics dictionaries in the algorithm classes
