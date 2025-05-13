import os
import cv2
import numpy as np
import time
import logging
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO
import base64
import threading
import queue
import json

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('algorithm_demo')

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'algorithm_demo_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Import algorithm modules
from algorithms.background_subtraction import BackgroundSubtractor
from algorithms.kalman_tracker import KalmanTracker

# Global variables
processing_queue = queue.Queue(maxsize=10)
results_queue = queue.Queue(maxsize=10)
current_algorithm = "background_subtraction"  # Default algorithm
current_video = None
processing_active = False
frame_count = 0
fps = 0
last_fps_time = time.time()
metrics = {
    "objects_detected": 0,
    "processing_time": 0,
    "average_confidence": 0,
    "false_positives": 0,
    "true_positives": 0
}

# Initialize algorithms
bg_subtractor = BackgroundSubtractor()
kalman_tracker = KalmanTracker()

# Routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/start_processing', methods=['POST'])
def start_processing():
    """Start video processing with selected algorithm."""
    global processing_active, current_algorithm, current_video

    data = request.get_json()
    algorithm = data.get('algorithm', 'background_subtraction')
    video_path = data.get('video_path', 'sample_video.mp4')

    # Validate algorithm
    if algorithm not in ['background_subtraction', 'kalman_tracker']:
        return jsonify({'error': 'Invalid algorithm'}), 400

    # Use videos from vigilance system
    vigilance_videos_path = os.path.join('..', 'vigilance_system', 'videos')
    if not os.path.exists(vigilance_videos_path):
        # Try alternative path
        vigilance_videos_path = os.path.join('..', 'videos')
        if not os.path.exists(vigilance_videos_path):
            # Create directory and inform user
            os.makedirs(os.path.join('static', 'videos'), exist_ok=True)
            return jsonify({'error': 'Vigilance system videos not found. Please add videos to the static/videos directory.'}), 404

    # Find videos in vigilance system directory
    video_files = []
    for root, dirs, files in os.walk(vigilance_videos_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))

    if not video_files:
        return jsonify({'error': 'No video files found in vigilance system directory'}), 404

    # Use the selected video if it exists, otherwise use the first video
    selected_video = None
    for video_file in video_files:
        if os.path.basename(video_file) == video_path:
            selected_video = video_file
            break

    if not selected_video:
        selected_video = video_files[0]

    # Update global variables
    current_algorithm = algorithm
    current_video = selected_video

    logger.info(f"Starting processing with algorithm: {algorithm}, video: {selected_video}")

    # Start processing if not already active
    if not processing_active:
        processing_active = True
        threading.Thread(target=process_video, daemon=True).start()

    return jsonify({'success': True, 'algorithm': algorithm, 'video': os.path.basename(selected_video)})

@app.route('/api/stop_processing', methods=['POST'])
def stop_processing():
    """Stop video processing."""
    global processing_active
    processing_active = False
    return jsonify({'success': True})

@app.route('/api/get_metrics', methods=['GET'])
def get_metrics():
    """Get current processing metrics."""
    global metrics, fps
    metrics['fps'] = fps
    return jsonify(metrics)

@app.route('/api/get_videos', methods=['GET'])
def get_videos():
    """Get available videos from vigilance system."""
    # Check vigilance system videos directory
    vigilance_videos_path = os.path.join('..', 'vigilance_system', 'videos')
    if not os.path.exists(vigilance_videos_path):
        # Try alternative path
        vigilance_videos_path = os.path.join('..', 'videos')
        if not os.path.exists(vigilance_videos_path):
            # Try D:\Main EL\videos
            vigilance_videos_path = r'D:\Main EL\videos'
            if not os.path.exists(vigilance_videos_path):
                return jsonify({'videos': []})

    # Find videos in vigilance system directory
    video_files = []
    for root, dirs, files in os.walk(vigilance_videos_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_files.append({
                    'path': file,
                    'name': os.path.splitext(file)[0].replace('_', ' ').title()
                })

    return jsonify({'videos': video_files})

def process_video():
    """Process video frames with selected algorithm."""
    global processing_active, current_algorithm, current_video, frame_count, fps, last_fps_time, metrics

    logger.info(f"Starting video processing with algorithm: {current_algorithm}")

    # Open video capture
    try:
        cap = cv2.VideoCapture(current_video)
        if not cap.isOpened():
            logger.error(f"Error opening video: {current_video}")
            # Try with a different path format
            if os.path.exists(current_video):
                logger.info(f"File exists but couldn't be opened. Trying with absolute path...")
                cap = cv2.VideoCapture(os.path.abspath(current_video))

            if not cap.isOpened():
                logger.error(f"Still couldn't open video. Stopping processing.")
                processing_active = False
                socketio.emit('error', {'message': f"Could not open video: {os.path.basename(current_video)}"})
                return
    except Exception as e:
        logger.error(f"Exception opening video: {str(e)}")
        processing_active = False
        socketio.emit('error', {'message': f"Error opening video: {str(e)}"})
        return

    logger.info(f"Successfully opened video: {current_video}")

    # Reset metrics
    metrics = {
        "objects_detected": 0,
        "processing_time": 0,
        "average_confidence": 0,
        "false_positives": 0,
        "true_positives": 0
    }

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video properties: {frame_width}x{frame_height}, {total_frames} frames")

    # Process frames
    while processing_active:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            # Loop back to beginning of video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Update frame count and FPS
        frame_count += 1
        current_time = time.time()
        if current_time - last_fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_fps_time = current_time

        # Process frame with selected algorithm
        start_time = time.time()

        try:
            if current_algorithm == "background_subtraction":
                result_frame, step_frames, frame_metrics = bg_subtractor.process(frame)
            else:  # kalman_tracker
                result_frame, step_frames, frame_metrics = kalman_tracker.process(frame)

            # Update metrics
            process_time = time.time() - start_time
            metrics["processing_time"] = process_time
            metrics.update(frame_metrics)

            # Encode frames for transmission
            encoded_frames = {}
            encoded_frames['result'] = encode_frame(result_frame)

            for step, step_frame in step_frames.items():
                encoded_frames[step] = encode_frame(step_frame)

            # Send frames to clients
            socketio.emit('frame_update', {
                'frames': encoded_frames,
                'metrics': metrics
            })

            # Control frame rate
            time.sleep(max(0, 1/30 - process_time))  # Target 30 FPS

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            # Continue processing next frame
            continue

    # Release resources
    cap.release()
    logger.info("Video processing stopped")

def encode_frame(frame):
    """Encode frame as base64 string."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/videos', exist_ok=True)

    # Create placeholder image
    try:
        from utils.create_placeholder import create_placeholder
        placeholder = create_placeholder()
        cv2.imwrite('static/placeholder.jpg', placeholder)
        logger.info("Created placeholder image")
    except Exception as e:
        logger.error(f"Error creating placeholder image: {str(e)}")

    # Check for videos
    logger.info("Checking for videos...")
    vigilance_videos_path = os.path.join('..', 'vigilance_system', 'videos')
    if not os.path.exists(vigilance_videos_path):
        vigilance_videos_path = os.path.join('..', 'videos')
        if not os.path.exists(vigilance_videos_path):
            vigilance_videos_path = r'D:\Main EL\videos'
            if not os.path.exists(vigilance_videos_path):
                logger.warning("No videos directory found")

    if os.path.exists(vigilance_videos_path):
        logger.info(f"Found videos directory: {vigilance_videos_path}")

    # Start the server
    logger.info("Starting server...")
    socketio.run(app, host='0.0.0.0', port=5050, debug=True)
