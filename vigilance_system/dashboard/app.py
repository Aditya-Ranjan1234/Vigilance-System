"""
Dashboard application module for the vigilance system.

This module provides a web-based dashboard for monitoring camera feeds,
viewing detections, and managing alerts.
"""

import os
import time
import threading
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
from flask_socketio import SocketIO
import base64

from vigilance_system.utils.logger import get_logger
from vigilance_system.utils.config import config
from vigilance_system.video_acquisition.stream_manager import stream_manager
from vigilance_system.detection.object_detector import create_detector_from_config
from vigilance_system.alert.decision_maker import decision_maker
from vigilance_system.alert.notifier import notifier

# Initialize logger
logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))
app.secret_key = os.urandom(24)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize detector
detector = create_detector_from_config()

# Processing state
processing_active = False
processing_thread = None
last_frames = {}
last_detections = {}
last_alerts = []


def requires_auth(f):
    """
    Decorator for routes that require authentication.
    
    Args:
        f: Function to decorate
    
    Returns:
        Function: Decorated function
    """
    def decorated(*args, **kwargs):
        auth_config = config.get('dashboard.authentication', {})
        
        if auth_config.get('enabled', True) and 'authenticated' not in session:
            return redirect(url_for('login'))
            
        return f(*args, **kwargs)
    
    decorated.__name__ = f.__name__
    return decorated


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handle login requests.
    
    Returns:
        Response: Login page or redirect to dashboard
    """
    auth_config = config.get('dashboard.authentication', {})
    
    if not auth_config.get('enabled', True):
        session['authenticated'] = True
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if (username == auth_config.get('username') and 
            password == auth_config.get('password')):
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    """
    Handle logout requests.
    
    Returns:
        Response: Redirect to login page
    """
    session.pop('authenticated', None)
    return redirect(url_for('login'))


@app.route('/')
@requires_auth
def index():
    """
    Render the main dashboard page.
    
    Returns:
        Response: Rendered dashboard template
    """
    camera_names = stream_manager.get_camera_names()
    return render_template('index.html', camera_names=camera_names)


@app.route('/api/cameras')
@requires_auth
def get_cameras():
    """
    Get list of available cameras.
    
    Returns:
        Response: JSON response with camera information
    """
    camera_names = stream_manager.get_camera_names()
    return jsonify({'cameras': camera_names})


@app.route('/api/start_processing', methods=['POST'])
@requires_auth
def start_processing():
    """
    Start the video processing thread.
    
    Returns:
        Response: JSON response with status
    """
    global processing_active, processing_thread
    
    if processing_active:
        return jsonify({'status': 'already_running'})
    
    processing_active = True
    processing_thread = threading.Thread(target=process_video_streams)
    processing_thread.daemon = True
    processing_thread.start()
    
    logger.info("Started video processing thread")
    return jsonify({'status': 'started'})


@app.route('/api/stop_processing', methods=['POST'])
@requires_auth
def stop_processing():
    """
    Stop the video processing thread.
    
    Returns:
        Response: JSON response with status
    """
    global processing_active
    
    if not processing_active:
        return jsonify({'status': 'not_running'})
    
    processing_active = False
    logger.info("Stopping video processing thread")
    return jsonify({'status': 'stopping'})


@app.route('/api/status')
@requires_auth
def get_status():
    """
    Get the current processing status.
    
    Returns:
        Response: JSON response with status information
    """
    return jsonify({
        'processing_active': processing_active,
        'camera_count': len(stream_manager.get_camera_names()),
        'detection_stats': detector.get_stats(),
        'alert_count': len(last_alerts)
    })


@app.route('/api/alerts')
@requires_auth
def get_alerts():
    """
    Get recent alerts.
    
    Returns:
        Response: JSON response with alert information
    """
    return jsonify({'alerts': last_alerts})


@socketio.on('connect')
def handle_connect():
    """Handle client connection to SocketIO."""
    logger.info(f"Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection from SocketIO."""
    logger.info(f"Client disconnected: {request.sid}")


def process_video_streams():
    """
    Main processing loop for video streams.
    
    Continuously processes frames from all cameras, runs object detection,
    and generates alerts.
    """
    global last_frames, last_detections, last_alerts
    
    logger.info("Starting video processing loop")
    
    # Start all cameras
    stream_manager.start_all_cameras()
    
    try:
        while processing_active:
            # Get frames from all cameras
            frames = stream_manager.get_all_latest_frames()
            
            for camera_name, frame in frames.items():
                # Store original frame
                last_frames[camera_name] = frame.copy()
                
                # Run object detection
                detections = detector.detect(frame)
                last_detections[camera_name] = detections
                
                # Process detections for alerts
                alerts = decision_maker.process_detections(camera_name, detections)
                
                # Handle alerts
                for alert in alerts:
                    # Add to recent alerts
                    last_alerts.append(alert)
                    if len(last_alerts) > 20:  # Keep only recent alerts
                        last_alerts.pop(0)
                    
                    # Send notification
                    notifier.send_alert(alert, frame)
                    
                    # Emit alert to clients
                    socketio.emit('new_alert', alert)
                
                # Draw detections on frame
                frame_with_detections = detector.draw_detections(frame, detections)
                
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame_with_detections)
                frame_bytes = buffer.tobytes()
                
                # Encode as base64 for sending to clients
                frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
                
                # Emit frame to clients
                socketio.emit('frame_update', {
                    'camera': camera_name,
                    'frame': frame_base64,
                    'detection_count': len(detections)
                })
            
            # Sleep to control processing rate
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Error in video processing loop: {str(e)}")
    finally:
        # Stop all cameras
        stream_manager.stop_all_cameras()
        logger.info("Video processing loop stopped")


def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Tuple[Flask, SocketIO]: Configured Flask app and SocketIO instance
    """
    # Initialize cameras from configuration
    stream_manager.initialize_cameras()
    
    # Configure app
    host = config.get('dashboard.host', '0.0.0.0')
    port = config.get('dashboard.port', 5000)
    debug = config.get('dashboard.debug', False)
    
    return app, socketio, host, port, debug


if __name__ == '__main__':
    app, socketio, host, port, debug = create_app()
    socketio.run(app, host=host, port=port, debug=debug)
