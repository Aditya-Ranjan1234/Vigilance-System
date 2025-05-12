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
import random
from typing import Dict, Any, List
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_socketio import SocketIO
import base64

from vigilance_system.utils.logger import get_logger
from vigilance_system.utils.config import config
from vigilance_system.video_acquisition.stream_manager import stream_manager
from vigilance_system.detection.object_detector import create_detector_from_config
from vigilance_system.alert.decision_maker import decision_maker
from vigilance_system.alert.notifier import notifier
from vigilance_system.detection.trajectory_helper import ensure_valid_trajectory_points
from vigilance_system.utils.cv_utils import safe_putText
from vigilance_system.network.simulation import network_simulator
from vigilance_system.network.visualization import network_visualizer

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
view_mode = 'grid'  # Default view mode


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

        # Get client IP address
        client_ip = request.remote_addr

        if (username == auth_config.get('username') and
            password == auth_config.get('password')):
            session['authenticated'] = True
            session['client_ip'] = client_ip

            # Log the successful login with IP address
            logger.info(f"User {username} logged in from IP: {client_ip}")

            return redirect(url_for('index'))
        else:
            # Log the failed login attempt with IP address
            logger.warning(f"Failed login attempt for user {username} from IP: {client_ip}")
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
    # Get visualization settings
    show_stabilization = config.get('dashboard.visualizations.show_stabilization', True)
    show_tracking = config.get('dashboard.visualizations.show_tracking', True)
    show_decision_making = config.get('dashboard.visualizations.show_decision_making', True)
    show_algorithm_steps = config.get('dashboard.visualizations.show_algorithm_steps', True)

    # Get algorithm settings
    detection_algorithm = config.get('detection.model', 'yolov5s')
    tracking_algorithm = config.get('tracking.algorithm', 'centroid')
    classifier_algorithm = config.get('classification.algorithm', 'svm')
    analysis_algorithm = config.get('analysis.algorithm', 'basic')

    # Get network settings from simulator
    network_stats = network_simulator.get_stats()
    frame_rate = network_stats.get('frame_rate', 25)
    resolution = network_stats.get('resolution', 'medium')
    routing_algorithm = network_stats.get('routing_algorithm', 'direct')

    # Get real network metrics from simulator
    bandwidth = f"{network_stats.get('bandwidth', 0.2):.2f} MB/s"
    latency = f"{network_stats.get('avg_latency', 0.03) * 1000:.1f} ms"
    packet_loss = f"{network_stats.get('packet_loss', 0.1):.1f}%"
    jitter = f"{network_stats.get('jitter', 0.012) * 1000:.1f} ms"

    return jsonify({
        'processing_active': processing_active,
        'camera_count': len(stream_manager.get_camera_names()),
        'detection_stats': detector.get_stats(),
        'alert_count': len(last_alerts),
        'visualizations': {
            'show_stabilization': show_stabilization,
            'show_tracking': show_tracking,
            'show_decision_making': show_decision_making,
            'show_algorithm_steps': show_algorithm_steps
        },
        'algorithms': {
            'detection_algorithm': detection_algorithm,
            'tracking_algorithm': tracking_algorithm,
            'classifier_algorithm': classifier_algorithm,
            'analysis_algorithm': analysis_algorithm
        },
        'network': {
            'frame_rate': frame_rate,
            'resolution': resolution,
            'routing_algorithm': routing_algorithm,
            'metrics': {
                'bandwidth': bandwidth,
                'latency': latency,
                'packet_loss': packet_loss,
                'jitter': jitter
            }
        }
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


@app.route('/api/network_status')
@requires_auth
def get_network_status():
    """
    Get detailed network status information.

    Returns:
        Response: JSON response with network status information
    """
    try:
        # Get network stats
        stats = network_simulator.get_stats()

        # Get node client stats
        from vigilance_system.network.node_client import node_client
        client_stats = node_client.get_stats()

        # Get current algorithm
        current_algorithm = client_stats.get('algorithm', stats.get('routing_algorithm', 'direct'))

        # Count real vs simulated nodes
        real_nodes = client_stats.get('real_nodes', 0)
        simulated_nodes = client_stats.get('simulated_nodes', 0)

        # Prepare node information
        nodes = []
        for node_id, node_info in network_simulator.nodes.items():
            # Calculate node load (random for demo if not available)
            load = node_info.get('load', random.random() * 0.8)

            # Check if node is simulated
            simulated = node_id not in client_stats.get('real_node_ids', [])

            nodes.append({
                'id': node_id,
                'active': True,  # All nodes in simulator are considered active
                'simulated': simulated,
                'capacity': node_info.get('capacity', 1.0),
                'load': load,
                'type': node_info.get('node_type', 'processing')
            })

        # Sort nodes by ID
        nodes.sort(key=lambda x: x['id'])

        return jsonify({
            'current_algorithm': current_algorithm,
            'real_nodes': real_nodes,
            'simulated_nodes': simulated_nodes,
            'total_nodes': real_nodes + simulated_nodes,
            'bandwidth': stats.get('bandwidth', 0.0),
            'avg_latency': stats.get('avg_latency', 0.0),
            'packet_loss': stats.get('packet_loss', 0.0),
            'jitter': stats.get('jitter', 0.0),
            'nodes': nodes
        })
    except Exception as e:
        logger.error(f"Error in network status API: {str(e)}")
        return jsonify({
            'error': str(e),
            'current_algorithm': 'direct',
            'real_nodes': 0,
            'simulated_nodes': 0,
            'total_nodes': 0,
            'bandwidth': 0.0,
            'avg_latency': 0.0,
            'packet_loss': 0.0,
            'jitter': 0.0,
            'nodes': []
        })


@app.route('/api/network_visualization')
@requires_auth
def get_network_visualization():
    """
    Get the network visualization frame.

    Returns:
        Response: JSON response with network visualization frame
    """
    try:
        # Update the network visualizer
        network_visualizer.update()

        # Get the visualization frame
        frame = network_visualizer.draw()

        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Encode as base64 for sending to clients
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

        # Get client IP from session
        client_ip = session.get('client_ip', request.remote_addr)

        # Get network stats
        stats = network_simulator.get_stats()

        # Add client IP to stats
        stats['client_ip'] = client_ip

        # Make sure routing algorithm is correctly reported
        from vigilance_system.network.node_client import node_client
        client_stats = node_client.get_stats()

        # Ensure routing algorithm is correctly reported
        current_algorithm = client_stats.get('algorithm', stats.get('routing_algorithm', 'direct'))
        stats['routing_algorithm'] = current_algorithm

        # Log the current routing algorithm for debugging
        logger.info(f"Current routing algorithm: {current_algorithm}")

        return jsonify({
            'frame': frame_base64,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error in network visualization API: {str(e)}")
        # Return a fallback response with error information
        return jsonify({
            'frame': '',  # Empty frame
            'stats': {
                'routing_algorithm': 'direct',
                'frame_rate': 25,
                'resolution': 'medium',
                'bandwidth': 0.0,
                'avg_latency': 0.0,
                'packet_loss': 0.0,
                'jitter': 0.0,
                'error': str(e)
            }
        })


@app.route('/api/update_visualizations', methods=['POST'])
@requires_auth
def update_visualizations():
    """
    Update visualization settings.

    Returns:
        Response: JSON response with status
    """
    global view_mode, show_stabilization, show_tracking, show_decision_making, show_algorithm_steps
    data = request.json

    # Update config
    config.set('dashboard.visualizations.show_stabilization', data.get('show_stabilization', True))
    config.set('dashboard.visualizations.show_tracking', data.get('show_tracking', True))
    config.set('dashboard.visualizations.show_decision_making', data.get('show_decision_making', True))
    config.set('dashboard.visualizations.show_algorithm_steps', data.get('show_algorithm_steps', True))

    # Update local variables for immediate effect
    show_stabilization = data.get('show_stabilization', True)
    show_tracking = data.get('show_tracking', True)
    show_decision_making = data.get('show_decision_making', True)
    show_algorithm_steps = data.get('show_algorithm_steps', True)

    # Handle view mode changes
    if 'view_mode' in data:
        view_mode = data['view_mode']
        logger.info(f"View mode changed to: {view_mode}")

        # Adjust visualization settings based on view mode
        if view_mode == 'single':
            # In single view, we want to show more detailed decision making
            # and less overlay text on the video
            show_algorithm_steps = False  # Hide algorithm steps in single view
            config.set('dashboard.visualizations.show_algorithm_steps', False)
        else:
            # In grid view, we can show more information
            show_algorithm_steps = data.get('show_algorithm_steps', True)
            config.set('dashboard.visualizations.show_algorithm_steps', show_algorithm_steps)

    logger.info(f"Updated visualization settings: {data}")
    return jsonify({'status': 'success'})


@app.route('/api/update_algorithms', methods=['POST'])
@requires_auth
def update_algorithms():
    """
    Update algorithm settings.

    Returns:
        Response: JSON response with status
    """
    data = request.json
    changes_made = False

    try:
        # Save each setting immediately to ensure persistence
        # Update algorithm config
        if 'detection_algorithm' in data:
            config.set('detection.model', data['detection_algorithm'], save=True)
            changes_made = True
            logger.info(f"Updated detection algorithm to {data['detection_algorithm']}")

        if 'tracking_algorithm' in data:
            config.set('tracking.algorithm', data['tracking_algorithm'], save=True)
            changes_made = True
            logger.info(f"Updated tracking algorithm to {data['tracking_algorithm']}")
            # Directly update the tracking algorithm in decision_maker
            if hasattr(decision_maker, 'set_tracking_algorithm'):
                decision_maker.set_tracking_algorithm(data['tracking_algorithm'])
            else:
                # If method doesn't exist, reset trackers to apply new algorithm
                decision_maker.reset_all_trackers()
            logger.info(f"Applied tracking algorithm {data['tracking_algorithm']}")

        if 'classifier_algorithm' in data:
            config.set('classification.algorithm', data['classifier_algorithm'], save=True)
            changes_made = True
            logger.info(f"Updated classifier algorithm to {data['classifier_algorithm']}")
            # Apply classifier algorithm change
            decision_maker.set_classifier_algorithm(data['classifier_algorithm'])
            logger.info(f"Applied classifier algorithm {data['classifier_algorithm']}")

        if 'analysis_algorithm' in data:
            config.set('analysis.algorithm', data['analysis_algorithm'], save=True)
            changes_made = True
            logger.info(f"Updated analysis algorithm to {data['analysis_algorithm']}")
            # Apply analysis algorithm change
            decision_maker.set_analysis_algorithm(data['analysis_algorithm'])
            logger.info(f"Applied analysis algorithm {data['analysis_algorithm']}")

        # Update network config
        if 'network' in data:
            network_data = data['network']

            if 'frame_rate' in network_data:
                frame_rate = network_data['frame_rate']
                config.set('network.frame_rate', frame_rate, save=True)
                # Update the network simulator
                network_simulator.frame_rate = frame_rate
                changes_made = True
                logger.info(f"Updated frame rate to {frame_rate}")

            if 'resolution' in network_data:
                resolution = network_data['resolution']
                config.set('network.resolution', resolution, save=True)
                # Update the network simulator
                network_simulator.resolution = resolution
                changes_made = True
                logger.info(f"Updated resolution to {resolution}")

            if 'routing_algorithm' in network_data:
                routing_algorithm = network_data['routing_algorithm']

                # Validate the algorithm
                valid_algorithms = ['direct', 'round_robin', 'least_connection', 'weighted', 'ip_hash', 'yolov8']
                if routing_algorithm not in valid_algorithms:
                    logger.warning(f"Invalid routing algorithm: {routing_algorithm}. Using 'direct' instead.")
                    routing_algorithm = 'direct'

                # Update configuration
                config.set('network.routing_algorithm', routing_algorithm, save=True)

                # Update the network simulator
                network_simulator.set_routing_algorithm(routing_algorithm)

                # Force immediate update to ensure changes take effect
                try:
                    # Make sure node client is updated directly
                    from vigilance_system.network.node_client import node_client
                    node_client.set_algorithm(routing_algorithm)

                    # Double-check that the change took effect
                    actual_algorithm = node_client.current_algorithm
                    if actual_algorithm != routing_algorithm:
                        logger.warning(f"Algorithm mismatch: set to {routing_algorithm} but got {actual_algorithm}")
                        # Try again with more force
                        node_client.set_algorithm(routing_algorithm)

                        # Verify again
                        actual_algorithm = node_client.current_algorithm
                        if actual_algorithm != routing_algorithm:
                            logger.error(f"Failed to set algorithm after retry: {actual_algorithm} != {routing_algorithm}")
                        else:
                            logger.info(f"Algorithm set successfully after retry: {routing_algorithm}")

                    # Force a reload of the network simulator with the new algorithm
                    network_simulator.set_routing_algorithm(routing_algorithm)

                    # Kill existing node processes before reconnecting
                    try:
                        import subprocess

                        # Kill all node server processes
                        logger.info("Terminating existing node processes before reconnecting...")
                        # Kill all python processes running node_server.py
                        subprocess.run(['taskkill', '/F', '/FI', 'WINDOWTITLE eq Node*'], shell=True)
                        logger.info("Existing node processes terminated")
                    except Exception as kill_error:
                        logger.error(f"Error terminating node processes: {str(kill_error)}")

                    # Force a reconnect to ensure the algorithm takes effect
                    try:
                        # Disconnect first
                        node_client.disconnect()
                        time.sleep(1)  # Give time for connections to close

                        # Connect with the new algorithm
                        node_client.connect()
                        logger.info(f"Reconnected node client to ensure algorithm change takes effect")

                        # Verify the algorithm was applied
                        if node_client.current_algorithm != routing_algorithm:
                            logger.error(f"Algorithm still not applied after reconnect: {node_client.current_algorithm} != {routing_algorithm}")
                        else:
                            logger.info(f"Algorithm successfully applied after reconnect: {routing_algorithm}")
                    except Exception as reconnect_error:
                        logger.error(f"Error reconnecting node client: {str(reconnect_error)}")

                    logger.info(f"Directly updated node client algorithm to {routing_algorithm}")
                    changes_made = True
                except Exception as e:
                    logger.error(f"Error updating node client algorithm: {str(e)}")

        # Apply changes to running components
        if changes_made:
            # Force reload the configuration to ensure changes are applied
            config.reload()
            logger.info("Configuration reloaded")

            # Reload detector with new model if detection algorithm changed
            if 'detection_algorithm' in data:
                global detector
                detector = create_detector_from_config()
                logger.info(f"Reloaded detector with {data['detection_algorithm']}")

            logger.info(f"Successfully updated and applied all algorithm and network settings")

        return jsonify({'status': 'success'})

    except Exception as e:
        logger.error(f"Error updating algorithm settings: {str(e)}")
        # Even if there's an error, try to save the config to ensure changes persist
        try:
            config.save()
            logger.info("Configuration saved despite error")
        except Exception as save_error:
            logger.error(f"Error saving configuration: {str(save_error)}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/visualizations')
@requires_auth
def visualizations():
    """
    Render the algorithm visualizations page.

    Returns:
        Response: Rendered visualizations template
    """
    return render_template('visualizations.html')


@app.route('/network')
@requires_auth
def network():
    """
    Render the network visualization page.

    Returns:
        Response: Rendered network template
    """
    # Use the ultra-basic template that should work in any environment
    return render_template('network_basic.html')


@app.route('/network/info')
@requires_auth
def network_info():
    """
    Render the network information page.

    Returns:
        Response: Rendered network info template
    """
    return render_template('network_info.html')


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

    # Get visualization settings from config
    global show_stabilization, show_tracking, show_decision_making, show_algorithm_steps, view_mode
    show_stabilization = config.get('dashboard.visualizations.show_stabilization', True)
    show_tracking = config.get('dashboard.visualizations.show_tracking', True)
    show_decision_making = config.get('dashboard.visualizations.show_decision_making', True)
    show_algorithm_steps = config.get('dashboard.visualizations.show_algorithm_steps', True)

    try:
        while processing_active:
            # Record start time for frame rate calculation
            start_time = time.time()

            # Get frames from all cameras
            frames = stream_manager.get_all_latest_frames()

            # Get connected client count to optimize processing
            client_count = len(socketio.server.eio.sockets)

            # Skip processing if no clients are connected to save resources
            if client_count == 0:
                # Just sleep a bit to avoid CPU overload when no clients
                time.sleep(0.1)
                continue

            for camera_name, frame in frames.items():
                # Store original frame (without copying to save memory)
                last_frames[camera_name] = frame

                # Send frame through network simulator
                frame_id = random.randint(1, 10000)  # Generate a random frame ID
                routing_info = network_simulator.send_frame(camera_name, frame, frame_id)

                # Create a visualization frame that will show algorithm steps
                # Only make a copy if we need to modify it
                visualization_frame = frame.copy()

                # Add routing information to the frame
                if show_algorithm_steps:
                    routing_text = f"Routing: {routing_info['routing_algorithm']} -> {routing_info['node_id']}"
                    safe_putText(visualization_frame, routing_text, (10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 1. Video Stabilization Visualization (if enabled)
                if show_stabilization:
                    # Add optical flow visualization
                    if hasattr(stream_manager.cameras[camera_name], 'stabilizer') and \
                       hasattr(stream_manager.cameras[camera_name].stabilizer, 'prev_gray') and \
                       stream_manager.cameras[camera_name].stabilizer.prev_gray is not None:

                        # Get current frame in grayscale
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # Calculate optical flow
                        prev_gray = stream_manager.cameras[camera_name].stabilizer.prev_gray
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                        # Visualize optical flow
                        h, w = flow.shape[:2]
                        flow_vis = np.zeros((h, w, 3), dtype=np.uint8)

                        # Convert flow to polar coordinates
                        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                        # Set hue according to the angle of optical flow
                        flow_vis[..., 0] = ang * 180 / np.pi / 2

                        # Set value according to the magnitude of optical flow
                        flow_vis[..., 1] = 255
                        flow_vis[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

                        # Convert HSV to BGR for visualization
                        flow_vis = cv2.cvtColor(flow_vis, cv2.COLOR_HSV2BGR)

                        # Add optical flow visualization to corner of frame
                        h_vis, w_vis = 120, 160  # Size of visualization
                        flow_vis_resized = cv2.resize(flow_vis, (w_vis, h_vis))

                        # Create a semi-transparent overlay
                        overlay = visualization_frame.copy()
                        overlay[10:10+h_vis, 10:10+w_vis] = flow_vis_resized

                        # Add text label
                        safe_putText(overlay, "Optical Flow", (10, h_vis+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Blend with original frame
                        cv2.addWeighted(overlay, 0.7, visualization_frame, 0.3, 0, visualization_frame)

                # Run object detection
                detections = detector.detect(frame)
                last_detections[camera_name] = detections

                # 2. Object Tracking Visualization (if enabled)
                if show_tracking:
                    # Draw tracking information
                    tracked_objects = decision_maker.tracked_objects.get(camera_name, {})

                    # Draw object trajectories
                    for obj_id, obj_info in tracked_objects.items():
                        # Only process person detections
                        if obj_info['class_id'] != 0:  # 0 is person in COCO
                            continue

                        # Get trajectory points
                        trajectory = obj_info['trajectory']
                        if len(trajectory) > 1:
                            # Draw trajectory line
                            points = np.array(trajectory, dtype=np.int32)
                            cv2.polylines(visualization_frame, [points], False, (0, 255, 255), 2)

                            # Draw duration
                            duration = obj_info['duration']
                            last_point = trajectory[-1]
                            # Ensure coordinates are integers
                            text_pos = (int(last_point[0]), int(last_point[1] - 10))
                            safe_putText(visualization_frame, f"{duration:.1f}s",
                                       text_pos,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Process detections for alerts
                alerts = decision_maker.process_detections(camera_name, detections)

                # 3. Decision Making Visualization (if enabled)
                if show_decision_making:
                    # Get current algorithm settings for visualization
                    classifier_algorithm = config.get('classification.algorithm', 'svm')
                    analysis_algorithm = config.get('analysis.algorithm', 'basic')

                    # Create a semi-transparent overlay for decision visualization
                    h, w = visualization_frame.shape[:2]
                    decision_overlay = visualization_frame.copy()

                    # Draw decision making panel on the right side
                    panel_width = 200
                    panel_x = w - panel_width - 10
                    panel_y = 50
                    panel_height = 200

                    # Create gradient background for panel
                    for i in range(panel_height):
                        alpha = 0.8 - (i / panel_height * 0.2)  # Gradient from 0.8 to 0.6 opacity
                        cv2.rectangle(decision_overlay,
                                     (panel_x, panel_y + i),
                                     (panel_x + panel_width, panel_y + i + 1),
                                     (20, 20, 40), -1)

                    # Apply the overlay with transparency
                    alpha = 0.85
                    cv2.addWeighted(decision_overlay, alpha, visualization_frame, 1 - alpha, 0, visualization_frame)

                    # Draw panel border
                    cv2.rectangle(visualization_frame,
                                 (panel_x, panel_y),
                                 (panel_x + panel_width, panel_y + panel_height),
                                 (100, 100, 150), 1)

                    # Draw panel title with better styling
                    title_pos = (int(panel_x + 10), int(panel_y + 20))
                    # Draw title background
                    title_bg = visualization_frame.copy()
                    cv2.rectangle(title_bg,
                                 (panel_x, panel_y),
                                 (panel_x + panel_width, panel_y + 35),
                                 (40, 40, 80), -1)
                    cv2.addWeighted(title_bg, 0.7, visualization_frame, 0.3, 0, visualization_frame)

                    # Draw title text
                    safe_putText(visualization_frame, "Algorithm Visualization",
                               title_pos,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 255), 1)

                    # Draw algorithm info with better styling
                    # Draw classifier info with icon
                    classifier_pos = (int(panel_x + 15), int(panel_y + 45))
                    classifier_icon_size = 10
                    cv2.rectangle(visualization_frame,
                                 (panel_x + 15, panel_y + 45 - classifier_icon_size),
                                 (panel_x + 15 + classifier_icon_size, panel_y + 45),
                                 (100, 100, 255), -1)
                    safe_putText(visualization_frame, f"Classifier: {classifier_algorithm.upper()}",
                               (classifier_pos[0] + classifier_icon_size + 5, classifier_pos[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

                    # Draw analysis info with icon
                    analysis_pos = (int(panel_x + 15), int(panel_y + 65))
                    analysis_icon_size = 10
                    cv2.rectangle(visualization_frame,
                                 (panel_x + 15, panel_y + 65 - analysis_icon_size),
                                 (panel_x + 15 + analysis_icon_size, panel_y + 65),
                                 (255, 100, 100), -1)
                    safe_putText(visualization_frame, f"Analysis: {analysis_algorithm.upper()}",
                               (analysis_pos[0] + analysis_icon_size + 5, analysis_pos[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

                    # Visualize loitering detection
                    loitering_threshold = decision_maker.person_loitering_time

                    # Draw loitering threshold info with better styling
                    loitering_pos = (int(panel_x + 15), int(panel_y + 95))
                    loitering_icon_size = 10
                    cv2.rectangle(visualization_frame,
                                 (panel_x + 15, panel_y + 95 - loitering_icon_size),
                                 (panel_x + 15 + loitering_icon_size, panel_y + 95),
                                 (100, 255, 100), -1)
                    safe_putText(visualization_frame, f"Loitering: {loitering_threshold}s threshold",
                               (loitering_pos[0] + loitering_icon_size + 5, loitering_pos[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

                    # Count loitering people
                    loitering_count = 0

                    # Draw loitering zones
                    for obj_id, obj_info in decision_maker.tracked_objects.get(camera_name, {}).items():
                        # Only process person detections
                        if obj_info['class_id'] != 0:  # 0 is person in COCO
                            continue

                        duration = obj_info['duration']
                        bbox = obj_info['bbox']

                        if bbox and duration > 0:
                            # Calculate loitering progress
                            progress = min(duration / loitering_threshold, 1.0)

                            # Count loitering people
                            if progress > 0.9:
                                loitering_count += 1

                            # Draw progress bar
                            x1, _, x2, y2 = bbox  # y1 not used
                            bar_width = int((x2 - x1) * progress)

                            # Draw background
                            cv2.rectangle(visualization_frame,
                                         (int(x1), int(y2) + 5),
                                         (int(x2), int(y2) + 15),
                                         (100, 100, 100), -1)

                            # Draw progress
                            color = (0, 255, 0)  # Green
                            if progress > 0.7:
                                color = (0, 165, 255)  # Orange
                            if progress > 0.9:
                                color = (0, 0, 255)  # Red

                            cv2.rectangle(visualization_frame,
                                         (int(x1), int(y2) + 5),
                                         (int(x1) + bar_width, int(y2) + 15),
                                         color, -1)

                            # Add text with algorithm info
                            loitering_pos = (int(x1), int(y2) + 30)
                            safe_putText(visualization_frame, f"Loitering: {duration:.1f}s",
                                       loitering_pos,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                            # Add classification info
                            classifier_pos = (int(x1), int(y2) + 50)
                            if classifier_algorithm == 'svm':
                                confidence = 0.85 + (progress * 0.1)  # Simulate SVM confidence
                                safe_putText(visualization_frame, f"SVM: {confidence:.2f}",
                                          classifier_pos,
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
                            elif classifier_algorithm == 'knn':
                                neighbors = 3 if progress > 0.5 else 2
                                safe_putText(visualization_frame, f"KNN: {neighbors}/3 neighbors",
                                          classifier_pos,
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
                            elif classifier_algorithm == 'naive_bayes':
                                prob = 0.75 + (progress * 0.2)  # Simulate Naive Bayes probability
                                safe_putText(visualization_frame, f"NB: P={prob:.2f}",
                                          classifier_pos,
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)

                    # Visualize crowd detection
                    people_count = sum(1 for d in detections if hasattr(d, 'class_id') and d.class_id == 0)
                    crowd_threshold = decision_maker.crowd_threshold

                    # Draw crowd threshold info with better styling
                    crowd_threshold_pos = (int(panel_x + 15), int(panel_y + 115))
                    crowd_icon_size = 10
                    cv2.rectangle(visualization_frame,
                                 (panel_x + 15, panel_y + 115 - crowd_icon_size),
                                 (panel_x + 15 + crowd_icon_size, panel_y + 115),
                                 (255, 200, 100), -1)
                    safe_putText(visualization_frame, f"Crowd: {crowd_threshold} people max",
                               (crowd_threshold_pos[0] + crowd_icon_size + 5, crowd_threshold_pos[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

                    # Draw current people count with better styling
                    current_people_pos = (int(panel_x + 15), int(panel_y + 135))
                    people_icon_size = 10
                    # Color based on count
                    people_color = (0, 255, 0)  # Green
                    if people_count > crowd_threshold * 0.7:
                        people_color = (0, 165, 255)  # Orange
                    if people_count >= crowd_threshold:
                        people_color = (0, 0, 255)  # Red

                    cv2.rectangle(visualization_frame,
                                 (panel_x + 15, panel_y + 135 - people_icon_size),
                                 (panel_x + 15 + people_icon_size, panel_y + 135),
                                 people_color, -1)
                    safe_putText(visualization_frame, f"Current: {people_count} people",
                               (current_people_pos[0] + people_icon_size + 5, current_people_pos[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, people_color, 1)

                    # Draw loitering people count with better styling
                    loitering_people_pos = (int(panel_x + 15), int(panel_y + 155))
                    loitering_people_icon_size = 10
                    # Color based on count
                    loitering_color = (0, 255, 0)  # Green
                    if loitering_count > 0:
                        loitering_color = (0, 0, 255)  # Red

                    cv2.rectangle(visualization_frame,
                                 (panel_x + 15, panel_y + 155 - loitering_people_icon_size),
                                 (panel_x + 15 + loitering_people_icon_size, panel_y + 155),
                                 loitering_color, -1)
                    safe_putText(visualization_frame, f"Loitering: {loitering_count} people",
                               (loitering_people_pos[0] + loitering_people_icon_size + 5, loitering_people_pos[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, loitering_color, 1)

                    # Draw crowd counter
                    counter_color = (0, 255, 0)  # Green
                    if people_count > crowd_threshold * 0.7:
                        counter_color = (0, 165, 255)  # Orange
                    if people_count >= crowd_threshold:
                        counter_color = (0, 0, 255)  # Red

                    people_count_pos = (10, 30)  # Integer coordinates
                    safe_putText(visualization_frame, f"People Count: {people_count}/{crowd_threshold}",
                               people_count_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, counter_color, 2)

                    # Draw decision status based on analysis algorithm
                    decision_status = "Normal"
                    decision_color = (0, 255, 0)  # Green

                    if people_count >= crowd_threshold or loitering_count > 0:
                        if analysis_algorithm == 'basic':
                            decision_status = "Alert"
                            decision_color = (0, 0, 255)  # Red
                        elif analysis_algorithm == 'weighted':
                            # Weighted algorithm gives more importance to loitering
                            if loitering_count > 0:
                                decision_status = "Alert"
                                decision_color = (0, 0, 255)  # Red
                            elif people_count >= crowd_threshold:
                                decision_status = "Warning"
                                decision_color = (0, 165, 255)  # Orange
                        elif analysis_algorithm == 'fuzzy':
                            # Fuzzy logic based on combination of factors
                            if loitering_count > 1 or (loitering_count > 0 and people_count >= crowd_threshold):
                                decision_status = "Alert"
                                decision_color = (0, 0, 255)  # Red
                            elif loitering_count > 0 or people_count >= crowd_threshold:
                                decision_status = "Warning"
                                decision_color = (0, 165, 255)  # Orange

                    # Draw decision status with better styling
                    # Create a status bar at the bottom of the panel
                    status_bar_height = 25
                    status_bar_y = panel_y + panel_height - status_bar_height

                    # Draw status bar background
                    cv2.rectangle(visualization_frame,
                                 (panel_x, status_bar_y),
                                 (panel_x + panel_width, panel_y + panel_height),
                                 (40, 40, 40), -1)

                    # Draw status indicator
                    status_indicator_width = 5
                    cv2.rectangle(visualization_frame,
                                 (panel_x, status_bar_y),
                                 (panel_x + status_indicator_width, panel_y + panel_height),
                                 decision_color, -1)

                    # Draw status text
                    decision_status_pos = (int(panel_x + 15), int(status_bar_y + 17))
                    safe_putText(visualization_frame, f"STATUS: {decision_status.upper()}",
                               decision_status_pos,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, decision_color, 1)

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

                # Draw detections on frame - this is the only text we want on the video feed
                frame_with_detections = detector.draw_detections(visualization_frame, detections)

                # Draw tracked objects and trajectories
                if camera_name in decision_maker.tracked_objects:
                    tracked_objects = decision_maker.tracked_objects[camera_name]

                    for obj_id, obj_info in tracked_objects.items():
                        # Only track people (class_id=0)
                        if obj_info.get('class_id') != 0:  # Skip if not a person
                            continue

                        # Draw trajectory if available
                        if 'trajectory' in obj_info and len(obj_info['trajectory']) > 1:
                            # Get trajectory points and ensure they are valid
                            trajectory = obj_info['trajectory']
                            valid_trajectory = ensure_valid_trajectory_points(trajectory)

                            # Only proceed if we have valid points
                            if len(valid_trajectory) > 1:
                                # Draw trajectory line
                                for i in range(1, len(valid_trajectory)):
                                    # Use different colors based on the tracking algorithm
                                    if decision_maker.classifier_algorithm == 'knn':
                                        color = (255, 0, 255)  # Magenta for KNN
                                    elif decision_maker.classifier_algorithm == 'svm':
                                        color = (255, 255, 0)  # Yellow for SVM
                                    else:
                                        color = (0, 165, 255)  # Orange for centroid

                                    # Draw line segment with validated points
                                    pt1 = valid_trajectory[i-1]
                                    pt2 = valid_trajectory[i]
                                    cv2.line(frame_with_detections, pt1, pt2, color, 2)

                                # Draw a circle at the current position
                                last_pt = valid_trajectory[-1]
                                cv2.circle(frame_with_detections, last_pt, 5, (0, 0, 255), -1)

                # In single view mode, we don't want any additional text on the video feed
                # All algorithm info will be shown in the decision tree container below the video

                # Get current algorithm settings for metadata (needed for socket.io emit)
                detection_algorithm = config.get('detection.model', 'yolov5s')
                tracking_algorithm = config.get('tracking.algorithm', 'centroid')
                classifier_algorithm = config.get('classification.algorithm', 'svm')
                analysis_algorithm = config.get('analysis.algorithm', 'basic')

                # Get frame rate for FPS display
                frame_rate = config.get('network.frame_rate', 25)

                # Add minimal stats overlay for all view modes
                # Get frame dimensions
                h, w = frame_with_detections.shape[:2]

                # Add stats overlay at the top - only show objects count and FPS
                stats_height = 20
                cv2.rectangle(frame_with_detections, (0, 0), (w, stats_height), (0, 0, 0), -1)

                # Add minimal stats - only show detection count and FPS
                stats_text = f"Objects: {len(detections)} | {frame_rate} FPS"
                # Ensure coordinates are integers
                safe_putText(frame_with_detections, stats_text,
                          (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Make sure the frame is not empty
                if frame_with_detections is not None and frame_with_detections.size > 0:
                    # Resize the frame to reduce bandwidth (based on resolution setting)
                    resolution_setting = config.get('network.resolution', 'medium')
                    if resolution_setting == 'low':
                        target_width = 640  # 480p
                    elif resolution_setting == 'medium':
                        target_width = 854  # 720p (16:9 aspect ratio)
                    else:
                        target_width = 1280  # 1080p

                    # Calculate height to maintain aspect ratio
                    h, w = frame_with_detections.shape[:2]
                    target_height = int(h * (target_width / w))

                    # Resize only if the frame is larger than target size
                    if w > target_width:
                        frame_with_detections = cv2.resize(frame_with_detections, (target_width, target_height),
                                                          interpolation=cv2.INTER_AREA)  # Better quality downsampling

                    # Optimize JPEG encoding quality based on resolution
                    # Lower quality for better performance
                    if resolution_setting == 'low':
                        quality = 65
                    elif resolution_setting == 'medium':
                        quality = 75
                    else:
                        quality = 85

                    # Convert frame to JPEG with optimized quality
                    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                    _, buffer = cv2.imencode('.jpg', frame_with_detections, encode_params)
                    frame_bytes = buffer.tobytes()

                    # Encode as base64 for sending to clients
                    frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
                else:
                    # Create a blank frame with error message if the frame is empty
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    # Ensure coordinates are integers
                    text_position = (50, 240)  # x, y coordinates as integers
                    safe_putText(blank_frame, "No video feed available", text_position,
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Convert blank frame to JPEG
                    _, buffer = cv2.imencode('.jpg', blank_frame)
                    frame_bytes = buffer.tobytes()

                    # Encode as base64 for sending to clients
                    frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

                    # Log the error
                    logger.warning(f"Empty frame detected for camera {camera_name}")

                # Emit frame to clients
                socketio.emit('frame_update', {
                    'camera': camera_name,
                    'frame': frame_base64,
                    'detection_count': len(detections),
                    'current_algorithms': {
                        'detection': detection_algorithm,
                        'tracking': tracking_algorithm,
                        'classification': classifier_algorithm,
                        'analysis': analysis_algorithm
                    }
                })

            # Get frame rate setting
            frame_rate = config.get('network.frame_rate', 25)

            # Calculate adaptive sleep time based on processing time
            # This ensures we maintain the target frame rate even with processing overhead
            end_time = time.time()
            processing_time = end_time - start_time

            # Target time per frame - adjust based on number of cameras and client count
            # More cameras or clients = lower per-camera frame rate to avoid overloading
            num_cameras = len(frames)
            client_factor = min(1.0, 3.0 / max(1, client_count))  # Scale down as clients increase

            # Calculate adjusted frame rate - lower with more cameras/clients
            adjusted_frame_rate = max(5, frame_rate * client_factor / max(1, num_cameras / 2))
            target_time = 1.0 / adjusted_frame_rate

            # Calculate sleep time (ensure it's not negative)
            sleep_time = max(0, target_time - processing_time)

            # Log performance metrics occasionally
            if random.random() < 0.01:  # Log roughly 1% of frames
                logger.info(f"Performance: Processing time={processing_time:.3f}s, "
                           f"Target FPS={adjusted_frame_rate:.1f}, Clients={client_count}")

            # Sleep to control processing rate based on frame rate
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif processing_time > 0.1:  # If we're taking too long to process
                # We're falling behind, log a warning
                logger.warning(f"Processing falling behind: {processing_time:.3f}s per frame")

    except Exception as e:
        logger.error(f"Error in video processing loop: {str(e)}")
    finally:
        # Stop all cameras
        stream_manager.stop_all_cameras()
        logger.info("Video processing loop stopped")


def shutdown_server():
    """Shutdown the server and terminate all node processes."""
    global processing_active
    processing_active = False
    logger.info("Shutting down server...")

    # Stop all cameras
    stream_manager.stop_all_cameras()

    # Stop the processing thread
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=5)
        logger.info("Processing thread stopped")

    # Disconnect from nodes
    from vigilance_system.network.node_client import node_client
    node_client.disconnect()
    logger.info("Disconnected from network nodes")

    # Terminate all node processes
    try:
        import subprocess

        # Kill all node server processes
        logger.info("Terminating all node server processes...")
        # Kill all python processes running node_server.py
        subprocess.run(['taskkill', '/F', '/FI', 'WINDOWTITLE eq Node*'], shell=True)
        # Also try to kill by process name
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe', '/FI', 'WINDOWTITLE eq Node*'], shell=True)
        logger.info("Node server processes terminated")
    except Exception as e:
        logger.error(f"Error terminating node processes: {str(e)}")


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

    # Register shutdown handler
    import atexit
    atexit.register(shutdown_server)

    return app, socketio, host, port, debug


if __name__ == '__main__':
    app, socketio, host, port, debug = create_app()
    socketio.run(app, host=host, port=port, debug=debug)
