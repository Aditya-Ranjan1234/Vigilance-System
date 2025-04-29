"""
Analysis dashboard application for the Vigilance System.

This module provides a web-based dashboard for analyzing and comparing
the performance of different algorithms.
"""

import os
import json
from typing import Dict, Any, List, Optional

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.analysis.metrics_collector import metrics_collector
from vigilance_system.analysis.metrics_generator import metrics_generator

logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vigilance_analysis_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")


def create_app(debug: bool = False) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        debug: Whether to enable debug mode

    Returns:
        Flask: The configured Flask application
    """
    app.debug = debug

    # Register routes
    register_routes(app)

    # Register socket events
    register_socket_events(socketio)

    return app


def register_routes(app: Flask) -> None:
    """
    Register routes for the application.

    Args:
        app: The Flask application
    """
    @app.route('/')
    def index():
        """Render the main dashboard page."""
        return render_template('index.html')

    @app.route('/api/metrics')
    def get_metrics():
        """Get all metrics."""
        return jsonify(format_metrics_for_api(metrics_collector.get_all_metrics()))

    @app.route('/api/metrics/<component>')
    def get_component_metrics(component):
        """Get metrics for a specific component."""
        metrics = metrics_collector.get_all_metrics().get(component, {})
        return jsonify(format_metrics_for_api({component: metrics}))

    @app.route('/api/metrics/<component>/<metric_name>')
    def get_specific_metric(component, metric_name):
        """Get a specific metric."""
        camera_name = request.args.get('camera')
        limit = request.args.get('limit')
        if limit:
            try:
                limit = int(limit)
            except ValueError:
                limit = None

        metrics = metrics_collector.get_metrics(component, metric_name, camera_name, limit)
        return jsonify(metrics)

    @app.route('/api/algorithms')
    def get_algorithms():
        """Get available algorithms."""
        # Define default algorithms in case they're not in the config
        default_algorithms = {
            'detection': ['background_subtraction', 'mog2', 'knn', 'svm_classifier'],
            'tracking': ['klt_tracker', 'kalman_filter', 'optical_flow'],
            'loitering': ['rule_based', 'timer_threshold', 'decision_tree'],
            'crowd': ['blob_counting', 'contour_counting', 'kmeans_clustering'],
            'preprocessing': ['feature_matching', 'orb', 'sift', 'affine_transform']
        }

        # Get algorithms from config or use defaults
        algorithms = {}
        for component, defaults in default_algorithms.items():
            config_key = f'{component}.algorithms'
            if component == 'loitering':
                config_key = 'alerts.loitering.algorithms'
            elif component == 'crowd':
                config_key = 'alerts.crowd.algorithms'

            # Get from config or use defaults
            algs = config.get(config_key, {})
            if algs and hasattr(algs, 'keys'):
                algorithms[component] = list(algs.keys())
            else:
                algorithms[component] = defaults

        return jsonify(algorithms)

    @app.route('/api/current_algorithms')
    def get_current_algorithms():
        """Get currently selected algorithms."""
        current = {
            'detection': config.get('detection.algorithm', 'background_subtraction'),
            'tracking': config.get('tracking.algorithm', 'klt_tracker'),
            'loitering': config.get('alerts.loitering.algorithm', 'rule_based'),
            'crowd': config.get('alerts.crowd.algorithm', 'blob_counting'),
            'preprocessing': config.get('preprocessing.algorithm', 'feature_matching')
        }
        return jsonify(current)

    @app.route('/api/set_algorithm', methods=['POST'])
    def set_algorithm():
        """Set the current algorithm for a component."""
        data = request.json
        component = data.get('component')
        algorithm = data.get('algorithm')

        if not component or not algorithm:
            return jsonify({'error': 'Missing component or algorithm'}), 400

        # Map component to configuration key
        component_map = {
            'detection': 'detection.algorithm',
            'tracking': 'tracking.algorithm',
            'loitering': 'alerts.loitering.algorithm',
            'crowd': 'alerts.crowd.algorithm',
            'preprocessing': 'preprocessing.algorithm'
        }

        if component not in component_map:
            return jsonify({'error': f'Invalid component: {component}'}), 400

        # Set the algorithm in the configuration
        try:
            config.set(component_map[component], algorithm)
        except AttributeError:
            logger.warning(f"Could not set {component_map[component]} in config (method not available)")

        # Update the metrics generator
        metrics_generator.update_current_algorithm(component, algorithm)

        # Log the change
        logger.info(f"Changed {component} algorithm to {algorithm}")

        return jsonify({'success': True, 'component': component, 'algorithm': algorithm})


def register_socket_events(socketio: SocketIO) -> None:
    """
    Register socket events for real-time updates.

    Args:
        socketio: The SocketIO instance
    """
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        logger.info("Client connected to analysis dashboard")

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        logger.info("Client disconnected from analysis dashboard")

    @socketio.on('request_metrics_update')
    def handle_metrics_update(data):
        """
        Handle request for metrics update.

        Args:
            data: Dictionary containing component and metric names
        """
        component = data.get('component')
        metric_name = data.get('metric_name')
        camera_name = data.get('camera_name')

        if component and metric_name:
            # Get specific metric
            metrics = metrics_collector.get_metrics(component, metric_name, camera_name)
            socketio.emit('metrics_update', {
                'component': component,
                'metric_name': metric_name,
                'camera_name': camera_name,
                'metrics': metrics
            })
        elif component:
            # Get all metrics for component
            metrics = {}
            all_metrics = metrics_collector.get_all_metrics()
            if component in all_metrics:
                metrics = all_metrics[component]
            socketio.emit('metrics_update', {
                'component': component,
                'metrics': format_metrics_for_api({component: metrics})[component]
            })
        else:
            # Get all metrics
            socketio.emit('metrics_update', {
                'metrics': format_metrics_for_api(metrics_collector.get_all_metrics())
            })


def format_metrics_for_api(metrics: Dict[str, Dict[str, List]]) -> Dict[str, Any]:
    """
    Format metrics for API response.

    Args:
        metrics: Raw metrics from the collector

    Returns:
        Formatted metrics for API response
    """
    formatted = {}

    for component, component_metrics in metrics.items():
        formatted[component] = {}
        for key, values in component_metrics.items():
            # Split camera_name and metric_name if needed
            parts = key.split('_', 1)
            if len(parts) > 1 and parts[0]:
                camera_name = parts[0]
                metric_name = parts[1]

                if camera_name not in formatted[component]:
                    formatted[component][camera_name] = {}

                formatted[component][camera_name][metric_name] = values
            else:
                # No camera name
                formatted[component][key] = values

    return formatted


def run_analysis_dashboard(host: str = '0.0.0.0', port: int = 5001, debug: bool = False) -> None:
    """
    Run the analysis dashboard.

    Args:
        host: Host to run the server on
        port: Port to run the server on
        debug: Whether to enable debug mode
    """
    # Start the metrics generator
    metrics_generator.start()

    try:
        app_instance = create_app(debug)
        socketio.run(app_instance, host=host, port=port)
    finally:
        # Stop the metrics generator when the server stops
        metrics_generator.stop()
