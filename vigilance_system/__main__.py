"""
Main entry point for the vigilance system.

This module initializes and starts all components of the vigilance system.
"""

import os
import sys
import argparse
import threading
from pathlib import Path

from vigilance_system.utils.logger import get_logger, setup_logger
from vigilance_system.utils.config import config
from vigilance_system.dashboard.app import create_app
from vigilance_system.analysis.dashboard import analysis_dashboard

# Initialize logger
logger = get_logger(__name__)


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Camera-Only Vigilance System')

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )

    parser.add_argument(
        '--host',
        type=str,
        help='Dashboard host address'
    )

    parser.add_argument(
        '--port',
        type=int,
        help='Dashboard port'
    )

    # Algorithm selection arguments
    parser.add_argument(
        '--detection-algorithm',
        type=str,
        choices=[
            # Non-deep learning algorithms
            'background_subtraction', 'mog2', 'knn', 'svm_classifier',
            # Legacy deep learning algorithms (for backward compatibility)
            'yolov5', 'yolov8', 'ssd', 'faster_rcnn'
        ],
        help='Object detection algorithm to use'
    )

    parser.add_argument(
        '--tracking-algorithm',
        type=str,
        choices=[
            # Non-deep learning algorithms
            'klt_tracker', 'kalman_filter', 'optical_flow',
            # Legacy deep learning algorithms (for backward compatibility)
            'sort', 'deep_sort', 'iou'
        ],
        help='Object tracking algorithm to use'
    )

    parser.add_argument(
        '--loitering-algorithm',
        type=str,
        choices=[
            # Non-deep learning algorithms
            'rule_based', 'timer_threshold', 'decision_tree',
            # Legacy deep learning algorithms (for backward compatibility)
            'time_threshold', 'trajectory_heatmap', 'lstm_prediction'
        ],
        help='Loitering detection algorithm to use'
    )

    parser.add_argument(
        '--crowd-algorithm',
        type=str,
        choices=[
            # Non-deep learning algorithms
            'blob_counting', 'contour_counting', 'kmeans_clustering',
            # Legacy deep learning algorithms (for backward compatibility)
            'count_threshold', 'density_map', 'clustering'
        ],
        help='Crowd detection algorithm to use'
    )

    parser.add_argument(
        '--preprocessing-algorithm',
        type=str,
        choices=[
            # Non-deep learning algorithms
            'feature_matching', 'orb', 'sift', 'affine_transform',
            # Legacy deep learning algorithms (for backward compatibility)
            'optical_flow', 'feature_based', 'deep_learning'
        ],
        help='Video preprocessing algorithm to use'
    )

    # Analysis dashboard arguments
    parser.add_argument(
        '--enable-analysis',
        action='store_true',
        help='Enable the analysis dashboard'
    )

    parser.add_argument(
        '--analysis-port',
        type=int,
        help='Analysis dashboard port'
    )

    return parser.parse_args()


def main():
    """Main entry point for the vigilance system."""
    # Parse command line arguments
    args = parse_args()

    # Setup logger with command line override
    setup_logger(args.log_level)

    # Log startup information
    logger.info("Starting Vigilance System")

    try:
        # Override configuration with command line arguments
        try:
            if args.detection_algorithm:
                config.set('detection.algorithm', args.detection_algorithm)
                config.set('detection.use_algorithm_detectors', True)
                logger.info(f"Using detection algorithm: {args.detection_algorithm}")

            if args.tracking_algorithm:
                config.set('tracking.algorithm', args.tracking_algorithm)
                logger.info(f"Using tracking algorithm: {args.tracking_algorithm}")

            if args.loitering_algorithm:
                config.set('alerts.loitering.algorithm', args.loitering_algorithm)
                logger.info(f"Using loitering detection algorithm: {args.loitering_algorithm}")

            if args.crowd_algorithm:
                config.set('alerts.crowd.algorithm', args.crowd_algorithm)
                logger.info(f"Using crowd detection algorithm: {args.crowd_algorithm}")

            if args.preprocessing_algorithm:
                config.set('preprocessing.algorithm', args.preprocessing_algorithm)
                logger.info(f"Using preprocessing algorithm: {args.preprocessing_algorithm}")

            # Enable analysis dashboard if requested
            if args.enable_analysis:
                config.set('analysis_dashboard.enabled', True)
                logger.info("Analysis dashboard enabled")

            if args.analysis_port:
                config.set('analysis_dashboard.port', args.analysis_port)
                logger.info(f"Analysis dashboard port set to {args.analysis_port}")
        except AttributeError:
            logger.error("Config object does not have 'set' method. Please update the config utility.")
            logger.warning("Continuing with default configuration.")

        # Start the analysis dashboard in a separate thread if enabled
        # Check if analysis dashboard should be enabled (either from config or command line args)
        analysis_enabled = config.get('analysis_dashboard.enabled', False) or args.enable_analysis

        if analysis_enabled:
            # Set default port if not specified
            if not config.get('analysis_dashboard.port') and args.analysis_port:
                try:
                    config.set('analysis_dashboard.port', args.analysis_port)
                except AttributeError:
                    pass

            analysis_thread = threading.Thread(target=analysis_dashboard.start)
            analysis_thread.daemon = True
            analysis_thread.start()
            logger.info(f"Started analysis dashboard in background thread on port {config.get('analysis_dashboard.port', 5001)}")

        # Create and run the main dashboard app
        app, socketio, host, port, debug = create_app()

        # Override with command line arguments if provided
        if args.host:
            host = args.host
        if args.port:
            port = args.port

        logger.info(f"Starting dashboard on {host}:{port}")
        socketio.run(app, host=host, port=port, debug=debug)

    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt")
        # Check if analysis dashboard should be enabled (either from config or command line args)
        analysis_enabled = config.get('analysis_dashboard.enabled', False) or args.enable_analysis
        if analysis_enabled:
            analysis_dashboard.stop()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
