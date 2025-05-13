"""
Main entry point for the vigilance system.

This module initializes and starts all components of the vigilance system.
"""

import os
import sys
import argparse
from pathlib import Path

from vigilance_system.utils.logger import get_logger, setup_logger
from vigilance_system.utils.config import config
from vigilance_system.dashboard.app import create_app

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

    return parser.parse_args()


def main():
    """Main entry point for the vigilance system."""
    # Parse command line arguments
    args = parse_args()

    # Setup logger with command line override
    setup_logger(args.log_level)

    # Log startup information
    logger.info("Starting Vigilance System")

    # Display helpful information about video locations
    surveillance_dir = 'D:\\Main EL\\videos\\surveillance'
    if not os.path.exists(surveillance_dir):
        try:
            os.makedirs(surveillance_dir, exist_ok=True)
            logger.info(f"Created surveillance directory at {surveillance_dir}")
        except Exception as e:
            logger.error(f"Failed to create surveillance directory: {str(e)}")

    # Print a clear message to the console about where to place videos
    print("\n" + "="*80)
    print(f"IMPORTANT: Place your videos in: {surveillance_dir}")
    print("="*80 + "\n")

    try:
        # Create and run the dashboard app
        app, socketio, host, port, debug = create_app()

        # Override with command line arguments if provided
        if args.host:
            host = args.host
        if args.port:
            port = args.port

        logger.info(f"Starting dashboard on {host}:{port}")
        print(f"Dashboard available at: http://{host}:{port}")
        print("Press Ctrl+C to stop the server\n")
        socketio.run(app, host=host, port=port, debug=debug)

    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
