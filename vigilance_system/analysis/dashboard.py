"""
Analysis dashboard for the Vigilance System.

This module provides a web-based dashboard for analyzing and comparing the performance
of different algorithms.
"""

import threading
from typing import Optional

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.analysis.app import run_analysis_dashboard
from vigilance_system.analysis.metrics_generator import metrics_generator

logger = get_logger(__name__)

# Global variables
_server_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


class AnalysisDashboard:
    """
    Analysis dashboard for the Vigilance System.

    This class provides methods to start and stop the analysis dashboard.
    """

    def start(self) -> None:
        """Start the analysis dashboard."""
        global _server_thread, _stop_event

        if _server_thread and _server_thread.is_alive():
            logger.warning("Analysis dashboard is already running")
            return

        # Check if dashboard is enabled
        if not config.get('analysis_dashboard.enabled', False):
            logger.info("Analysis dashboard is disabled in configuration")
            return

        # Reset stop event
        _stop_event.clear()

        # Get configuration
        host = config.get('analysis_dashboard.host', '0.0.0.0')
        port = config.get('analysis_dashboard.port', 5001)
        debug = config.get('analysis_dashboard.debug', False)

        # Start dashboard in a new thread
        _server_thread = threading.Thread(
            target=run_analysis_dashboard,
            args=(host, port, debug),
            daemon=True
        )
        _server_thread.start()

        logger.info(f"Analysis dashboard started on http://{host}:{port}")

    def stop(self) -> None:
        """Stop the analysis dashboard."""
        global _server_thread, _stop_event

        if not _server_thread or not _server_thread.is_alive():
            logger.warning("Analysis dashboard is not running")
            return

        # Set stop event
        _stop_event.set()

        # Stop the metrics generator
        metrics_generator.stop()

        # Wait for thread to terminate
        _server_thread.join(timeout=5.0)

        if _server_thread.is_alive():
            logger.warning("Analysis dashboard thread did not terminate gracefully")
        else:
            logger.info("Analysis dashboard stopped")

        _server_thread = None


# Create a global instance
analysis_dashboard = AnalysisDashboard()
