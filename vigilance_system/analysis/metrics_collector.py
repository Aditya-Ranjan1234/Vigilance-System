"""
Metrics collector for the Vigilance System.

This module collects and stores performance metrics for different algorithms.
"""

import time
import threading
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """
    Collects and stores performance metrics for different algorithms.
    
    This class is responsible for collecting metrics from different components
    of the system and storing them for analysis.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one metrics collector exists."""
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the metrics collector."""
        if self._initialized:
            return
            
        self._metrics = defaultdict(lambda: defaultdict(lambda: deque(maxlen=config.get('analysis_dashboard.visualization.history_length', 100))))
        self._lock = threading.Lock()
        self._initialized = True
        
        logger.info("Metrics collector initialized")
    
    def add_metric(self, component: str, metric_name: str, value: float, camera_name: Optional[str] = None) -> None:
        """
        Add a metric value for a specific component.
        
        Args:
            component: The component name (e.g., 'detection', 'tracking')
            metric_name: The name of the metric (e.g., 'fps', 'map')
            value: The metric value
            camera_name: Optional camera name for per-camera metrics
        """
        with self._lock:
            timestamp = time.time()
            key = f"{camera_name}_{metric_name}" if camera_name else metric_name
            self._metrics[component][key].append((timestamp, value))
    
    def get_metrics(self, component: str, metric_name: str, camera_name: Optional[str] = None, 
                   limit: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        Get metrics for a specific component.
        
        Args:
            component: The component name (e.g., 'detection', 'tracking')
            metric_name: The name of the metric (e.g., 'fps', 'map')
            camera_name: Optional camera name for per-camera metrics
            limit: Optional limit on the number of data points to return
        
        Returns:
            List of (timestamp, value) tuples
        """
        with self._lock:
            key = f"{camera_name}_{metric_name}" if camera_name else metric_name
            metrics = list(self._metrics[component][key])
            if limit is not None and limit > 0:
                metrics = metrics[-limit:]
            return metrics
    
    def get_latest_metric(self, component: str, metric_name: str, camera_name: Optional[str] = None) -> Optional[float]:
        """
        Get the latest metric value for a specific component.
        
        Args:
            component: The component name (e.g., 'detection', 'tracking')
            metric_name: The name of the metric (e.g., 'fps', 'map')
            camera_name: Optional camera name for per-camera metrics
        
        Returns:
            The latest metric value or None if no metrics are available
        """
        metrics = self.get_metrics(component, metric_name, camera_name, limit=1)
        if metrics:
            return metrics[0][1]
        return None
    
    def get_average_metric(self, component: str, metric_name: str, camera_name: Optional[str] = None,
                          window: int = 10) -> Optional[float]:
        """
        Get the average metric value over a window of time.
        
        Args:
            component: The component name (e.g., 'detection', 'tracking')
            metric_name: The name of the metric (e.g., 'fps', 'map')
            camera_name: Optional camera name for per-camera metrics
            window: Number of data points to average over
        
        Returns:
            The average metric value or None if no metrics are available
        """
        metrics = self.get_metrics(component, metric_name, camera_name, limit=window)
        if metrics:
            values = [m[1] for m in metrics]
            return np.mean(values)
        return None
    
    def get_all_metrics(self) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
        """
        Get all metrics.
        
        Returns:
            Dictionary of all metrics
        """
        with self._lock:
            return {component: dict(metrics) for component, metrics in self._metrics.items()}
    
    def clear_metrics(self, component: Optional[str] = None, metric_name: Optional[str] = None,
                     camera_name: Optional[str] = None) -> None:
        """
        Clear metrics.
        
        Args:
            component: Optional component name to clear metrics for
            metric_name: Optional metric name to clear
            camera_name: Optional camera name to clear metrics for
        """
        with self._lock:
            if component is None:
                self._metrics.clear()
            elif metric_name is None:
                self._metrics[component].clear()
            else:
                key = f"{camera_name}_{metric_name}" if camera_name else metric_name
                self._metrics[component][key].clear()


# Create a global instance
metrics_collector = MetricsCollector()
