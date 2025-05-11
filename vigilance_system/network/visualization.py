"""
Network visualization module for displaying routing algorithms.

This module provides functionality to visualize the network routing
and display statistics about the network performance.
"""

import numpy as np
import cv2
from typing import List
import time
import math
import colorsys

from vigilance_system.utils.logger import get_logger
from vigilance_system.utils.cv_utils import safe_putText
from vigilance_system.network.simulation import network_simulator

# Initialize logger
logger = get_logger(__name__)


class NetworkVisualizer:
    """Visualizes network routing and statistics."""

    def __init__(self, width: int = 1024, height: int = 768):
        """
        Initialize the network visualizer.

        Args:
            width: Width of the visualization
            height: Height of the visualization
        """
        self.width = width
        self.height = height
        self.background_color = (15, 15, 25)  # Dark blue background
        self.node_color = (60, 180, 60)  # Brighter green for nodes
        self.camera_color = (60, 60, 180)  # Brighter blue for cameras
        self.text_color = (220, 220, 220)  # Light text for dark background
        self.line_color = (70, 70, 90)  # Slightly brighter lines
        self.active_line_color = (120, 220, 120)  # Brighter active lines

        # Packet colors for different cameras
        self.packet_colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
        ]

        # Node and camera positions
        self.node_positions = {}
        self.camera_positions = {}

        # Active connections
        self.active_connections = {}

        # Animation state
        self.packet_positions = {}
        self.packet_speeds = {}
        self.frame_count = 0

        # Statistics
        self.stats = {}
        self.last_update_time = time.time()

        logger.info(f"Initialized network visualizer with size {width}x{height}")

    def update(self) -> None:
        """Update the visualization state."""
        # Get latest network stats
        self.stats = network_simulator.get_stats()

        # Force refresh of routing algorithm from node client
        try:
            from vigilance_system.network.node_client import node_client
            current_algorithm = node_client.current_algorithm
            # Update the stats with the current algorithm from node_client
            self.stats['routing_algorithm'] = current_algorithm
            # Also update the simulator with the current algorithm
            network_simulator.set_routing_algorithm(current_algorithm)
        except Exception as e:
            logger.error(f"Error updating routing algorithm in visualization: {str(e)}")

        # Get client stats
        client_stats = self.stats.get('client_stats', {})

        # Update node positions if needed
        num_nodes = client_stats.get('nodes', 0)
        node_ids = [f"node_{i+1}" for i in range(num_nodes)]

        if set(node_ids) != set(self.node_positions.keys()):
            self._calculate_node_positions(node_ids)

        # Update camera positions if needed
        cameras = self.stats.get('cameras', {})
        if set(cameras.keys()) != set(self.camera_positions.keys()):
            self._calculate_camera_positions(list(cameras.keys()))

        # Update active connections
        for camera_id, camera_info in cameras.items():
            last_node = camera_info.get('last_node')
            if last_node:
                self.active_connections[camera_id] = last_node

                # Create packet animation if not exists
                if camera_id not in self.packet_positions:
                    self.packet_positions[camera_id] = 0.0
                    self.packet_speeds[camera_id] = 0.05 + 0.05 * np.random.random()

        # Update packet animations
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time
        self.last_update_time = current_time

        for camera_id in list(self.packet_positions.keys()):
            if camera_id in self.active_connections:
                # Move packet along the connection
                self.packet_positions[camera_id] += self.packet_speeds[camera_id] * elapsed_time * 5

                # Reset when reaching the end
                if self.packet_positions[camera_id] >= 1.0:
                    self.packet_positions[camera_id] = 0.0
            else:
                # Remove inactive connections
                del self.packet_positions[camera_id]
                if camera_id in self.packet_speeds:
                    del self.packet_speeds[camera_id]

    def draw(self) -> np.ndarray:
        """
        Draw the network visualization.

        Returns:
            np.ndarray: Visualization image
        """
        # Create background
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array(self.background_color, dtype=np.uint8)

        # Draw connections
        for camera_id, node_id in self.active_connections.items():
            if camera_id in self.camera_positions and node_id in self.node_positions:
                camera_pos = self.camera_positions[camera_id]
                node_pos = self.node_positions[node_id]

                # Draw connection line
                cv2.line(image, camera_pos, node_pos, self.line_color, 1, cv2.LINE_AA)

                # Draw packet animation
                if camera_id in self.packet_positions:
                    packet_pos = self.packet_positions[camera_id]
                    x = int(camera_pos[0] + (node_pos[0] - camera_pos[0]) * packet_pos)
                    y = int(camera_pos[1] + (node_pos[1] - camera_pos[1]) * packet_pos)

                    # Use different colors for different cameras
                    color_index = hash(camera_id) % len(self.packet_colors)
                    packet_color = self.packet_colors[color_index]

                    # Draw packet with tail for better visibility
                    cv2.circle(image, (x, y), 6, packet_color, -1, cv2.LINE_AA)

                    # Draw a tail behind the packet
                    tail_length = 3
                    for i in range(1, tail_length + 1):
                        tail_pos = packet_pos - (i * 0.05)
                        if tail_pos >= 0:
                            tail_x = int(camera_pos[0] + (node_pos[0] - camera_pos[0]) * tail_pos)
                            tail_y = int(camera_pos[1] + (node_pos[1] - camera_pos[1]) * tail_pos)
                            tail_alpha = 1.0 - (i / (tail_length + 1))
                            tail_color = tuple([int(c * tail_alpha) for c in packet_color])
                            cv2.circle(image, (tail_x, tail_y), 6 - i, tail_color, -1, cv2.LINE_AA)

        # Draw nodes
        for node_id, pos in self.node_positions.items():
            # Get node load from network simulator if available
            node_load = 0.0

            # Get node info from network simulator
            node_info = network_simulator.nodes.get(node_id, {})

            # Calculate load based on active connections
            if node_id in self.active_connections.values():
                # Count how many cameras are using this node
                connection_count = list(self.active_connections.values()).count(node_id)
                # Calculate load based on connection count and node capacity
                capacity = node_info.get('capacity', 1.0)
                node_load = min(0.95, connection_count / (capacity * 2))
            else:
                # Default load for visualization
                node_load = 0.1 + 0.1 * np.random.random()

            # Store load in node info for future reference
            if node_id in network_simulator.nodes:
                network_simulator.nodes[node_id]['load'] = node_load

            # Use the calculated load
            load = node_load

            # Check if this is a simulated node
            is_simulated = "simulated" in node_id.lower() or node_id not in self.stats.get('client_stats', {}).get('real_node_ids', [])

            # Draw terminal-like rectangle for node with different color for simulated nodes
            bg_color = (20, 20, 40) if is_simulated else (30, 30, 30)
            border_color = (80, 80, 160) if is_simulated else (100, 100, 100)

            cv2.rectangle(image, (pos[0] - 30, pos[1] - 20), (pos[0] + 30, pos[1] + 20), bg_color, -1, cv2.LINE_AA)
            cv2.rectangle(image, (pos[0] - 30, pos[1] - 20), (pos[0] + 30, pos[1] + 20), border_color, 2, cv2.LINE_AA)

            # Draw terminal title bar
            title_color = (50, 50, 180) if is_simulated else (50, 50, 150)
            cv2.rectangle(image, (pos[0] - 30, pos[1] - 20), (pos[0] + 30, pos[1] - 12), title_color, -1, cv2.LINE_AA)

            # Draw node label in title bar
            label = f"{node_id}" + (" (SIM)" if is_simulated else "")
            safe_putText(image, label, (pos[0] - 25, pos[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw terminal content with real IP if available
            if 'client_ip' in self.stats:
                ip_address = self.stats.get('client_ip', '127.0.0.1')
            else:
                ip_address = "127.0.0.1"

            safe_putText(image, ip_address, (pos[0] - 25, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            safe_putText(image, f"Load: {load:.1%}", (pos[0] - 25, pos[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # Draw load indicator bar below terminal
            cv2.rectangle(image, (pos[0] - 30, pos[1] + 25), (pos[0] + 30, pos[1] + 30), (50, 50, 50), -1, cv2.LINE_AA)
            cv2.rectangle(image, (pos[0] - 30, pos[1] + 25), (pos[0] - 30 + int(60 * load), pos[1] + 30), (50, 200, 50), -1, cv2.LINE_AA)

            # Draw node label below load bar
            safe_putText(image, f"{node_id} ({load:.0%})", (pos[0] - 25, pos[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1, cv2.LINE_AA)

        # Draw cameras
        for camera_id, pos in self.camera_positions.items():
            # Draw camera icon
            cv2.rectangle(image, (pos[0] - 15, pos[1] - 10), (pos[0] + 15, pos[1] + 10), self.camera_color, -1, cv2.LINE_AA)
            cv2.rectangle(image, (pos[0] - 15, pos[1] - 10), (pos[0] + 15, pos[1] + 10), (30, 30, 100), 2, cv2.LINE_AA)
            cv2.circle(image, (pos[0] + 20, pos[1]), 5, self.camera_color, -1, cv2.LINE_AA)

            # Draw camera label
            safe_putText(image, camera_id, (pos[0], pos[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1, cv2.LINE_AA)

        # Draw statistics
        self._draw_statistics(image)

        return image

    def _calculate_node_positions(self, node_ids: List[str]) -> None:
        """
        Calculate positions for nodes in a circular layout.

        Args:
            node_ids: List of node IDs
        """
        num_nodes = len(node_ids)
        if num_nodes == 0:
            return

        # Calculate positions in a circle
        center_x = self.width // 2
        center_y = self.height // 2
        radius = min(self.width, self.height) // 3

        for i, node_id in enumerate(node_ids):
            angle = 2 * math.pi * i / num_nodes
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            self.node_positions[node_id] = (x, y)

    def _calculate_camera_positions(self, camera_ids: List[str]) -> None:
        """
        Calculate positions for cameras in a circular layout.

        Args:
            camera_ids: List of camera IDs
        """
        num_cameras = len(camera_ids)
        if num_cameras == 0:
            return

        # Calculate positions in a larger circle
        center_x = self.width // 2
        center_y = self.height // 2
        radius = min(self.width, self.height) // 2 - 50

        for i, camera_id in enumerate(camera_ids):
            angle = 2 * math.pi * i / num_cameras
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            self.camera_positions[camera_id] = (x, y)

    def _draw_statistics(self, image: np.ndarray) -> None:
        """
        Draw network statistics on the image.

        Args:
            image: Image to draw on
        """
        # Increment frame count
        self.frame_count += 1

        # Draw semi-transparent overlay for stats
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # Draw title with highlight
        algorithm = self.stats.get('routing_algorithm', 'unknown').upper()
        title = f"Network Routing: {algorithm}"

        # Draw title shadow for better visibility
        safe_putText(image, title, (22, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw title with color that changes over time for attention
        # Use different hue ranges for different algorithms to make changes more visible
        algorithm_hue_offsets = {
            'DIRECT': 0.0,
            'ROUND_ROBIN': 0.2,
            'LEAST_CONNECTION': 0.4,
            'WEIGHTED': 0.6,
            'IP_HASH': 0.8
        }

        # Get base hue for the algorithm
        base_hue = algorithm_hue_offsets.get(algorithm, 0.0)
        # Add oscillation for animation
        hue = base_hue + 0.1 * math.sin(self.frame_count / 10.0)
        # Ensure hue is in valid range
        hue = hue % 1.0

        title_color = tuple([int(c) for c in colorsys.hsv_to_rgb(hue, 0.7, 1.0) * np.array([255, 255, 255])])
        safe_putText(image, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2, cv2.LINE_AA)

        # Draw statistics with colored indicators
        stats = [
            ("Frame Rate", f"{self.stats.get('actual_frame_rate', 0):.1f} fps", (100, 255, 100)),
            ("Bandwidth", f"{self.stats.get('bandwidth', 0):.2f} MB/s", (100, 100, 255)),
            ("Packet Loss", f"{self.stats.get('packet_loss', 0):.1f}%", (255, 100, 100)),
            ("Avg Latency", f"{self.stats.get('avg_latency', 0) * 1000:.1f} ms", (255, 255, 100)),
            ("Jitter", f"{self.stats.get('jitter', 0) * 1000:.1f} ms", (255, 100, 255))
        ]

        for i, (label, value, color) in enumerate(stats):
            # Draw label
            safe_putText(image, f"{label}:", (20, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # Draw value with color
            safe_putText(image, value, (150, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Draw semi-transparent overlay for algorithm description
        overlay = image.copy()
        cv2.rectangle(overlay, (10, self.height - 80), (self.width - 10, self.height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # Draw algorithm description
        algorithm = self.stats.get('routing_algorithm', '')
        description = self._get_algorithm_description(algorithm)

        # Draw description at the bottom with highlight
        y_pos = self.height - 60
        for line in description.split('\n'):
            # Draw shadow for better visibility
            safe_putText(image, line, (22, y_pos + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw text with bright color
            safe_putText(image, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            y_pos += 20

    def _get_algorithm_description(self, algorithm: str) -> str:
        """
        Get description for a routing algorithm.

        Args:
            algorithm: Name of the algorithm

        Returns:
            str: Description of the algorithm
        """
        descriptions = {
            'direct': "Direct Connection: All traffic goes to a single node.",
            'round_robin': "Round Robin: Distributes traffic evenly across all nodes in sequence.",
            'least_connection': "Least Connection: Sends traffic to the node with the fewest active connections.",
            'weighted': "Weighted Distribution: Routes based on node capacity weights.",
            'ip_hash': "IP Hash: Consistently routes the same camera to the same node."
        }

        return descriptions.get(algorithm, "Unknown algorithm")


# Create a default instance
network_visualizer = NetworkVisualizer()
