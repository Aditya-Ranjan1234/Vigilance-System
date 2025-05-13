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
import random
import os

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

        # Packet colors for different cameras - brighter colors for better visibility
        self.packet_colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
            (255, 150, 0),    # Orange
            (180, 0, 255),    # Purple
            (0, 255, 200),    # Teal
            (255, 0, 150),    # Pink
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
        # Store previous algorithm to detect changes
        previous_algorithm = self.stats.get('routing_algorithm', 'unknown')

        # Get latest network stats
        self.stats = network_simulator.get_stats()

        # Force refresh of routing algorithm from node client
        try:
            from vigilance_system.network.node_client import node_client

            # First check the current_algorithm.txt file - this is the source of truth
            algorithm_file = os.path.join(os.getcwd(), 'current_algorithm.txt')
            file_algorithm = None

            # Try to read the algorithm from file
            if os.path.exists(algorithm_file):
                try:
                    with open(algorithm_file, 'r') as f:
                        file_algorithm = f.read().strip()
                        # Validate the algorithm
                        valid_algorithms = ['direct', 'round_robin', 'least_connection', 'weighted']
                        if file_algorithm not in valid_algorithms:
                            logger.warning(f"Invalid algorithm in file: {file_algorithm}, using direct instead")
                            file_algorithm = "direct"
                            # Fix the file
                            with open(algorithm_file, 'w') as f:
                                f.write("direct")
                        logger.info(f"Read algorithm from file: {file_algorithm}")
                except Exception as file_error:
                    logger.error(f"Error reading algorithm file: {str(file_error)}")
                    # If there's an error reading, create a new file with default algorithm
                    try:
                        with open(algorithm_file, 'w') as f:
                            f.write("direct")
                        file_algorithm = "direct"
                        logger.info(f"Created new algorithm file with default algorithm after read error")
                    except Exception as write_error:
                        logger.error(f"Error creating algorithm file after read error: {str(write_error)}")
            else:
                # If file doesn't exist, create it with default algorithm
                try:
                    with open(algorithm_file, 'w') as f:
                        f.write("direct")
                    file_algorithm = "direct"
                    logger.info(f"Created algorithm file with default algorithm: {file_algorithm}")
                except Exception as file_error:
                    logger.error(f"Error creating algorithm file: {str(file_error)}")

            # Get the algorithm from node_client
            client_algorithm = node_client.current_algorithm

            # ALWAYS use the file algorithm as the source of truth if available
            if file_algorithm:
                # If there's a mismatch, update node_client to match the file
                if file_algorithm != client_algorithm:
                    logger.warning(f"Algorithm mismatch: file={file_algorithm}, client={client_algorithm}")
                    # Update node_client to match the file
                    try:
                        node_client.set_algorithm(file_algorithm)
                    except Exception as e:
                        logger.error(f"Error updating node_client algorithm: {str(e)}")
                current_algorithm = file_algorithm
            else:
                # If we couldn't read from file, use client algorithm
                current_algorithm = client_algorithm

                # Try to write it to file for future consistency
                try:
                    with open(algorithm_file, 'w') as f:
                        f.write(current_algorithm)
                    logger.info(f"Saved client algorithm to file: {current_algorithm}")
                except Exception as e:
                    logger.error(f"Error saving client algorithm to file: {str(e)}")

            # Update the stats with the current algorithm
            self.stats['routing_algorithm'] = current_algorithm

            # Also update the simulator with the current algorithm
            network_simulator.set_routing_algorithm(current_algorithm)

            # If algorithm changed, reset active connections to force new routing pattern
            if previous_algorithm != current_algorithm:
                logger.info(f"Algorithm changed from {previous_algorithm} to {current_algorithm}, resetting connections")
                self.active_connections = {}
                self.packet_positions = {}
                self.packet_speeds = {}
                # Force node client to reset connections
                node_client.camera_last_nodes = {}

                # Force network simulator to update routing
                try:
                    # Update the routing algorithm in the simulator
                    network_simulator.set_routing_algorithm(current_algorithm)

                    # Wait a moment for the algorithm to take effect
                    time.sleep(0.5)

                    # Force recalculation of connections based on new algorithm
                    # This is critical for the visualization to update properly
                    self._calculate_connections()

                    # Log the new connections
                    logger.info(f"New connections after algorithm change: {self.active_connections}")

                    # Update node_client with the new connections
                    node_client.camera_last_nodes = self.active_connections.copy()
                except Exception as e:
                    logger.error(f"Error updating network simulator algorithm: {str(e)}")
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

        # Update active connections - get real connections from node_client
        try:
            from vigilance_system.network.node_client import node_client
            client_stats = node_client.get_stats()

            # Get the actual camera-to-node mappings from node_client
            camera_last_nodes = node_client.camera_last_nodes

            # Log the camera_last_nodes for debugging (only at debug level)
            logger.debug(f"Camera to node mappings: {camera_last_nodes}")

            # Check if we have any active connections
            if not camera_last_nodes:
                logger.info("No active connections found in node_client, recalculating connections")
                # Force recalculation of connections based on current algorithm
                self._calculate_connections()
            else:
                # Get current algorithm for special handling
                current_algorithm = self.stats.get('routing_algorithm', 'direct')

                # Handle algorithm-specific virtual connections
                if current_algorithm == 'direct':
                    # Direct: Keep existing virtual connections (those with _direct_ in the name)
                    preserved_connections = {k: v for k, v in self.active_connections.items() if '_direct_' in k}
                    self.active_connections = preserved_connections

                    # If we don't have any virtual connections yet, force a recalculation
                    if not preserved_connections:
                        logger.info("No virtual Direct connections found, recalculating")
                        self._calculate_connections()
                        return

                elif current_algorithm == 'round_robin':
                    # Round Robin: Keep existing virtual connections (those with _rr_ in the name)
                    preserved_connections = {k: v for k, v in self.active_connections.items() if '_rr_' in k}
                    self.active_connections = preserved_connections

                    # If we don't have any virtual connections yet, force a recalculation
                    if not preserved_connections:
                        logger.info("No virtual Round Robin connections found, recalculating")
                        self._calculate_connections()
                        return

                elif current_algorithm == 'least_connection':
                    # Least Connection: Keep existing virtual connections (those with _lc_ in the name)
                    preserved_connections = {k: v for k, v in self.active_connections.items() if '_lc_' in k}
                    self.active_connections = preserved_connections

                    # If we don't have any virtual connections yet, force a recalculation
                    if not preserved_connections:
                        logger.info("No virtual Least Connection connections found, recalculating")
                        self._calculate_connections()
                        return

                elif current_algorithm == 'weighted':
                    # Weighted: Keep existing virtual connections (those with _weighted_ in the name)
                    preserved_connections = {k: v for k, v in self.active_connections.items() if '_weighted_' in k}
                    self.active_connections = preserved_connections

                    # If we don't have any virtual connections yet, force a recalculation
                    if not preserved_connections:
                        logger.info("No virtual Weighted connections found, recalculating")
                        self._calculate_connections()
                        return
                else:
                    # For other algorithms, clear all connections
                    self.active_connections = {}

                # Update active connections based on real data
                for camera_id, node_id in camera_last_nodes.items():
                    if node_id:
                        self.active_connections[camera_id] = node_id

                        # Create packet animation if not exists
                        if camera_id not in self.packet_positions:
                            self.packet_positions[camera_id] = 0.0
                            self.packet_speeds[camera_id] = 0.05 + 0.05 * np.random.random()

                # Check if we need to recalculate virtual connections for the current algorithm
                if current_algorithm == 'direct':
                    has_virtual_connections = any('_direct_' in k for k in self.active_connections.keys())
                    if not has_virtual_connections:
                        logger.info("Direct algorithm active but no virtual connections found, recalculating")
                        self._calculate_connections()

                elif current_algorithm == 'round_robin':
                    has_virtual_connections = any('_rr_' in k for k in self.active_connections.keys())
                    if not has_virtual_connections:
                        logger.info("Round Robin algorithm active but no virtual connections found, recalculating")
                        self._calculate_connections()

                elif current_algorithm == 'least_connection':
                    has_virtual_connections = any('_lc_' in k for k in self.active_connections.keys())
                    if not has_virtual_connections:
                        logger.info("Least Connection algorithm active but no virtual connections found, recalculating")
                        self._calculate_connections()

                elif current_algorithm == 'weighted':
                    has_virtual_connections = any('_weighted_' in k for k in self.active_connections.keys())
                    if not has_virtual_connections:
                        logger.info("Weighted algorithm active but no virtual connections found, recalculating")
                        self._calculate_connections()

            # Log the active connections for debugging (only at debug level)
            logger.debug(f"Active connections: {self.active_connections}")

            # If still no active connections after trying to get them from node_client,
            # force a recalculation
            if not self.active_connections:
                logger.warning("No active connections found after checking node_client, forcing recalculation")
                self._calculate_connections()

                # Log the new connections
                logger.info(f"Recalculated connections: {self.active_connections}")

                # Update node_client with the new connections
                node_client.camera_last_nodes = self.active_connections.copy()
        except Exception as e:
            logger.error(f"Error updating active connections: {str(e)}")

            # Fallback to using camera info from stats if node_client access fails
            for camera_id, camera_info in cameras.items():
                last_node = camera_info.get('last_node')
                if last_node:
                    self.active_connections[camera_id] = last_node

                    # Create packet animation if not exists
                    if camera_id not in self.packet_positions:
                        self.packet_positions[camera_id] = 0.0
                        self.packet_speeds[camera_id] = 0.05 + 0.05 * np.random.random()

            # If still no active connections, create default connections
            if not self.active_connections:
                logger.warning("No active connections found in fallback, creating default connections")

                # Get available cameras and nodes
                camera_ids = list(self.camera_positions.keys())
                node_ids = list(self.node_positions.keys())

                if camera_ids and node_ids:
                    # Create default connections - distribute cameras evenly across nodes
                    for i, camera_id in enumerate(camera_ids):
                        node_index = i % len(node_ids)
                        node_id = node_ids[node_index]
                        self.active_connections[camera_id] = node_id

                        # Create packet animation
                        if camera_id not in self.packet_positions:
                            self.packet_positions[camera_id] = 0.0
                            self.packet_speeds[camera_id] = 0.05 + 0.05 * np.random.random()

                    logger.info(f"Created default connections in fallback: {self.active_connections}")

        # Update packet animations
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time
        self.last_update_time = current_time

        # Get current algorithm for algorithm-specific animations
        current_algorithm = self.stats.get('routing_algorithm', 'direct')

        # Make a copy of the keys to avoid "dictionary changed size during iteration" error
        packet_position_keys = list(self.packet_positions.keys())
        active_connections_copy = self.active_connections.copy()

        for camera_id in packet_position_keys:
            if camera_id in active_connections_copy:
                # Get node ID for this connection
                node_id = active_connections_copy[camera_id]

                # Adjust speed based on algorithm
                speed_multiplier = 5.0  # Default speed multiplier

                if current_algorithm == 'direct':
                    # Direct: Consistent speed
                    speed_multiplier = 5.0
                elif current_algorithm == 'round_robin':
                    # Round Robin: Varying speeds in a pattern
                    speed_multiplier = 5.0 + 2.0 * math.sin(self.frame_count / 10.0)
                elif current_algorithm == 'least_connection':
                    # Least Connection: Faster to less loaded nodes
                    # Get node load
                    node_load = 0.5  # Default load
                    if node_id in network_simulator.nodes:
                        node_load = network_simulator.nodes[node_id].get('load', 0.5)
                    # Faster to less loaded nodes
                    speed_multiplier = 5.0 + 3.0 * (1.0 - node_load)
                elif current_algorithm == 'weighted':
                    # Weighted: Faster for higher capacity nodes
                    if 'node_1' in node_id or 'node_2' in node_id:
                        speed_multiplier = 7.0  # Faster for high capacity
                    elif 'node_3' in node_id or 'node_4' in node_id:
                        speed_multiplier = 5.5
                    else:
                        speed_multiplier = 4.0
                # IP Hash removed as it is not a routing algorithm

                # Move packet along the connection with algorithm-specific speed
                self.packet_positions[camera_id] += self.packet_speeds[camera_id] * elapsed_time * speed_multiplier

                # Reset when reaching the end
                if self.packet_positions[camera_id] >= 1.0:
                    # For some algorithms, add multiple packets in a burst
                    if current_algorithm == 'round_robin' and random.random() < 0.3:
                        # Round Robin: Sometimes send packets in bursts
                        self.packet_positions[camera_id] = 0.0
                        # Add 1-2 more packets at different positions (reduced from 1-3 to avoid too many)
                        for _ in range(random.randint(1, 2)):
                            # Create a new packet ID - use a more unique ID to avoid collisions
                            burst_id = random.randint(10000, 99999)
                            new_packet_id = f"{camera_id}_burst_{burst_id}"

                            # Only add if this packet ID doesn't already exist
                            if new_packet_id not in self.packet_positions:
                                # Add at a random position
                                self.packet_positions[new_packet_id] = random.uniform(0.1, 0.3)
                                self.packet_speeds[new_packet_id] = self.packet_speeds[camera_id] * random.uniform(0.8, 1.2)
                                # Set active connection for this packet
                                self.active_connections[new_packet_id] = node_id
                    else:
                        # Normal reset
                        self.packet_positions[camera_id] = 0.0
            else:
                # Remove inactive connections
                del self.packet_positions[camera_id]
                if camera_id in self.packet_speeds:
                    del self.packet_speeds[camera_id]

        # Add new packets based on algorithm
        if current_algorithm == 'direct':
            # Direct: Add packets at a steady rate to one node
            if random.random() < 0.2:  # 20% chance each update
                for camera_id in self.camera_positions.keys():
                    if camera_id not in self.packet_positions and camera_id in self.active_connections:
                        self.packet_positions[camera_id] = 0.0
                        self.packet_speeds[camera_id] = 0.05 + 0.02 * random.random()

        elif current_algorithm == 'round_robin':
            # Round Robin: Add packets in a rotating pattern
            if self.frame_count % 10 == 0:  # Every 10 frames
                # Select a camera based on frame count
                camera_ids = list(self.camera_positions.keys())
                if camera_ids:
                    camera_id = camera_ids[self.frame_count % len(camera_ids)]
                    if camera_id not in self.packet_positions and camera_id in self.active_connections:
                        self.packet_positions[camera_id] = 0.0
                        self.packet_speeds[camera_id] = 0.06 + 0.03 * random.random()

        elif current_algorithm == 'least_connection':
            # Least Connection: More packets to less loaded nodes
            for camera_id in self.camera_positions.keys():
                if camera_id not in self.packet_positions and camera_id in self.active_connections:
                    node_id = self.active_connections[camera_id]
                    # Get node load
                    node_load = 0.5  # Default load
                    if node_id in network_simulator.nodes:
                        node_load = network_simulator.nodes[node_id].get('load', 0.5)
                    # More likely to add packets to less loaded nodes
                    if random.random() < 0.15 * (1.0 - node_load):
                        self.packet_positions[camera_id] = 0.0
                        self.packet_speeds[camera_id] = 0.04 + 0.04 * random.random()

        elif current_algorithm == 'weighted':
            # Weighted: More packets to higher capacity nodes
            for camera_id in self.camera_positions.keys():
                if camera_id not in self.packet_positions and camera_id in self.active_connections:
                    node_id = self.active_connections[camera_id]
                    # Determine probability based on node capacity
                    prob = 0.1  # Default
                    if 'node_1' in node_id or 'node_2' in node_id:
                        prob = 0.2  # Higher for high capacity nodes
                    elif 'node_3' in node_id or 'node_4' in node_id:
                        prob = 0.15

                    if random.random() < prob:
                        self.packet_positions[camera_id] = 0.0
                        self.packet_speeds[camera_id] = 0.05 + 0.03 * random.random()

        # Default packet generation for any algorithm
        for camera_id in self.camera_positions.keys():
            if camera_id not in self.packet_positions and camera_id in self.active_connections:
                # Generate packets with a random probability
                if random.random() < 0.1:  # 10% chance to generate a packet
                    self.packet_positions[camera_id] = 0.0
                    self.packet_speeds[camera_id] = 0.05 + 0.03 * random.random()

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

                # Get current algorithm for algorithm-specific line styles
                current_algorithm = self.stats.get('routing_algorithm', 'direct')

                # Draw connection line with algorithm-specific styling
                if current_algorithm == 'direct':
                    # Direct: Simple solid line with higher visibility
                    cv2.line(image, camera_pos, node_pos, (100, 100, 200), 2, cv2.LINE_AA)

                elif current_algorithm == 'round_robin':
                    # Round Robin: Dashed line with rotating pattern
                    # Draw dashed line
                    dash_length = 10
                    gap_length = 5
                    # Calculate total distance
                    dx = node_pos[0] - camera_pos[0]
                    dy = node_pos[1] - camera_pos[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Calculate unit vector
                    if dist > 0:
                        dx, dy = dx/dist, dy/dist
                    else:
                        dx, dy = 0, 0
                    # Draw dashes
                    offset = (self.frame_count % (dash_length + gap_length))
                    pos = 0
                    while pos < dist:
                        start_pos = pos + offset
                        if start_pos > dist:
                            break
                        end_pos = min(start_pos + dash_length, dist)
                        if start_pos < end_pos:
                            start_x = int(camera_pos[0] + dx * start_pos)
                            start_y = int(camera_pos[1] + dy * start_pos)
                            end_x = int(camera_pos[0] + dx * end_pos)
                            end_y = int(camera_pos[1] + dy * end_pos)
                            cv2.line(image, (start_x, start_y), (end_x, end_y), (100, 100, 180), 1, cv2.LINE_AA)
                        pos = end_pos + gap_length

                elif current_algorithm == 'least_connection':
                    # Least Connection: Line thickness based on load
                    node_id = self.active_connections[camera_id]
                    # Get node load
                    node_load = 0.5  # Default load
                    if node_id in network_simulator.nodes:
                        node_load = network_simulator.nodes[node_id].get('load', 0.5)
                    # Thicker line for less loaded nodes
                    thickness = max(1, int(3 * (1.0 - node_load)))
                    # Color based on load (green for low load, yellow for medium, red for high)
                    if node_load < 0.3:
                        color = (50, 200, 50)  # Green
                    elif node_load < 0.7:
                        color = (50, 200, 200)  # Yellow
                    else:
                        color = (50, 50, 200)  # Red
                    cv2.line(image, camera_pos, node_pos, color, thickness, cv2.LINE_AA)

                elif current_algorithm == 'weighted':
                    # Weighted: Line thickness based on node capacity
                    node_id = self.active_connections[camera_id]
                    # Get actual node load and check if it's a simulated node
                    node_load = 0.0
                    is_simulated = False

                    try:
                        from vigilance_system.network.node_client import node_client
                        node_stats = node_client.get_stats()

                        # Check if this is a simulated node
                        if node_id in node_client.node_sockets:
                            is_simulated = node_client.node_sockets[node_id] is None

                        if 'node_stats' in node_stats and node_id in node_stats['node_stats']:
                            node_load = node_stats['node_stats'][node_id].get('load', 0.0)
                    except Exception as e:
                        logger.error(f"Error getting node load: {str(e)}")

                    # Determine capacity based on node ID
                    capacity = 1.0

                    # If it's a simulated node, use a distinct visual style
                    if is_simulated:
                        # Use a dashed pattern for simulated nodes
                        color = (80, 80, 80)  # Grey for simulated nodes
                    else:
                        # Real nodes with different capacities
                        if 'node_1' in node_id or 'node_2' in node_id:
                            capacity = 1.5
                            color = (100, 200, 100)  # Bright green for high capacity
                        elif 'node_3' in node_id or 'node_4' in node_id:
                            capacity = 1.2
                            # If node has no clients, use a different color
                            if node_load < 0.01:
                                color = (100, 100, 100)  # Grey for unused nodes
                            else:
                                color = (100, 200, 200)  # Teal for medium capacity
                        else:
                            capacity = 1.0
                            color = (200, 100, 100)  # Red for low capacity

                    # Thicker line for higher capacity
                    thickness = max(1, int(capacity * 1.5))
                    cv2.line(image, camera_pos, node_pos, color, thickness, cv2.LINE_AA)

                # IP Hash algorithm removed as it is not a routing algorithm

                else:
                    # Default visualization for any other algorithm
                    # Draw a simple line with a gradient effect
                    cv2.line(image, camera_pos, node_pos, (50, 150, 200), 2, cv2.LINE_AA)

                    # Add some visual interest with small circles along the line
                    num_circles = 3
                    for i in range(1, num_circles + 1):
                        t = i / (num_circles + 1)
                        circle_x = int(camera_pos[0] + (node_pos[0] - camera_pos[0]) * t)
                        circle_y = int(camera_pos[1] + (node_pos[1] - camera_pos[1]) * t)

                        # Draw circle with pulsing effect
                        circle_size = int(4 + 2 * math.sin(self.frame_count / 15.0 + i))
                        cv2.circle(image, (circle_x, circle_y), circle_size, (50, 150, 200), 1, cv2.LINE_AA)

                    # Add arrow to show direction
                    self._draw_arrow(image, camera_pos, node_pos, (50, 150, 200))

                # Draw packet animation with algorithm-specific styling
                if camera_id in self.packet_positions:
                    packet_pos = self.packet_positions[camera_id]
                    x = int(camera_pos[0] + (node_pos[0] - camera_pos[0]) * packet_pos)
                    y = int(camera_pos[1] + (node_pos[1] - camera_pos[1]) * packet_pos)

                    # Use different colors for different cameras
                    color_index = hash(camera_id) % len(self.packet_colors)
                    packet_color = self.packet_colors[color_index]

                    # Get current algorithm for algorithm-specific visualizations
                    current_algorithm = self.stats.get('routing_algorithm', 'direct')

                    # Draw packet with algorithm-specific styling
                    if current_algorithm == 'direct':
                        # Direct: Simple circles with tail
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

                    elif current_algorithm == 'round_robin':
                        # Round Robin: Rotating squares
                        angle = (self.frame_count + int(packet_pos * 100)) % 360
                        rect_size = 10
                        rect_points = np.array([
                            [x - rect_size/2, y - rect_size/2],
                            [x + rect_size/2, y - rect_size/2],
                            [x + rect_size/2, y + rect_size/2],
                            [x - rect_size/2, y + rect_size/2]
                        ], dtype=np.float32)

                        # Rotate points
                        rad = np.radians(angle)
                        cos_val = np.cos(rad)
                        sin_val = np.sin(rad)
                        center = np.array([x, y])
                        for i in range(4):
                            rect_points[i] = np.array([
                                cos_val * (rect_points[i][0] - center[0]) - sin_val * (rect_points[i][1] - center[1]) + center[0],
                                sin_val * (rect_points[i][0] - center[0]) + cos_val * (rect_points[i][1] - center[1]) + center[1]
                            ])

                        cv2.fillPoly(image, [rect_points.astype(np.int32)], packet_color)

                    elif current_algorithm == 'least_connection':
                        # Least Connection: Triangles pointing to least loaded node
                        triangle_size = 10
                        cv2.drawMarker(image, (x, y), packet_color, cv2.MARKER_TRIANGLE_UP, triangle_size, 2)

                        # Add a pulsing effect to show "seeking" the least loaded node
                        pulse_size = int(5 + 3 * math.sin(self.frame_count / 5.0 + hash(camera_id) % 10))
                        cv2.circle(image, (x, y), pulse_size, packet_color, 1, cv2.LINE_AA)

                    elif current_algorithm == 'weighted':
                        # Weighted: Different sized circles based on node weight
                        # Get node capacity/weight from node ID (simulated)
                        weight = 1.0
                        if 'node_1' in node_id or 'node_2' in node_id:
                            weight = 1.5
                        elif 'node_3' in node_id or 'node_4' in node_id:
                            weight = 1.2

                        # Size based on weight
                        size = int(5 * weight)
                        cv2.circle(image, (x, y), size, packet_color, -1, cv2.LINE_AA)

                        # Draw weight indicator
                        weight_text = f"{weight:.1f}x"
                        text_size = cv2.getTextSize(weight_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        text_x = x - text_size[0] // 2
                        text_y = y - 10
                        safe_putText(image, weight_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                    else:
                        # Handle any other algorithm with default visualization
                        # This ensures backward compatibility with any algorithm
                        cv2.circle(image, (x, y), 6, packet_color, -1, cv2.LINE_AA)

                        # Draw a tail behind the packet for visibility
                        tail_length = 3
                        for i in range(1, tail_length + 1):
                            tail_pos = packet_pos - (i * 0.05)
                            if tail_pos >= 0:
                                tail_x = int(camera_pos[0] + (node_pos[0] - camera_pos[0]) * tail_pos)
                                tail_y = int(camera_pos[1] + (node_pos[1] - camera_pos[1]) * tail_pos)
                                tail_alpha = 1.0 - (i / (tail_length + 1))
                                tail_color = tuple([int(c * tail_alpha) for c in packet_color])
                                cv2.circle(image, (tail_x, tail_y), 6 - i, tail_color, -1, cv2.LINE_AA)

                        # Draw the main packet
                        cv2.circle(image, (x, y), 8, packet_color, -1, cv2.LINE_AA)

                        # Add a white center for better visibility
                        cv2.circle(image, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)

                        # Draw a longer, more visible tail behind the packet
                        tail_length = 5
                        for i in range(1, tail_length + 1):
                            tail_pos = packet_pos - (i * 0.05)
                            if tail_pos >= 0:
                                tail_x = int(camera_pos[0] + (node_pos[0] - camera_pos[0]) * tail_pos)
                                tail_y = int(camera_pos[1] + (node_pos[1] - camera_pos[1]) * tail_pos)
                                tail_alpha = 1.0 - (i / (tail_length + 1))
                                tail_color = tuple([int(c * tail_alpha) for c in packet_color])
                                cv2.circle(image, (tail_x, tail_y), 8 - i, tail_color, -1, cv2.LINE_AA)

        # Draw nodes
        for node_id, pos in self.node_positions.items():
            # Get node load from network simulator if available
            node_load = 0.0

            # Get node info from network simulator
            node_info = network_simulator.nodes.get(node_id, {})

            # Calculate load and client connections
            connection_count = 0
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
            is_simulated = False
            try:
                from vigilance_system.network.node_client import node_client
                # A node is simulated if it's in node_sockets but the socket is None
                if node_id in node_client.node_sockets:
                    is_simulated = node_client.node_sockets[node_id] is None
            except Exception as e:
                logger.error(f"Error checking if node is simulated: {str(e)}")
                # Fallback to checking node ID
                is_simulated = "simulated" in node_id.lower() or node_id not in self.stats.get('client_stats', {}).get('real_node_ids', [])

            # Get actual client count from node_client
            client_count = 0
            try:
                from vigilance_system.network.node_client import node_client
                # Check if this node is in the active connections
                client_count = list(node_client.camera_last_nodes.values()).count(node_id)
            except Exception as e:
                logger.error(f"Error getting client count: {str(e)}")
                # Fallback to connection_count
                client_count = connection_count

            # Determine node status color based on client count
            if client_count > 0:
                status_color = (0, 255, 0)  # Green for active connections
                status_text = f"CLIENTS: {client_count}"
            else:
                status_color = (100, 100, 100)  # Gray for no connections
                status_text = "CLIENTS: 0"

            # Draw terminal-like rectangle for node with different color for simulated nodes
            # Use more vibrant colors for better visibility
            bg_color = (30, 30, 60) if is_simulated else (40, 40, 40)

            # Use a more visible border color based on connection status
            if connection_count > 0:
                # Active node with connections - use a bright green border
                border_color = (80, 200, 80) if not is_simulated else (80, 160, 80)
            else:
                # Inactive node - use a gray/blue border
                border_color = (100, 100, 180) if is_simulated else (120, 120, 120)

            # Make the node terminal larger to fit more information
            # Add a glow effect for active nodes
            if connection_count > 0:
                # Draw outer glow for active nodes
                for i in range(5, 0, -1):
                    alpha = 0.2 - (i * 0.03)
                    glow_color = tuple([int(c * (1 + alpha)) for c in border_color])
                    cv2.rectangle(image,
                                 (pos[0] - 40 - i, pos[1] - 25 - i),
                                 (pos[0] + 40 + i, pos[1] + 25 + i),
                                 glow_color, 1, cv2.LINE_AA)

            # Draw main node rectangle
            cv2.rectangle(image, (pos[0] - 40, pos[1] - 25), (pos[0] + 40, pos[1] + 25), bg_color, -1, cv2.LINE_AA)
            cv2.rectangle(image, (pos[0] - 40, pos[1] - 25), (pos[0] + 40, pos[1] + 25), border_color, 2, cv2.LINE_AA)

            # Draw terminal title bar with more vibrant colors
            title_color = (70, 70, 200) if is_simulated else (70, 70, 170)
            cv2.rectangle(image, (pos[0] - 40, pos[1] - 25), (pos[0] + 40, pos[1] - 15), title_color, -1, cv2.LINE_AA)

            # Draw node label in title bar
            label = f"{node_id}" + (" (SIM)" if is_simulated else "")
            safe_putText(image, label, (pos[0] - 35, pos[1] - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw client connection status with appropriate color
            safe_putText(image, status_text, (pos[0] - 35, pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1, cv2.LINE_AA)

            # Draw terminal content with real IP if available
            if 'client_ip' in self.stats:
                ip_address = self.stats.get('client_ip', '127.0.0.1')
            else:
                ip_address = "127.0.0.1"

            safe_putText(image, ip_address, (pos[0] - 35, pos[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            safe_putText(image, f"Load: {load:.1%}", (pos[0] - 35, pos[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # Draw load indicator bar below terminal
            cv2.rectangle(image, (pos[0] - 40, pos[1] + 30), (pos[0] + 40, pos[1] + 35), (50, 50, 50), -1, cv2.LINE_AA)

            # Color the load bar based on load level
            if load < 0.3:
                load_color = (50, 200, 50)  # Green for low load
            elif load < 0.7:
                load_color = (50, 200, 200)  # Yellow for medium load
            else:
                load_color = (50, 50, 200)  # Red for high load

            cv2.rectangle(image, (pos[0] - 40, pos[1] + 30), (pos[0] - 40 + int(80 * load), pos[1] + 35), load_color, -1, cv2.LINE_AA)

            # Draw node label below load bar
            label = f"{node_id} ({load:.0%})"

            # Add SIM label for simulated nodes
            if is_simulated:
                label += " (SIM)"

            # Add client count to label
            try:
                from vigilance_system.network.node_client import node_client
                node_stats = node_client.get_stats()
                if 'node_stats' in node_stats and node_id in node_stats['node_stats']:
                    client_count = node_stats['node_stats'][node_id].get('clients', 0)
                    label += f" CLIENTS: {client_count}"
            except Exception as e:
                logger.error(f"Error getting client count: {str(e)}")

            safe_putText(image, label, (pos[0] - 35, pos[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1, cv2.LINE_AA)

        # Draw cameras
        for camera_id, pos in self.camera_positions.items():
            # Check if this camera has active connections
            is_active = camera_id in self.active_connections

            # Use brighter colors for active cameras
            camera_color = (100, 100, 220) if is_active else self.camera_color
            border_color = (50, 50, 180) if is_active else (30, 30, 100)

            # Draw camera icon with improved visibility
            # Add glow effect for active cameras
            if is_active:
                # Draw outer glow
                for i in range(4, 0, -1):
                    alpha = 0.3 - (i * 0.05)
                    glow_color = tuple([int(c * (1 + alpha)) for c in camera_color])
                    cv2.rectangle(image,
                                 (pos[0] - 15 - i, pos[1] - 10 - i),
                                 (pos[0] + 15 + i, pos[1] + 10 + i),
                                 glow_color, 1, cv2.LINE_AA)

            # Draw main camera body
            cv2.rectangle(image, (pos[0] - 15, pos[1] - 10), (pos[0] + 15, pos[1] + 10), camera_color, -1, cv2.LINE_AA)
            cv2.rectangle(image, (pos[0] - 15, pos[1] - 10), (pos[0] + 15, pos[1] + 10), border_color, 2, cv2.LINE_AA)

            # Draw camera lens
            cv2.circle(image, (pos[0] + 20, pos[1]), 5, camera_color, -1, cv2.LINE_AA)
            cv2.circle(image, (pos[0] + 20, pos[1]), 5, border_color, 1, cv2.LINE_AA)

            # Add a small indicator light
            indicator_color = (50, 220, 50) if is_active else (220, 50, 50)  # Green if active, red if inactive
            cv2.circle(image, (pos[0] - 10, pos[1] - 5), 3, indicator_color, -1, cv2.LINE_AA)

            # Draw camera label with better visibility
            # Add a small background for the text
            text_size = cv2.getTextSize(camera_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_bg_x1 = pos[0] - text_size[0] // 2 - 2
            text_bg_y1 = pos[1] + 20
            text_bg_x2 = pos[0] + text_size[0] // 2 + 2
            text_bg_y2 = pos[1] + 20 + text_size[1] + 2

            # Draw text background
            cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 0, 0, 128), -1, cv2.LINE_AA)

            # Draw camera label
            safe_putText(image, camera_id, (pos[0] - text_size[0] // 2, pos[1] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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

        # Adjust radius based on number of nodes
        if num_nodes <= 5:
            radius = min(self.width, self.height) // 3
        else:
            # For more nodes, use a larger radius to avoid overcrowding
            radius = min(self.width, self.height) // 2.5

        # Sort node IDs to ensure consistent positioning
        sorted_node_ids = sorted(node_ids)

        for i, node_id in enumerate(sorted_node_ids):
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

    def _calculate_connections(self) -> None:
        """
        Calculate connections between cameras and nodes based on the current routing algorithm.
        This is used to force a recalculation of connections when the algorithm changes.
        """
        try:
            # Get the current algorithm from file first - this is the source of truth
            algorithm_file = os.path.join(os.getcwd(), 'current_algorithm.txt')
            file_algorithm = None
            if os.path.exists(algorithm_file):
                try:
                    with open(algorithm_file, 'r') as f:
                        file_algorithm = f.read().strip()
                        # Validate the algorithm
                        valid_algorithms = ['direct', 'round_robin', 'least_connection', 'weighted']
                        if file_algorithm not in valid_algorithms:
                            logger.warning(f"Invalid algorithm in file: {file_algorithm}, using direct instead")
                            file_algorithm = "direct"
                except Exception as file_error:
                    logger.error(f"Error reading algorithm file: {str(file_error)}")

            # Use file algorithm if available, otherwise fall back to stats
            current_algorithm = file_algorithm if file_algorithm else self.stats.get('routing_algorithm', 'direct')

            # Update stats to match the file
            self.stats['routing_algorithm'] = current_algorithm

            logger.info(f"Calculating connections for algorithm: {current_algorithm}")

            # Get available cameras and nodes
            camera_ids = list(self.camera_positions.keys())
            node_ids = list(self.node_positions.keys())

            if not camera_ids or not node_ids:
                logger.warning("No cameras or nodes available for connection calculation")
                return

            # Clear existing connections
            self.active_connections = {}

            # Calculate new connections based on algorithm
            if current_algorithm == 'direct':
                # Direct: All cameras connect to the first node
                first_node = node_ids[0]

                # Store existing animation state to preserve it
                existing_virtual_positions = {k: v for k, v in self.packet_positions.items() if '_direct_' in k}
                existing_virtual_speeds = {k: v for k, v in self.packet_speeds.items() if '_direct_' in k}

                # Create virtual connections for visualization
                # For Direct, we want to show all cameras connecting to a single node
                for camera_id in camera_ids:
                    # Create a virtual connection for each camera to the first node
                    virtual_camera_id = f"{camera_id}_direct_0"
                    self.active_connections[virtual_camera_id] = first_node

                    # Add camera position for virtual camera
                    if camera_id in self.camera_positions:
                        self.camera_positions[virtual_camera_id] = self.camera_positions[camera_id]

                    # Initialize packet positions for new connections
                    # Preserve existing animation state if available
                    if virtual_camera_id in existing_virtual_positions:
                        self.packet_positions[virtual_camera_id] = existing_virtual_positions[virtual_camera_id]
                    elif virtual_camera_id not in self.packet_positions:
                        # Start new packets at different positions along the path
                        self.packet_positions[virtual_camera_id] = random.random() * 0.5  # Random position in first half

                    # Preserve existing speeds if available
                    if virtual_camera_id in existing_virtual_speeds:
                        self.packet_speeds[virtual_camera_id] = existing_virtual_speeds[virtual_camera_id]
                    elif virtual_camera_id not in self.packet_speeds:
                        # Vary speeds slightly to avoid synchronized animation
                        self.packet_speeds[virtual_camera_id] = 0.05 + 0.03 * np.random.random()

                # Also keep the original camera connections for compatibility
                for camera_id in camera_ids:
                    self.active_connections[camera_id] = first_node

                    # Make sure we have packet positions for real cameras too
                    if camera_id not in self.packet_positions:
                        self.packet_positions[camera_id] = 0.0
                        self.packet_speeds[camera_id] = 0.05 + 0.03 * np.random.random()

                # Log the number of virtual connections created
                virtual_count = sum(1 for k in self.active_connections if '_direct_' in k)
                logger.info(f"Created {virtual_count} virtual connections for Direct visualization")

            elif current_algorithm == 'round_robin':
                # Round Robin: Distribute cameras evenly across ALL nodes
                # Create multiple connections for each camera to show distribution

                # Store existing animation state to preserve it
                existing_virtual_positions = {k: v for k, v in self.packet_positions.items() if '_rr_' in k}
                existing_virtual_speeds = {k: v for k, v in self.packet_speeds.items() if '_rr_' in k}

                # Create virtual connections for visualization
                for camera_id in camera_ids:
                    # For visualization purposes, create multiple virtual camera IDs
                    # to show connections to all nodes in round robin fashion
                    for i, node_id in enumerate(sorted(node_ids)):
                        virtual_camera_id = f"{camera_id}_rr_{i}"
                        self.active_connections[virtual_camera_id] = node_id

                        # Add camera position for virtual camera
                        if camera_id in self.camera_positions:
                            self.camera_positions[virtual_camera_id] = self.camera_positions[camera_id]

                        # Initialize packet positions for new connections
                        # Preserve existing animation state if available
                        if virtual_camera_id in existing_virtual_positions:
                            self.packet_positions[virtual_camera_id] = existing_virtual_positions[virtual_camera_id]
                        elif virtual_camera_id not in self.packet_positions:
                            # Start new packets at different positions along the path
                            self.packet_positions[virtual_camera_id] = random.random() * 0.5  # Random position in first half

                        # Preserve existing speeds if available
                        if virtual_camera_id in existing_virtual_speeds:
                            self.packet_speeds[virtual_camera_id] = existing_virtual_speeds[virtual_camera_id]
                        elif virtual_camera_id not in self.packet_speeds:
                            # Vary speeds slightly to avoid synchronized animation
                            self.packet_speeds[virtual_camera_id] = 0.05 + 0.03 * np.random.random()

                # Also keep the original camera connections for compatibility
                for i, camera_id in enumerate(camera_ids):
                    node_index = i % len(node_ids)
                    self.active_connections[camera_id] = node_ids[node_index]

                    # Make sure we have packet positions for real cameras too
                    if camera_id not in self.packet_positions:
                        self.packet_positions[camera_id] = 0.0
                        self.packet_speeds[camera_id] = 0.05 + 0.03 * np.random.random()

                # Log the number of virtual connections created
                virtual_count = sum(1 for k in self.active_connections if '_rr_' in k)
                logger.info(f"Created {virtual_count} virtual connections for Round Robin visualization")

            elif current_algorithm == 'least_connection':
                # Least Connection: Distribute based on node load
                # Get node loads or use default distribution
                node_loads = {}
                for node_id in node_ids:
                    if node_id in network_simulator.nodes:
                        node_loads[node_id] = network_simulator.nodes[node_id].get('load', 0.5)
                    else:
                        node_loads[node_id] = 0.5

                # Sort nodes by load (least loaded first)
                sorted_nodes = sorted(node_loads.items(), key=lambda x: x[1])

                # Store existing animation state to preserve it
                existing_virtual_positions = {k: v for k, v in self.packet_positions.items() if '_lc_' in k}
                existing_virtual_speeds = {k: v for k, v in self.packet_speeds.items() if '_lc_' in k}

                # Create virtual connections for visualization
                # For Least Connection, we want to show more traffic going to less loaded nodes
                for camera_id in camera_ids:
                    # Create virtual connections to all nodes with varying packet rates based on load
                    for i, (node_id, load) in enumerate(sorted_nodes):
                        virtual_camera_id = f"{camera_id}_lc_{i}"
                        self.active_connections[virtual_camera_id] = node_id

                        # Add camera position for virtual camera
                        if camera_id in self.camera_positions:
                            self.camera_positions[virtual_camera_id] = self.camera_positions[camera_id]

                        # Initialize packet positions for new connections
                        # Preserve existing animation state if available
                        if virtual_camera_id in existing_virtual_positions:
                            self.packet_positions[virtual_camera_id] = existing_virtual_positions[virtual_camera_id]
                        elif virtual_camera_id not in self.packet_positions:
                            # Start new packets at different positions along the path
                            self.packet_positions[virtual_camera_id] = random.random() * 0.5  # Random position in first half

                        # Preserve existing speeds if available
                        if virtual_camera_id in existing_virtual_speeds:
                            self.packet_speeds[virtual_camera_id] = existing_virtual_speeds[virtual_camera_id]
                        elif virtual_camera_id not in self.packet_speeds:
                            # Vary speeds based on load - faster for less loaded nodes
                            # This creates a visual effect where less loaded nodes get packets faster
                            speed_factor = 1.0 - load  # Invert load to make less loaded nodes faster
                            self.packet_speeds[virtual_camera_id] = 0.05 + 0.05 * speed_factor + 0.02 * np.random.random()

                # Also keep the original camera connections for compatibility
                # Distribute cameras to least loaded nodes
                for i, camera_id in enumerate(camera_ids):
                    # Use modulo to cycle through sorted nodes
                    node_index = i % len(sorted_nodes)
                    self.active_connections[camera_id] = sorted_nodes[node_index][0]

                # Log the number of virtual connections created
                virtual_count = sum(1 for k in self.active_connections if '_lc_' in k)
                logger.info(f"Created {virtual_count} virtual connections for Least Connection visualization")

            elif current_algorithm == 'weighted':
                # Weighted: Distribute based on node capacity
                # Higher capacity nodes get more cameras
                capacities = {}
                for node_id in node_ids:
                    if 'node_1' in node_id or 'node_2' in node_id:
                        capacities[node_id] = 3  # High capacity
                    elif 'node_3' in node_id or 'node_4' in node_id:
                        capacities[node_id] = 2  # Medium capacity
                    else:
                        capacities[node_id] = 1  # Standard capacity

                # Store existing animation state to preserve it
                existing_virtual_positions = {k: v for k, v in self.packet_positions.items() if '_weighted_' in k}
                existing_virtual_speeds = {k: v for k, v in self.packet_speeds.items() if '_weighted_' in k}

                # Create virtual connections for visualization
                # For Weighted, we want to show more traffic going to higher capacity nodes
                for camera_id in camera_ids:
                    # Create virtual connections to all nodes with varying packet rates based on capacity
                    for i, node_id in enumerate(sorted(node_ids)):
                        # Create multiple virtual connections based on capacity
                        # Higher capacity nodes get more virtual connections
                        node_capacity = capacities.get(node_id, 1)

                        # Create multiple virtual connections for each node based on its capacity
                        for j in range(node_capacity):
                            virtual_camera_id = f"{camera_id}_weighted_{i}_{j}"
                            self.active_connections[virtual_camera_id] = node_id

                            # Add camera position for virtual camera
                            if camera_id in self.camera_positions:
                                self.camera_positions[virtual_camera_id] = self.camera_positions[camera_id]

                            # Initialize packet positions for new connections
                            # Preserve existing animation state if available
                            if virtual_camera_id in existing_virtual_positions:
                                self.packet_positions[virtual_camera_id] = existing_virtual_positions[virtual_camera_id]
                            elif virtual_camera_id not in self.packet_positions:
                                # Start new packets at different positions along the path
                                self.packet_positions[virtual_camera_id] = random.random() * 0.5  # Random position in first half

                            # Preserve existing speeds if available
                            if virtual_camera_id in existing_virtual_speeds:
                                self.packet_speeds[virtual_camera_id] = existing_virtual_speeds[virtual_camera_id]
                            elif virtual_camera_id not in self.packet_speeds:
                                # Vary speeds based on capacity - faster for higher capacity nodes
                                # This creates a visual effect where higher capacity nodes process packets faster
                                speed_factor = node_capacity / 3.0  # Normalize to max capacity
                                self.packet_speeds[virtual_camera_id] = 0.05 + 0.05 * speed_factor + 0.02 * np.random.random()

                # Create a list with repeated node IDs based on capacity for real connections
                weighted_nodes = []
                for node_id, capacity in capacities.items():
                    weighted_nodes.extend([node_id] * capacity)

                # Distribute cameras across weighted nodes for real connections
                for i, camera_id in enumerate(camera_ids):
                    node_index = i % len(weighted_nodes)
                    self.active_connections[camera_id] = weighted_nodes[node_index]

                # Log the number of virtual connections created
                virtual_count = sum(1 for k in self.active_connections if '_weighted_' in k)
                logger.info(f"Created {virtual_count} virtual connections for Weighted visualization")

            else:
                # Default: Distribute evenly
                for i, camera_id in enumerate(camera_ids):
                    node_index = i % len(node_ids)
                    self.active_connections[camera_id] = node_ids[node_index]

            # Initialize packet positions for new connections
            for camera_id in self.active_connections:
                if camera_id not in self.packet_positions:
                    self.packet_positions[camera_id] = 0.0
                    self.packet_speeds[camera_id] = 0.05 + 0.05 * np.random.random()

            # Update node_client's camera_last_nodes to reflect these connections
            # Only update with real camera IDs, not virtual ones for visualization
            from vigilance_system.network.node_client import node_client
            real_connections = {cam_id: node_id for cam_id, node_id in self.active_connections.items()
                               if "_rr_" not in cam_id}
            node_client.camera_last_nodes = real_connections.copy()

            logger.info(f"Recalculated connections for algorithm {current_algorithm}: {len(self.active_connections)} connections")

        except Exception as e:
            logger.error(f"Error calculating connections: {str(e)}")

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
            'WEIGHTED': 0.6
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
            'direct': "Direct Connection: All traffic goes to a single node.\nSimple but not scalable. Good for small deployments with one powerful server.\nVisualization shows all cameras sending packets to the first node.",

            'round_robin': "Round Robin: Distributes traffic evenly across ALL nodes in sequence.\nEach packet is sent to the next node in the rotation, creating connections to every node.\nBalances load but doesn't consider node capacity or current load.\nVisualization shows cameras connecting to all nodes with equal packet distribution.",

            'least_connection': "Least Connection: Sends traffic to the node with the fewest active connections.\nAdaptive to changing loads. Good for mixed workloads.\nVisualization shows more packets flowing to less loaded nodes (green bars).",

            'weighted': "Weighted Distribution: Routes based on node capacity weights.\nSends more traffic to higher capacity nodes. Good for heterogeneous clusters.\nVisualization shows higher capacity nodes (green) receiving more packets than lower capacity nodes (red)."
            # IP Hash and YOLOv8 removed as they are not routing algorithms
        }

        return descriptions.get(algorithm, "Unknown algorithm")

    def _draw_arrow(self, image, start_point, end_point, color, arrow_size=15):
        """
        Draw an arrow from start_point to end_point.

        Args:
            image: Image to draw on
            start_point: Starting point (x, y)
            end_point: Ending point (x, y)
            color: Arrow color (B, G, R)
            arrow_size: Size of the arrow head
        """
        # Calculate direction vector
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]

        # Normalize the direction vector
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return

        dx = dx / length
        dy = dy / length

        # Calculate arrow head points
        # Position the arrow head at 70% of the way from start to end
        arrow_pos = (
            int(start_point[0] + dx * length * 0.7),
            int(start_point[1] + dy * length * 0.7)
        )

        # Calculate perpendicular vector
        perpx = -dy
        perpy = dx

        # Calculate arrow head points
        p1 = (
            int(arrow_pos[0] - dx * arrow_size + perpx * arrow_size * 0.5),
            int(arrow_pos[1] - dy * arrow_size + perpy * arrow_size * 0.5)
        )
        p2 = (
            int(arrow_pos[0] - dx * arrow_size - perpx * arrow_size * 0.5),
            int(arrow_pos[1] - dy * arrow_size - perpy * arrow_size * 0.5)
        )

        # Draw arrow head
        cv2.fillPoly(image, [np.array([arrow_pos, p1, p2])], color)


# Create a default instance
network_visualizer = NetworkVisualizer()
