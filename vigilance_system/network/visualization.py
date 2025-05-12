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
        # Store previous algorithm to detect changes
        previous_algorithm = self.stats.get('routing_algorithm', 'unknown')

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

            # If algorithm changed, reset active connections to force new routing pattern
            if previous_algorithm != current_algorithm:
                logger.info(f"Algorithm changed from {previous_algorithm} to {current_algorithm}, resetting connections")
                self.active_connections = {}
                self.packet_positions = {}
                self.packet_speeds = {}
                # Force node client to reset connections
                node_client.camera_last_nodes = {}
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

        # Get current algorithm for algorithm-specific animations
        current_algorithm = self.stats.get('routing_algorithm', 'direct')

        for camera_id in list(self.packet_positions.keys()):
            if camera_id in self.active_connections:
                # Get node ID for this connection
                node_id = self.active_connections[camera_id]

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
                elif current_algorithm == 'ip_hash':
                    # IP Hash: Consistent speed per camera
                    hash_val = hash(camera_id) % 100
                    speed_multiplier = 4.0 + (hash_val / 100.0) * 3.0  # Range from 4.0 to 7.0

                # Move packet along the connection with algorithm-specific speed
                self.packet_positions[camera_id] += self.packet_speeds[camera_id] * elapsed_time * speed_multiplier

                # Reset when reaching the end
                if self.packet_positions[camera_id] >= 1.0:
                    # For some algorithms, add multiple packets in a burst
                    if current_algorithm == 'round_robin' and random.random() < 0.3:
                        # Round Robin: Sometimes send packets in bursts
                        self.packet_positions[camera_id] = 0.0
                        # Add 1-3 more packets at different positions
                        for _ in range(random.randint(1, 3)):
                            # Create a new packet ID
                            new_packet_id = f"{camera_id}_burst_{random.randint(1000, 9999)}"
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

        elif current_algorithm == 'ip_hash':
            # IP Hash: Consistent packet generation per camera
            for camera_id in self.camera_positions.keys():
                if camera_id not in self.packet_positions and camera_id in self.active_connections:
                    # Use hash of camera ID for consistent behavior
                    hash_val = hash(camera_id) % 100
                    prob = 0.05 + (hash_val / 500.0)  # Range from 0.05 to 0.25

                    if random.random() < prob:
                        self.packet_positions[camera_id] = 0.0
                        # Consistent speed based on camera ID
                        base_speed = 0.04 + (hash_val / 1000.0)  # Range from 0.04 to 0.14
                        self.packet_speeds[camera_id] = base_speed + 0.01 * random.random()

        elif current_algorithm == 'yolov8':
            # YOLOv8: Smart packet generation based on node scores
            for camera_id in self.camera_positions.keys():
                if camera_id not in self.packet_positions and camera_id in self.active_connections:
                    node_id = self.active_connections[camera_id]

                    # Get node load
                    node_load = 0.5  # Default load
                    if node_id in network_simulator.nodes:
                        node_load = network_simulator.nodes[node_id].get('load', 0.5)

                    # Determine capacity based on node ID
                    capacity = 1.0
                    if 'node_1' in node_id or 'node_2' in node_id:
                        capacity = 1.5
                    elif 'node_3' in node_id or 'node_4' in node_id:
                        capacity = 1.2

                    # Calculate score (higher is better)
                    if node_load < 0.1:  # Very low load
                        score = capacity * 10  # Heavily favor capacity for unused nodes
                    else:
                        score = capacity / (node_load + 0.1)  # Balance capacity and load

                    # Higher probability for higher scores
                    prob = min(0.3, 0.05 + (score / 50.0))

                    # Batch processing - sometimes send multiple packets at once
                    if random.random() < prob:
                        # Add main packet
                        self.packet_positions[camera_id] = 0.0
                        self.packet_speeds[camera_id] = 0.05 + 0.03 * random.random()

                        # Sometimes add batch of additional packets (simulating YOLO batch processing)
                        if random.random() < 0.2:  # 20% chance of batch
                            batch_size = random.randint(2, 4)
                            for i in range(batch_size):
                                # Create unique ID for additional packets
                                batch_id = f"{camera_id}_batch_{self.frame_count}_{i}"
                                # Stagger the starting positions slightly
                                self.packet_positions[batch_id] = 0.02 * i
                                self.packet_speeds[batch_id] = self.packet_speeds[camera_id] * random.uniform(0.9, 1.1)
                                # Set active connection for this packet
                                self.active_connections[batch_id] = node_id

        else:
            # Default behavior for other algorithms
            for camera_id in self.camera_positions.keys():
                if camera_id not in self.packet_positions and camera_id in self.active_connections:
                    if random.random() < 0.1:  # 10% chance each update
                        self.packet_positions[camera_id] = 0.0
                        self.packet_speeds[camera_id] = 0.05 + 0.05 * random.random()

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
                    # Direct: Simple solid line
                    cv2.line(image, camera_pos, node_pos, self.line_color, 1, cv2.LINE_AA)

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
                    # Determine capacity based on node ID
                    capacity = 1.0
                    if 'node_1' in node_id or 'node_2' in node_id:
                        capacity = 1.5
                        color = (100, 200, 100)  # Bright green for high capacity
                    elif 'node_3' in node_id or 'node_4' in node_id:
                        capacity = 1.2
                        color = (100, 200, 200)  # Teal for medium capacity
                    else:
                        capacity = 1.0
                        color = (200, 100, 100)  # Red for low capacity

                    # Thicker line for higher capacity
                    thickness = max(1, int(capacity * 1.5))
                    cv2.line(image, camera_pos, node_pos, color, thickness, cv2.LINE_AA)

                elif current_algorithm == 'ip_hash':
                    # IP Hash: Consistent color per camera
                    # Use hash of camera ID for consistent color
                    hash_val = hash(camera_id) % 360
                    # Convert to HSV and then to BGR
                    hue = hash_val / 360.0
                    line_color = tuple([int(c) for c in colorsys.hsv_to_rgb(hue, 0.6, 0.7) * np.array([255, 255, 255])])

                    # Draw line with consistent color
                    cv2.line(image, camera_pos, node_pos, line_color, 2, cv2.LINE_AA)

                    # Draw small hash markers along the line
                    num_markers = 3
                    for i in range(1, num_markers + 1):
                        t = i / (num_markers + 1)
                        marker_x = int(camera_pos[0] + (node_pos[0] - camera_pos[0]) * t)
                        marker_y = int(camera_pos[1] + (node_pos[1] - camera_pos[1]) * t)
                        cv2.drawMarker(image, (marker_x, marker_y), line_color, cv2.MARKER_DIAMOND, 8, 1)

                elif current_algorithm == 'yolov8':
                    # YOLOv8: Advanced visualization with AI-inspired elements
                    node_id = self.active_connections[camera_id]

                    # Get node load and capacity
                    node_load = 0.5  # Default load
                    if node_id in network_simulator.nodes:
                        node_load = network_simulator.nodes[node_id].get('load', 0.5)

                    # Determine capacity based on node ID
                    capacity = 1.0
                    if 'node_1' in node_id or 'node_2' in node_id:
                        capacity = 1.5
                    elif 'node_3' in node_id or 'node_4' in node_id:
                        capacity = 1.2

                    # Calculate score (higher is better)
                    if node_load < 0.1:  # Very low load
                        score = capacity * 10  # Heavily favor capacity for unused nodes
                    else:
                        score = capacity / (node_load + 0.1)  # Balance capacity and load

                    # Color based on score (green for high score, yellow for medium, red for low)
                    if score > 10:
                        color = (50, 220, 50)  # Bright green for high score
                    elif score > 5:
                        color = (50, 220, 220)  # Yellow for medium score
                    else:
                        color = (50, 50, 220)  # Red for low score

                    # Draw a neural network inspired line
                    # Main line
                    cv2.line(image, camera_pos, node_pos, color, 2, cv2.LINE_AA)

                    # Draw neural network nodes along the line
                    num_nodes = 4
                    for i in range(1, num_nodes + 1):
                        t = i / (num_nodes + 1)
                        node_x = int(camera_pos[0] + (node_pos[0] - camera_pos[0]) * t)
                        node_y = int(camera_pos[1] + (node_pos[1] - camera_pos[1]) * t)

                        # Draw neural node with pulsing effect
                        node_size = int(6 + 3 * math.sin(self.frame_count / 10.0 + i))
                        cv2.circle(image, (node_x, node_y), node_size, color, -1, cv2.LINE_AA)
                        cv2.circle(image, (node_x, node_y), node_size + 2, color, 1, cv2.LINE_AA)

                        # Add small connecting lines to simulate neural network
                        if i < num_nodes:
                            next_t = (i + 1) / (num_nodes + 1)
                            next_x = int(camera_pos[0] + (node_pos[0] - camera_pos[0]) * next_t)
                            next_y = int(camera_pos[1] + (node_pos[1] - camera_pos[1]) * next_t)

                            # Draw small branch lines
                            branch_len = 10
                            angle = math.atan2(next_y - node_y, next_x - node_x) + math.pi/2
                            branch1_x = int(node_x + branch_len * math.cos(angle))
                            branch1_y = int(node_y + branch_len * math.sin(angle))
                            branch2_x = int(node_x - branch_len * math.cos(angle))
                            branch2_y = int(node_y - branch_len * math.sin(angle))

                            # Draw branches with fading effect
                            alpha = 0.7 - (i / num_nodes * 0.3)
                            branch_color = tuple([int(c * alpha) for c in color])
                            cv2.line(image, (node_x, node_y), (branch1_x, branch1_y), branch_color, 1, cv2.LINE_AA)
                            cv2.line(image, (node_x, node_y), (branch2_x, branch2_y), branch_color, 1, cv2.LINE_AA)

                else:
                    # Default: Simple line
                    cv2.line(image, camera_pos, node_pos, self.line_color, 1, cv2.LINE_AA)

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

                    elif current_algorithm == 'ip_hash':
                        # IP Hash: Diamond shapes with consistent color per camera
                        # Use hash of camera ID for consistent color
                        hash_val = hash(camera_id) % 360
                        # Convert to HSV and then to BGR
                        hue = hash_val / 360.0
                        consistent_color = tuple([int(c) for c in colorsys.hsv_to_rgb(hue, 0.8, 1.0) * np.array([255, 255, 255])])

                        # Draw diamond
                        diamond_size = 10
                        cv2.drawMarker(image, (x, y), consistent_color, cv2.MARKER_DIAMOND, diamond_size, 2)

                        # Draw hash indicator
                        hash_text = f"#{hash_val % 100:02d}"
                        text_size = cv2.getTextSize(hash_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        text_x = x - text_size[0] // 2
                        text_y = y - 10
                        safe_putText(image, hash_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                    elif current_algorithm == 'yolov8':
                        # YOLOv8: AI-inspired visualization with bounding box style
                        # Get node ID for this connection
                        node_id = self.active_connections[camera_id]

                        # Get node load and capacity for color
                        node_load = 0.5  # Default load
                        if node_id in network_simulator.nodes:
                            node_load = network_simulator.nodes[node_id].get('load', 0.5)

                        # Determine capacity based on node ID
                        capacity = 1.0
                        if 'node_1' in node_id or 'node_2' in node_id:
                            capacity = 1.5
                        elif 'node_3' in node_id or 'node_4' in node_id:
                            capacity = 1.2

                        # Calculate score (higher is better)
                        if node_load < 0.1:  # Very low load
                            score = capacity * 10  # Heavily favor capacity for unused nodes
                        else:
                            score = capacity / (node_load + 0.1)  # Balance capacity and load

                        # Color based on score
                        if score > 10:
                            yolo_color = (0, 255, 0)  # Green for high score
                        elif score > 5:
                            yolo_color = (0, 255, 255)  # Yellow for medium score
                        else:
                            yolo_color = (0, 0, 255)  # Red for low score

                        # Draw YOLO-style bounding box
                        box_size = 12
                        # Draw corners only (YOLO style)
                        line_length = int(box_size * 0.6)

                        # Top-left corner
                        cv2.line(image, (x - box_size, y - box_size), (x - box_size + line_length, y - box_size), yolo_color, 2, cv2.LINE_AA)
                        cv2.line(image, (x - box_size, y - box_size), (x - box_size, y - box_size + line_length), yolo_color, 2, cv2.LINE_AA)

                        # Top-right corner
                        cv2.line(image, (x + box_size, y - box_size), (x + box_size - line_length, y - box_size), yolo_color, 2, cv2.LINE_AA)
                        cv2.line(image, (x + box_size, y - box_size), (x + box_size, y - box_size + line_length), yolo_color, 2, cv2.LINE_AA)

                        # Bottom-left corner
                        cv2.line(image, (x - box_size, y + box_size), (x - box_size + line_length, y + box_size), yolo_color, 2, cv2.LINE_AA)
                        cv2.line(image, (x - box_size, y + box_size), (x - box_size, y + box_size - line_length), yolo_color, 2, cv2.LINE_AA)

                        # Bottom-right corner
                        cv2.line(image, (x + box_size, y + box_size), (x + box_size - line_length, y + box_size), yolo_color, 2, cv2.LINE_AA)
                        cv2.line(image, (x + box_size, y + box_size), (x + box_size, y + box_size - line_length), yolo_color, 2, cv2.LINE_AA)

                        # Add confidence score like YOLO
                        conf_score = min(0.99, score / 15.0)  # Scale score to 0-0.99 range
                        conf_text = f"{conf_score:.2f}"
                        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        text_x = x - text_size[0] // 2
                        text_y = y - box_size - 5

                        # Draw text background
                        cv2.rectangle(image,
                                     (text_x - 2, text_y - text_size[1] - 2),
                                     (text_x + text_size[0] + 2, text_y + 2),
                                     (0, 0, 0), -1, cv2.LINE_AA)

                        # Draw confidence text
                        safe_putText(image, conf_text, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, yolo_color, 1, cv2.LINE_AA)

                    else:
                        # Default: Simple circles with tail
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
            'direct': "Direct Connection: All traffic goes to a single node.\nSimple but not scalable. Good for small deployments with one powerful server.",
            'round_robin': "Round Robin: Distributes traffic evenly across all nodes in sequence.\nBalances load but doesn't consider node capacity or current load.",
            'least_connection': "Least Connection: Sends traffic to the node with the fewest active connections.\nAdaptive to changing loads. Good for mixed workloads.",
            'weighted': "Weighted Distribution: Routes based on node capacity weights.\nSends more traffic to higher capacity nodes. Good for heterogeneous clusters.",
            'ip_hash': "IP Hash: Consistently routes the same camera to the same node.\nEnsures session persistence. Good for stateful processing.",
            'yolov8': "YOLOv8 Optimized: Smart routing optimized for YOLOv8 processing.\nBalances between node capacity and current load. Ideal for AI workloads."
        }

        return descriptions.get(algorithm, "Unknown algorithm")


# Create a default instance
network_visualizer = NetworkVisualizer()
