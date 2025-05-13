"""
Network simulation module for demonstrating routing algorithms.

This module provides functionality to simulate a distributed camera network
and demonstrate different routing algorithms.
"""

import time
import random
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque
import threading
import queue
import os
import subprocess
import sys

from vigilance_system.utils.logger import get_logger
from vigilance_system.utils.config import config
from vigilance_system.network.node_client import node_client

# Initialize logger
logger = get_logger(__name__)


class NetworkNode:
    """Simulated network node for processing video frames."""

    def __init__(self, node_id: str, processing_capacity: float = 1.0, latency: float = 0.01):
        """
        Initialize a network node.

        Args:
            node_id: Unique identifier for the node
            processing_capacity: Relative processing capacity (1.0 = standard)
            latency: Network latency in seconds
        """
        self.node_id = node_id
        self.processing_capacity = processing_capacity
        self.latency = latency
        self.load = 0.0  # Current load (0.0 - 1.0)
        self.frames_processed = 0
        self.processing_queue = queue.Queue(maxsize=100)
        self.is_running = False
        self.processing_thread = None

        logger.info(f"Created network node {node_id} with capacity={processing_capacity}, latency={latency}")

    def start(self):
        """Start the node's processing thread."""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
            self.processing_thread.start()
            logger.info(f"Started network node {self.node_id}")

    def stop(self):
        """Stop the node's processing thread."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            logger.info(f"Stopped network node {self.node_id}")

    def _process_frames(self):
        """Process frames from the queue."""
        while self.is_running:
            try:
                # Get frame from queue with timeout
                frame_data = self.processing_queue.get(timeout=0.1)

                # Simulate processing time based on capacity
                processing_time = 0.05 / self.processing_capacity
                time.sleep(processing_time)

                # Update metrics
                self.frames_processed += 1
                self.load = min(1.0, self.processing_queue.qsize() / self.processing_queue.maxsize)

                # Mark task as done
                self.processing_queue.task_done()

            except queue.Empty:
                # No frames to process
                self.load = 0.0
                time.sleep(0.01)

    def send_frame(self, frame_data: Dict[str, Any]) -> bool:
        """
        Send a frame to this node for processing.

        Args:
            frame_data: Frame data including frame, camera_id, and timestamp

        Returns:
            bool: True if frame was accepted, False if queue is full
        """
        try:
            # Simulate network latency
            time.sleep(self.latency)

            # Try to add to queue without blocking
            self.processing_queue.put_nowait(frame_data)
            return True
        except queue.Full:
            # Queue is full, frame dropped
            logger.warning(f"Node {self.node_id} dropped frame - queue full")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get node statistics.

        Returns:
            Dict[str, Any]: Dictionary with node statistics
        """
        return {
            'node_id': self.node_id,
            'processing_capacity': self.processing_capacity,
            'latency': self.latency,
            'load': self.load,
            'frames_processed': self.frames_processed,
            'queue_size': self.processing_queue.qsize(),
            'queue_capacity': self.processing_queue.maxsize
        }


class NetworkSimulator:
    """Simulates a distributed camera network with different routing algorithms."""

    def __init__(self, num_nodes: int = 10):
        """
        Initialize the network simulator.

        Args:
            num_nodes: Number of nodes in the network (default: 10)
        """
        self.nodes = {}
        self.cameras = {}
        self.num_nodes = num_nodes  # Store the number of nodes
        self.routing_algorithm = config.get('network.routing_algorithm', 'direct')
        self.frame_rate = config.get('network.frame_rate', 25)
        self.resolution = config.get('network.resolution', 'medium')
        self.node_processes = []

        # Initialize metrics
        self.frames_sent = 0
        self.frames_dropped = 0
        self.bandwidth_usage = 0.0
        self.last_update_time = time.time()

        # Launch real network nodes if not already running
        self._launch_network_nodes(num_nodes)

        # Connect to nodes
        node_client.connect()
        node_client.set_algorithm(self.routing_algorithm)

        logger.info(f"Initialized network simulator with {num_nodes} nodes and routing algorithm: {self.routing_algorithm}")

    def _launch_network_nodes(self, num_nodes: int):
        """
        Launch network nodes in separate terminals.

        Args:
            num_nodes: Number of nodes to launch
        """
        # First, make sure no old nodes are running
        self._stop_all_node_processes()

        # Check if nodes are already running
        nodes_file = os.path.join(os.getcwd(), 'nodes.json')

        # Remove the nodes.json file to force recreation
        if os.path.exists(nodes_file):
            try:
                os.remove(nodes_file)
                logger.info(f"Removed existing {nodes_file} to ensure clean node creation")
            except Exception as e:
                logger.error(f"Error removing nodes file: {str(e)}")
                # If we can't remove the file, try to continue anyway

        # Get the path to the launch_nodes script
        script_path = os.path.join(os.path.dirname(__file__), 'launch_nodes.py')

        # Launch nodes
        try:
            python_exe = sys.executable
            cmd = [
                python_exe,
                script_path,
                '--nodes', str(num_nodes),
                '--output', nodes_file,
                '--visible'
            ]

            # Launch in a separate process
            process = subprocess.Popen(cmd)

            # Store the process ID for later cleanup
            self.node_processes.append(process.pid)

            # Wait for nodes to start - increase wait time to ensure all nodes have time to initialize
            wait_time = num_nodes * 2.0  # 2 seconds per node
            logger.info(f"Waiting {wait_time} seconds for {num_nodes} nodes to start...")
            time.sleep(wait_time)

            # Verify nodes were created
            if os.path.exists(nodes_file):
                try:
                    with open(nodes_file, 'r') as f:
                        created_nodes = json.load(f)
                        logger.info(f"Successfully created {len(created_nodes)} nodes")
                except Exception as e:
                    logger.error(f"Error reading created nodes file: {str(e)}")
            else:
                logger.error(f"Nodes file {nodes_file} was not created")

            logger.info(f"Launched {num_nodes} network nodes")

        except Exception as e:
            logger.error(f"Error launching network nodes: {str(e)}")

    def register_camera(self, camera_id: str) -> None:
        """
        Register a camera with the network.

        Args:
            camera_id: Unique identifier for the camera
        """
        if camera_id not in self.cameras:
            # Assign random weights for weighted routing
            weights = {node_id: random.uniform(0.1, 1.0) for node_id in self.nodes}

            self.cameras[camera_id] = {
                'camera_id': camera_id,
                'frames_sent': 0,
                'last_node': None,
                'weights': weights,
                'connection_count': {node_id: 0 for node_id in self.nodes}
            }

            logger.info(f"Registered camera {camera_id} with the network")

    def set_routing_algorithm(self, algorithm: str) -> None:
        """
        Set the routing algorithm to use.

        Args:
            algorithm: Name of the routing algorithm
        """
        valid_algorithms = ['direct', 'round_robin', 'least_connection', 'weighted']
        if algorithm not in valid_algorithms:
            logger.warning(f"Invalid routing algorithm: {algorithm}. Using 'direct' instead.")
            algorithm = 'direct'

        # If the algorithm is changing, restart the nodes
        if hasattr(self, 'routing_algorithm') and self.routing_algorithm != algorithm:
            logger.info(f"Routing algorithm changing from {self.routing_algorithm} to {algorithm}, restarting nodes")

            # Stop all nodes
            self._stop_all_node_processes()

            # Remove nodes.json to force recreation with new algorithm
            nodes_file = os.path.join(os.getcwd(), 'nodes.json')
            if os.path.exists(nodes_file):
                try:
                    os.remove(nodes_file)
                    logger.info(f"Removed {nodes_file} to force node recreation with new algorithm")
                except Exception as e:
                    logger.error(f"Error removing nodes file: {str(e)}")

            # Wait a moment for processes to terminate
            time.sleep(2)

            # Launch new nodes
            self._launch_network_nodes(self.num_nodes)

            # Reconnect the node client
            try:
                node_client.disconnect()
                time.sleep(1)  # Give time for disconnect to complete
                node_client.connect()

                # Set the algorithm in the node client
                node_client.set_algorithm(algorithm)

                # Wait for the algorithm to be applied
                time.sleep(1)

                # Verify the algorithm was set correctly
                if node_client.current_algorithm != algorithm:
                    logger.warning(f"Algorithm mismatch: set {algorithm}, but node_client has {node_client.current_algorithm}")
                    # Try again
                    node_client.set_algorithm(algorithm)
            except Exception as e:
                logger.error(f"Error reconnecting node client: {str(e)}")

        # Always update our local routing algorithm
        self.routing_algorithm = algorithm

        # Update the node client even if algorithm didn't change
        try:
            node_client.set_algorithm(algorithm)
        except Exception as e:
            logger.error(f"Error setting algorithm in node client: {str(e)}")

        logger.info(f"Set routing algorithm to: {algorithm}")

        # Save to config
        config.set('network.routing_algorithm', algorithm, save=True)

    def _stop_all_node_processes(self):
        """Stop all node processes."""
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and len(cmdline) > 2:
                        cmd_str = ' '.join(cmdline)
                        if 'node_server.py' in cmd_str or 'Network Nodes Launcher' in cmd_str:
                            logger.info(f"Terminating node process: {proc.pid}")
                            proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                    logger.error(f"Error terminating process: {str(e)}")
        except ImportError:
            logger.warning("psutil not available, can't terminate node processes")

    def send_frame(self, camera_id: str, frame: np.ndarray, frame_id: int) -> Dict[str, Any]:
        """
        Send a frame from a camera to a node based on the routing algorithm.

        Args:
            camera_id: ID of the camera sending the frame
            frame: Video frame
            frame_id: ID of the frame

        Returns:
            Dict[str, Any]: Dictionary with routing information
        """
        # Register camera if not already registered
        if camera_id not in self.cameras:
            self.register_camera(camera_id)

        # Send frame to a node using the node client
        routing_info = node_client.send_frame(camera_id, frame)

        # Update metrics
        self.frames_sent += 1
        self.cameras[camera_id]['frames_sent'] += 1

        if 'node_id' in routing_info:
            node_id = routing_info['node_id']
            self.cameras[camera_id]['last_node'] = node_id

            if node_id in self.cameras[camera_id]['connection_count']:
                self.cameras[camera_id]['connection_count'][node_id] += 1

        if not routing_info.get('success', False):
            self.frames_dropped += 1

        # Calculate bandwidth usage based on frame size and resolution
        height, width = frame.shape[:2]
        frame_size = height * width * 3  # RGB bytes

        # Apply compression factor based on resolution
        compression_factor = 0.1  # Default compression
        if self.resolution == 'low':
            compression_factor = 0.05
        elif self.resolution == 'medium':
            compression_factor = 0.1
        elif self.resolution == 'high':
            compression_factor = 0.2

        # Calculate bandwidth in MB/s
        self.bandwidth_usage = (frame_size * compression_factor * self.frame_rate) / (1024 * 1024)

        # Return routing information
        return {
            'camera_id': camera_id,
            'node_id': routing_info.get('node_id', 'unknown'),
            'success': routing_info.get('success', False),
            'timestamp': time.time(),
            'routing_algorithm': routing_info.get('algorithm', self.routing_algorithm)
        }

    def _select_node(self, camera_id: str, frame_id: int) -> str:
        """
        Select a node based on the current routing algorithm.

        Args:
            camera_id: ID of the camera
            frame_id: ID of the frame

        Returns:
            str: ID of the selected node
        """
        if self.routing_algorithm == 'direct':
            # Always use the first node
            return list(self.nodes.keys())[0]

        elif self.routing_algorithm == 'round_robin':
            # Use frame_id for round robin
            node_ids = list(self.nodes.keys())
            return node_ids[frame_id % len(node_ids)]

        elif self.routing_algorithm == 'least_connection':
            # Find node with lowest load
            return min(self.nodes.items(), key=lambda x: x[1].load)[0]

        elif self.routing_algorithm == 'weighted':
            # Use weighted random selection
            weights = self.cameras[camera_id]['weights']
            node_ids = list(weights.keys())
            node_weights = [weights[node_id] for node_id in node_ids]
            return random.choices(node_ids, weights=node_weights, k=1)[0]

        # IP Hash algorithm removed as it's not a routing algorithm


        # Default to first node
        return list(self.nodes.keys())[0]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get network statistics.

        Returns:
            Dict[str, Any]: Dictionary with network statistics
        """
        # Get client stats
        client_stats = node_client.get_stats()

        # Always update our routing algorithm from the node client
        # This ensures we're showing the correct algorithm
        self.routing_algorithm = client_stats.get('algorithm', self.routing_algorithm)

        # If we were previously in simulation mode but now have real connections,
        # resume any paused cameras
        was_simulation = getattr(self, 'was_simulation_mode', True)
        current_simulation = client_stats.get('using_simulation_mode', True)

        if was_simulation and not current_simulation:
            try:
                # Import here to avoid circular imports
                from vigilance_system.video_acquisition.stream_manager import stream_manager
                if stream_manager:
                    stream_manager.resume_cameras()
                    logger.info("Resumed cameras after reconnecting to network nodes")
            except Exception as e:
                logger.error(f"Error resuming cameras: {str(e)}")

        # Store current simulation mode for next check
        self.was_simulation_mode = current_simulation

        # Calculate metrics
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time

        # Calculate frame rate
        if elapsed_time > 0:
            actual_frame_rate = self.frames_sent / elapsed_time
        else:
            actual_frame_rate = 0

        # Calculate packet loss
        if self.frames_sent > 0:
            packet_loss = (self.frames_dropped / self.frames_sent) * 100
        else:
            packet_loss = 0

        # Reset counters
        self.frames_sent = 0
        self.frames_dropped = 0
        self.last_update_time = current_time

        # Calculate average latency (simulated)
        avg_latency = 0.03  # 30ms default

        # Calculate jitter (simulated)
        jitter = 0.005  # 5ms default

        # Generate some random variation in the metrics to make them look more realistic
        # This helps show that the stats are updating
        bandwidth_variation = random.uniform(0.9, 1.1)
        latency_variation = random.uniform(0.9, 1.1)
        jitter_variation = random.uniform(0.8, 1.2)

        return {
            'routing_algorithm': self.routing_algorithm,
            'frame_rate': self.frame_rate,
            'resolution': self.resolution,
            'actual_frame_rate': client_stats.get('frame_rate', actual_frame_rate),
            'bandwidth': self.bandwidth_usage * bandwidth_variation,  # MB/s with variation
            'packet_loss': client_stats.get('drop_rate', packet_loss/100) * 100,  # %
            'avg_latency': avg_latency * latency_variation,  # seconds with variation
            'jitter': jitter * jitter_variation,  # seconds with variation
            'nodes': client_stats.get('nodes', 0),
            'cameras': self.cameras,
            'client_stats': client_stats,
            'timestamp': current_time  # Add timestamp to force updates
        }

    def stop(self):
        """Stop all nodes in the network."""
        # Disconnect from nodes
        node_client.disconnect()

        # Kill all node processes
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and len(cmdline) > 2:
                        cmd_str = ' '.join(cmdline)
                        if 'node_server.py' in cmd_str or 'Network Nodes Launcher' in cmd_str:
                            logger.info(f"Terminating node process: {proc.pid}")
                            proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                    logger.error(f"Error terminating process: {str(e)}")
        except ImportError:
            logger.warning("psutil not available, can't terminate node processes")

        logger.info("Disconnected from all network nodes")


# Create a default instance
network_simulator = NetworkSimulator(num_nodes=10)
