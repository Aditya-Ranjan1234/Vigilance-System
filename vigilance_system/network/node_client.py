"""
Network node client for distributed video processing.

This module provides a client implementation to communicate with network nodes
for distributed video processing.
"""

import os  # Used for file operations and environment variables
import sys  # Used for executable path
import time
import socket
import json
import threading
import queue
import logging
import random
import base64
import cv2
import numpy as np
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('node_client.log')
    ]
)

logger = logging.getLogger('node_client')


class NodeClient:
    """Client for communicating with network nodes."""

    def __init__(self, nodes_file: str = 'nodes.json'):
        """
        Initialize the node client.

        Args:
            nodes_file: Path to the JSON file with node information
        """
        self.nodes_file = nodes_file
        self.nodes = self._load_nodes()
        self.node_sockets = {}
        self.response_queues = {}
        self.running = False

        # Initialize with default algorithm
        self.current_algorithm = 'direct'

        # Try to load saved algorithm from current_algorithm.txt in the current directory
        try:
            # First check for current_algorithm.txt in the current working directory
            algorithm_file = os.path.join(os.getcwd(), 'current_algorithm.txt')
            if os.path.exists(algorithm_file):
                with open(algorithm_file, 'r') as f:
                    saved_algorithm = f.read().strip()
                    if saved_algorithm in ['direct', 'round_robin', 'least_connection', 'weighted']:
                        self.current_algorithm = saved_algorithm
                        logger.info(f"Loaded saved algorithm from current directory: {saved_algorithm}")
                    else:
                        logger.warning(f"Invalid saved algorithm in current directory: {saved_algorithm}, using default")
                        # Write the default algorithm to the file
                        with open(algorithm_file, 'w') as f:
                            f.write('direct')
                        logger.info("Wrote default algorithm 'direct' to current_algorithm.txt")
            else:
                # If file doesn't exist in current directory, create it with default algorithm
                with open(algorithm_file, 'w') as f:
                    f.write('direct')
                logger.info("Created current_algorithm.txt with default algorithm 'direct'")
        except Exception as e:
            logger.warning(f"Failed to load/create algorithm file in current directory: {str(e)}")

        self.frame_counter = 0
        self.camera_counters = {}
        self.camera_last_nodes = {}

        # Statistics
        self.frames_sent = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.start_time = time.time()

        logger.info(f"Initialized node client with {len(self.nodes)} nodes")

    def _load_nodes(self) -> List[Dict[str, Any]]:
        """
        Load node information from the JSON file.

        Returns:
            List[Dict[str, Any]]: List of node information dictionaries
        """
        try:
            with open(self.nodes_file, 'r') as f:
                nodes = json.load(f)

            logger.info(f"Loaded {len(nodes)} nodes from {self.nodes_file}")
            return nodes

        except Exception as e:
            logger.error(f"Error loading nodes from {self.nodes_file}: {str(e)}")
            return []

    def connect(self):
        """Connect to all nodes."""
        # Initialize simulation mode flag
        self.using_simulation_mode = False

        # If no nodes file exists, create a default one
        if not self.nodes:
            self._create_default_nodes_file()
            self.nodes = self._load_nodes()

        # Try to launch nodes if they don't exist
        self._ensure_nodes_running()

        # Wait longer for nodes to start up
        logger.info("Waiting 5 seconds for nodes to start up...")
        time.sleep(5)  # Give nodes more time to start up, but not too long

        # Try to connect to each node with a reasonable timeout
        max_retries = 5  # Reduced number of retries to avoid excessive waiting
        connected_count = 0

        for retry in range(max_retries):
            if retry > 0:
                logger.info(f"Retry attempt {retry}/{max_retries} to connect to nodes...")
                # Use a fixed wait time to avoid excessive waiting
                wait_time = 3
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

            for node in self.nodes:
                node_id = node['node_id']
                host = node['host']
                port = node['port']

                # Skip if already connected
                if node_id in self.node_sockets and self.node_sockets[node_id] is not None:
                    continue

                try:
                    # Create socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)  # Longer timeout to give nodes more time to respond
                    logger.info(f"Attempting to connect to node {node_id} at {host}:{port}...")

                    # Try to connect
                    try:
                        sock.connect((host, port))
                        sock.settimeout(None)  # Reset timeout for normal operation

                        # Store socket
                        self.node_sockets[node_id] = sock

                        # Create response queue
                        self.response_queues[node_id] = queue.Queue()

                        # Start response handler thread
                        thread = threading.Thread(
                            target=self._handle_responses,
                            args=(node_id, sock)
                        )
                        thread.daemon = True
                        thread.start()

                        connected_count += 1
                        logger.info(f"Connected to node {node_id} at {host}:{port}")

                    except (socket.timeout, ConnectionRefusedError) as e:
                        # Try a different port in case the default one is in use
                        alternate_port = port + 1
                        logger.warning(f"Failed to connect to {host}:{port}, trying alternate port {alternate_port}...")

                        try:
                            sock.close()  # Close the failed socket
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(3)
                            sock.connect((host, alternate_port))
                            sock.settimeout(None)

                            # Store socket with alternate port
                            self.node_sockets[node_id] = sock

                            # Create response queue
                            self.response_queues[node_id] = queue.Queue()

                            # Start response handler thread
                            thread = threading.Thread(
                                target=self._handle_responses,
                                args=(node_id, sock)
                            )
                            thread.daemon = True
                            thread.start()

                            connected_count += 1
                            logger.info(f"Connected to node {node_id} at {host}:{alternate_port} (alternate port)")

                        except Exception as inner_e:
                            logger.warning(f"Failed to connect to alternate port: {str(inner_e)}")
                            sock.close()
                            # Will fall through to the exception handler below

                except Exception as e:
                    logger.warning(f"Could not connect to node {node_id} at {host}:{port}: {str(e)}")

            # If we've connected to at least 4 nodes, that's good enough
            if connected_count >= 4:
                logger.info(f"Connected to {connected_count} nodes, which is sufficient")
                break

        # Check if we need to use simulation mode
        if connected_count == 0:
            logger.warning("No real nodes connected, switching to simulation mode")
            self.using_simulation_mode = True

        # Create simulated nodes for any missing connections
        logger.info("Creating simulated nodes for any missing connections")
        self._create_simulated_nodes()

        self.running = True
        logger.info(f"Connected to {len(self.node_sockets)} nodes ({connected_count} real, {len(self.node_sockets) - connected_count} simulated)")

    def _ensure_nodes_running(self):
        """Ensure that network nodes are running."""
        # Check if we should try to launch real nodes
        # If we're already in simulation mode or have existing connections, skip launching
        if hasattr(self, 'using_simulation_mode') and self.using_simulation_mode:
            logger.info("Already in simulation mode, skipping node launch")
            return

        # Check if nodes are already running by checking for processes
        try:
            import psutil
            node_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and len(cmdline) > 2 and 'node_server.py' in ' '.join(cmdline):
                        node_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            if node_processes:
                logger.info(f"Found {len(node_processes)} node processes already running, skipping launch")
                return
        except ImportError:
            logger.warning("psutil not available, can't check for existing node processes")

        # Check if nodes.json exists and has 10 nodes
        if os.path.exists(self.nodes_file):
            try:
                with open(self.nodes_file, 'r') as f:
                    existing_nodes = json.load(f)
                    if len(existing_nodes) != 10:
                        logger.warning(f"Found {len(existing_nodes)} nodes in {self.nodes_file}, but 10 are required. Removing file to recreate nodes.")
                        os.remove(self.nodes_file)
                        # Also clear self.nodes to force recreation
                        self.nodes = []
            except Exception as e:
                logger.error(f"Error reading nodes file: {str(e)}")
                # If there's an error reading the file, remove it and recreate
                try:
                    os.remove(self.nodes_file)
                    # Also clear self.nodes to force recreation
                    self.nodes = []
                except:
                    pass

        try:
            # Check if we need to launch nodes
            import subprocess

            # Get the path to the launch_nodes script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_dir, 'launch_nodes.py')

            # Check if the script exists
            if os.path.exists(script_path):
                logger.info(f"Launching network nodes using {script_path}")

                # Launch nodes in visible terminals
                python_exe = sys.executable
                script_path_abs = os.path.abspath(script_path)

                # Create the command with proper quoting - always launch 10 nodes
                cmd_str = f'"{python_exe}" "{script_path_abs}" --nodes 10 --output "{self.nodes_file}" --visible'

                # Create the terminal command
                terminal_cmd = f'start "Network Nodes Launcher" cmd /k "{cmd_str}"'
                logger.info(f"Launching nodes with command: {cmd_str}")

                try:
                    # Launch the process in a visible terminal
                    subprocess.Popen(terminal_cmd, shell=True)
                    logger.info("Launched nodes in visible terminal")
                    launched = True
                except Exception as e:
                    logger.error(f"Failed to launch nodes in visible terminal: {str(e)}")
                    launched = False

                # If we managed to launch nodes, wait a bit for them to start
                if launched:
                    logger.info("Waiting for nodes to start...")
                    # Wait longer to ensure nodes have time to initialize
                    for i in range(10):
                        logger.info(f"Waiting for nodes to start... {i+1}/10 seconds")
                        time.sleep(1)
                else:
                    # If all launch methods failed, mark as simulation mode
                    logger.warning("Failed to launch real nodes, will use simulation mode")
                    self.using_simulation_mode = True
            else:
                logger.warning(f"Launch script not found at {script_path}, will use simulation mode")
                self.using_simulation_mode = True

        except Exception as e:
            logger.error(f"Error ensuring nodes are running: {str(e)}")
            # Mark as simulation mode on any error
            self.using_simulation_mode = True

    def _create_simulated_nodes(self):
        """Create simulated nodes for all missing connections."""
        # Create a simulated node for each real node that failed to connect
        for node in self.nodes:
            node_id = node['node_id']

            # Skip if already connected
            if node_id in self.node_sockets and self.node_sockets[node_id] is not None:
                continue

            # Create simulated node
            self._create_simulated_node(node_id)
            logger.info(f"Created simulated node for {node_id} because it failed to connect")

        # Make sure we have at least 10 nodes total (real + simulated)
        # This ensures we always have 10 nodes even if some failed to connect
        existing_nodes = list(self.node_sockets.keys())
        for i in range(1, 11):
            node_id = f"node_{i}"
            if node_id not in existing_nodes:
                self._create_simulated_node(node_id)
                logger.info(f"Created additional simulated node {node_id} to ensure 10 total nodes")

        # Create default camera connections if none exist
        if not self.camera_last_nodes:
            logger.info("No camera connections found, creating default connections")

            # Create 3 default cameras
            for i in range(1, 4):
                camera_id = f"Camera {i}"

                # Distribute cameras evenly across nodes
                node_index = (i - 1) % len(self.node_sockets)
                node_id = list(self.node_sockets.keys())[node_index]

                # Set the camera's last node
                self.camera_last_nodes[camera_id] = node_id

                logger.info(f"Created default connection: Camera {i} -> {node_id}")

    def _simulate_frame_processing(self, message):
        """
        Simulate processing a frame.

        Args:
            message: Frame message to process
        """
        # This method simulates the actual processing that would happen on a node
        # For now, we just count it as processed, but in the future we could
        # implement actual frame processing here (object detection, etc.)

        # Log that we're processing the frame
        frame_id = message.get('frame_id', 0)
        camera_id = message.get('camera_id', 'unknown')
        logger.debug(f"Simulating processing of frame {frame_id} from camera {camera_id}")

        # In a real implementation, we would decode the frame and process it
        # For now, we just simulate a small delay
        time.sleep(0.01)

        # Simulate occasional frame drops to make it clear this is simulated
        if random.random() < 0.3:  # 30% chance to drop a frame in simulation
            return False  # Indicate frame was dropped

        return True  # Indicate frame was processed

    def _create_simulated_node(self, node_id="simulated_node"):
        """
        Create a simulated node for testing.

        Args:
            node_id: ID for the simulated node
        """
        self.node_sockets[node_id] = None  # No actual socket
        self.response_queues[node_id] = queue.Queue()

        # Start a thread to simulate responses
        thread = threading.Thread(
            target=self._simulate_responses,
            args=(node_id,)
        )
        thread.daemon = True
        thread.start()

        logger.info(f"Created simulated node: {node_id}")

    def _create_default_nodes_file(self):
        """Create a default nodes file if none exists."""
        default_nodes = []

        # Create 10 nodes with different capacities and latencies
        for i in range(10):
            node_id = f"node_{i+1}"

            # Vary capacity and latency to demonstrate different node capabilities
            if i < 2:
                # High capacity nodes (servers)
                capacity = 2.0
                latency = 0.005
            elif i < 4:
                # Medium-high capacity nodes
                capacity = 1.5
                latency = 0.008
            elif i < 6:
                # Medium capacity nodes
                capacity = 1.0
                latency = 0.01
            elif i < 8:
                # Lower capacity nodes
                capacity = 0.7
                latency = 0.015
            else:
                # Edge nodes
                capacity = 0.5
                latency = 0.02

            node_info = {
                'node_id': node_id,
                'host': 'localhost',
                'port': 8000 + (i * 10),  # Use bigger gaps between ports to avoid conflicts
                'capacity': capacity,
                'latency': latency,
                'node_type': 'processing' if i < 8 else 'edge'
            }
            default_nodes.append(node_info)

        # Save to file
        with open(self.nodes_file, 'w') as f:
            import json
            json.dump(default_nodes, f, indent=2)

        logger.info(f"Created default nodes file: {self.nodes_file}")

    def _simulate_responses(self, node_id):
        """Simulate responses from a node."""
        # Initialize node statistics
        frames_processed = 0
        frames_dropped = 0
        start_time = time.time()
        current_load = 0.0

        # Initialize pending frames list if not exists
        if not hasattr(self, 'pending_frames'):
            self.pending_frames = []

        # Simulate node activity
        while self.running:
            # Simulate processing time (shorter for faster response)
            time.sleep(0.05)

            # Simulate random load changes
            if random.random() < 0.1:  # 10% chance to change load
                current_load = min(0.9, max(0.1, current_load + random.uniform(-0.2, 0.2)))

            # Process any pending frames
            if self.pending_frames:
                # Process a pending frame
                frame_data = self.pending_frames.pop(0)

                # Simulate occasional packet loss
                if random.random() < 0.05:  # 5% chance to drop frame
                    frames_dropped += 1
                    status = 'dropped'
                    # Update global statistics for dropped frames
                    self.frames_dropped += 1
                else:
                    frames_processed += 1
                    status = 'processed'
                    # Note: We don't increment self.frames_processed here
                    # because we already do it in the send_frame method

                # Create a simulated response
                response = {
                    'type': 'frame_response',
                    'frame_id': frame_data.get('frame_id', 0),
                    'camera_id': frame_data.get('camera_id', 'unknown'),
                    'status': status,
                    'node_id': node_id,
                    'timestamp': time.time(),
                    'load': current_load
                }

                # Add to response queue
                self.response_queues[node_id].put(response)

                # Process the response
                self._process_response(response)

            # Periodically send status updates even without frames
            elif random.random() < 0.02:  # 2% chance per cycle
                # Calculate uptime
                uptime = time.time() - start_time

                # Calculate frame rate
                frame_rate = frames_processed / uptime if uptime > 0 else 0

                # Create a status response
                status_response = {
                    'type': 'status_response',
                    'node_id': node_id,
                    'uptime': uptime,
                    'frames_processed': frames_processed,
                    'frames_dropped': frames_dropped,
                    'frame_rate': frame_rate,
                    'load': current_load,
                    'timestamp': time.time()
                }

                # Add to response queue
                self.response_queues[node_id].put(status_response)

                # Process the response
                self._process_response(status_response)

    def disconnect(self):
        """Disconnect from all nodes."""
        self.running = False

        # Signal to any video processing that nodes are disconnected
        self.frames_processed = 0
        self.frames_dropped = 0
        self.frame_counter = 0
        self.camera_counters.clear()
        self.camera_last_nodes.clear()

        # Stop all active connections
        for node_id, sock in self.node_sockets.items():
            try:
                if sock is not None:  # Check if it's not a simulated node
                    sock.close()
                logger.info(f"Disconnected from node {node_id}")
            except Exception as e:
                logger.error(f"Error disconnecting from node {node_id}: {str(e)}")

        self.node_sockets.clear()
        self.response_queues.clear()

        # Signal that we're disconnected
        self.using_simulation_mode = True
        logger.info("Disconnected from all nodes")

        # Notify any listeners that we've disconnected
        try:
            # Import here to avoid circular imports
            from vigilance_system.video_acquisition.stream_manager import stream_manager
            if stream_manager:
                stream_manager.on_network_disconnect()
        except Exception as e:
            logger.error(f"Error notifying stream manager of disconnect: {str(e)}")

    def _handle_responses(self, node_id: str, sock: socket.socket):
        """
        Handle responses from a node.

        Args:
            node_id: Node ID
            sock: Socket connected to the node
        """
        try:
            while self.running:
                # Receive message size
                size_data = sock.recv(4)
                if not size_data:
                    break

                # Get message size
                message_size = int.from_bytes(size_data, byteorder='big')

                # Receive message
                message_data = b''
                while len(message_data) < message_size:
                    chunk = sock.recv(min(4096, message_size - len(message_data)))
                    if not chunk:
                        break
                    message_data += chunk

                if len(message_data) < message_size:
                    logger.warning(f"Received incomplete message from node {node_id}")
                    break

                # Parse message
                message = json.loads(message_data.decode('utf-8'))

                # Add to response queue
                self.response_queues[node_id].put(message)

                # Process response
                self._process_response(message)

        except Exception as e:
            if self.running:
                logger.error(f"Error handling responses from node {node_id}: {str(e)}")

        finally:
            if self.running:
                logger.warning(f"Lost connection to node {node_id}")

    def _process_response(self, response: Dict[str, Any]):
        """
        Process a response from a node.

        Args:
            response: Response data
        """
        # Get response type
        response_type = response.get('type')

        if response_type == 'frame_response':
            # Process frame response
            frame_id = response.get('frame_id', 0)
            camera_id = response.get('camera_id', 'unknown')
            status = response.get('status', 'unknown')
            node_id = response.get('node_id', 'unknown')

            if status == 'processed':
                # Only increment for real nodes, not simulated ones
                # For simulated nodes, we already increment in send_frame
                if node_id != "simulated_node" and self.node_sockets.get(node_id) is not None:
                    self.frames_processed += 1
                logger.debug(f"Frame {frame_id} from camera {camera_id} processed by node {node_id}")

            elif status == 'dropped':
                # For real nodes, increment dropped frames
                # For simulated nodes, we already handle this in _simulate_responses
                if node_id != "simulated_node" and self.node_sockets.get(node_id) is not None:
                    self.frames_dropped += 1
                logger.warning(f"Frame {frame_id} from camera {camera_id} dropped by node {node_id}")

        elif response_type == 'status_response':
            # Process status response
            node_id = response.get('node_id', 'unknown')
            logger.debug(f"Received status from node {node_id}")

    def send_frame(self, camera_id: str, frame: np.ndarray) -> Dict[str, Any]:
        """
        Send a frame to a node based on the current routing algorithm.

        Args:
            camera_id: Camera ID
            frame: Frame to send

        Returns:
            Dict[str, Any]: Routing information
        """
        # Increment frame counter
        self.frame_counter += 1

        # Initialize camera counter if needed
        if camera_id not in self.camera_counters:
            self.camera_counters[camera_id] = 0

        # Increment camera counter
        self.camera_counters[camera_id] += 1

        # Select node based on routing algorithm
        node_id = self._select_node(camera_id)

        # Store last node for this camera
        self.camera_last_nodes[camera_id] = node_id

        # Check if node is connected
        if node_id not in self.node_sockets:
            logger.warning(f"Node {node_id} is not connected")
            return {
                'camera_id': camera_id,
                'node_id': node_id,
                'success': False,
                'algorithm': self.current_algorithm,
                'error': 'Node not connected'
            }

        try:
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')

            # Create message
            message = {
                'type': 'frame',
                'frame_id': self.frame_counter,
                'camera_id': camera_id,
                'frame_data': frame_data,
                'timestamp': time.time()
            }

            # Handle simulated node
            if node_id == "simulated_node" or self.node_sockets[node_id] is None:
                # Store frame for simulated processing
                if not hasattr(self, 'pending_frames'):
                    self.pending_frames = []
                self.pending_frames.append(message)

                # Simulate processing the frame
                # This ensures frames are processed even with simulated nodes
                processed = self._simulate_frame_processing(message)

                # Update statistics
                self.frames_sent += 1

                if processed:
                    self.frames_processed += 1  # Count as processed
                else:
                    self.frames_dropped += 1  # Count as dropped

                # Log simulation status
                logger.info(f"Simulated node {node_id} {'processed' if processed else 'dropped'} frame {self.frame_counter} from camera {camera_id}")

                # Return routing information
                return {
                    'camera_id': camera_id,
                    'node_id': node_id,
                    'success': True,
                    'algorithm': self.current_algorithm,
                    'frame_id': self.frame_counter
                }

            # Encode message for real node
            message_data = json.dumps(message).encode('utf-8')

            # Send message size
            size_data = len(message_data).to_bytes(4, byteorder='big')
            self.node_sockets[node_id].sendall(size_data)

            # Send message
            self.node_sockets[node_id].sendall(message_data)

            # Update statistics
            self.frames_sent += 1

            # Return routing information
            return {
                'camera_id': camera_id,
                'node_id': node_id,
                'success': True,
                'algorithm': self.current_algorithm,
                'frame_id': self.frame_counter
            }

        except Exception as e:
            logger.error(f"Error sending frame to node {node_id}: {str(e)}")

            # If error occurs, try to create a simulated node as fallback
            if "simulated_node" not in self.node_sockets:
                logger.info("Creating simulated node as fallback")
                self._create_simulated_node("simulated_node")

                # Try again with simulated node
                return self.send_frame(camera_id, frame)

            return {
                'camera_id': camera_id,
                'node_id': node_id,
                'success': False,
                'algorithm': self.current_algorithm,
                'error': str(e)
            }

    def _select_node(self, camera_id: str) -> str:
        """
        Select a node based on the current routing algorithm.

        Args:
            camera_id: ID of the camera to route

        Returns:
            str: ID of the selected node
        """
        if not self.node_sockets:
            return "no_nodes"

        if self.current_algorithm == 'direct':
            # Always use the first node
            return list(self.node_sockets.keys())[0]

        elif self.current_algorithm == 'round_robin':
            # Use a global counter for true round robin across all cameras
            # This ensures we're cycling through nodes regardless of which camera is sending
            if not hasattr(self, 'global_rr_counter'):
                self.global_rr_counter = 0

            # Get sorted node IDs for consistent ordering
            node_ids = sorted(list(self.node_sockets.keys()))

            # Select node and increment counter
            selected_node = node_ids[self.global_rr_counter % len(node_ids)]
            self.global_rr_counter += 1

            # Log the selection for debugging
            logger.debug(f"Round Robin selected node {selected_node} (counter: {self.global_rr_counter-1})")

            return selected_node

        elif self.current_algorithm == 'least_connection':
            # Get node status and select the one with the lowest load
            # This is a simplified version - in a real system, we would
            # query the nodes for their current load
            node_loads = {}

            for node_id in self.node_sockets.keys():
                # Count how many cameras are using this node
                count = sum(1 for last_node in self.camera_last_nodes.values() if last_node == node_id)
                node_loads[node_id] = count

            # Select node with lowest load
            return min(node_loads.items(), key=lambda x: x[1])[0]

        elif self.current_algorithm == 'weighted':
            # Use weighted random selection based on node capacity and current load
            node_weights = {}

            # First, get current load for each node
            node_loads = {}
            for node_id in self.node_sockets.keys():
                # Count how many cameras are using this node
                count = sum(1 for last_node in self.camera_last_nodes.values() if last_node == node_id)
                node_loads[node_id] = count

            # Calculate weights based on capacity and current load
            for node in self.nodes:
                node_id = node['node_id']
                if node_id in self.node_sockets:
                    # Get capacity from node info
                    capacity = node.get('capacity', 1.0)

                    # Get current load
                    current_load = node_loads.get(node_id, 0)

                    # Calculate weight: higher capacity and lower load = higher weight
                    # Add 1 to load to avoid division by zero
                    weight = capacity / (current_load + 1)

                    # Store the weight
                    node_weights[node_id] = weight

                    # Log the weight calculation for debugging
                    logger.debug(f"Node {node_id}: capacity={capacity}, load={current_load}, weight={weight}")

            # If no weights were calculated, use default behavior
            if not node_weights:
                return list(self.node_sockets.keys())[0]

            # Select node based on weights
            total_weight = sum(node_weights.values())
            if total_weight <= 0:
                return list(self.node_sockets.keys())[0]

            r = random.random() * total_weight
            cumulative_weight = 0

            for node_id, weight in node_weights.items():
                cumulative_weight += weight
                if r <= cumulative_weight:
                    # Log the selected node for debugging
                    logger.debug(f"Weighted algorithm selected node {node_id} with weight {weight}")
                    return node_id

            return list(self.node_sockets.keys())[0]

        # Default fallback for any algorithm
        # If we reach here, use the first available node
        if self.node_sockets:
            return list(self.node_sockets.keys())[0]
        return None

    def set_algorithm(self, algorithm: str):
        """
        Set the routing algorithm.

        Args:
            algorithm: Routing algorithm name
        """
        valid_algorithms = ['direct', 'round_robin', 'least_connection', 'weighted']

        if algorithm not in valid_algorithms:
            logger.warning(f"Invalid algorithm: {algorithm}. Using 'direct' instead.")
            algorithm = 'direct'

        # Store the previous algorithm for logging
        previous_algorithm = getattr(self, 'current_algorithm', 'unknown')

        # Only proceed if the algorithm is actually changing
        if previous_algorithm == algorithm:
            logger.info(f"Algorithm already set to {algorithm}, no change needed")
            return self.current_algorithm

        # Set the new algorithm
        self.current_algorithm = algorithm

        # Log the change
        logger.info(f"Set routing algorithm from {previous_algorithm} to {algorithm}")

        # Save the algorithm to current_algorithm.txt in the current directory
        try:
            # Always save to current_algorithm.txt in the current working directory
            algorithm_file = os.path.join(os.getcwd(), 'current_algorithm.txt')
            with open(algorithm_file, 'w') as f:
                f.write(algorithm)
            logger.info(f"Saved algorithm to current_algorithm.txt: {algorithm}")
        except Exception as e:
            logger.error(f"Failed to save algorithm to current_algorithm.txt: {str(e)}")

        # Force a reset of active connections to ensure the new algorithm takes effect
        self.camera_last_nodes = {}

        # Reset counters to ensure clean routing patterns
        self.camera_counters = {}

        # Notify all connected nodes about the algorithm change
        self._notify_nodes_of_algorithm_change(algorithm)

        # Save the algorithm to a file to ensure persistence across restarts
        try:
            algorithm_file = os.path.join(os.path.dirname(self.nodes_file), 'current_algorithm.txt')
            with open(algorithm_file, 'w') as f:
                f.write(algorithm)
            logger.info(f"Saved algorithm {algorithm} to {algorithm_file}")
        except Exception as e:
            logger.error(f"Failed to save algorithm to file: {str(e)}")

        return self.current_algorithm

    def _notify_nodes_of_algorithm_change(self, algorithm: str):
        """
        Notify all connected nodes about an algorithm change.

        Args:
            algorithm: The new routing algorithm
        """
        # Create a message to notify nodes about the algorithm change
        message = {
            'type': 'algorithm_change',
            'algorithm': algorithm,
            'timestamp': time.time()
        }

        # Send to all real nodes
        for node_id, sock in self.node_sockets.items():
            # Skip simulated nodes
            if sock is None:
                continue

            try:
                # Encode message
                message_data = json.dumps(message).encode('utf-8')

                # Send message size
                size_data = len(message_data).to_bytes(4, byteorder='big')
                sock.sendall(size_data)

                # Send message
                sock.sendall(message_data)

                # Force a status request to update client counts
                status_message = {
                    'type': 'status',
                    'timestamp': time.time()
                }

                # Encode status message
                status_data = json.dumps(status_message).encode('utf-8')

                # Send status message size
                size_data = len(status_data).to_bytes(4, byteorder='big')
                sock.sendall(size_data)

                # Send status message
                sock.sendall(status_data)

                logger.info(f"Notified node {node_id} about algorithm change to {algorithm}")
            except Exception as e:
                logger.error(f"Failed to notify node {node_id} about algorithm change: {str(e)}")

                # Try to reconnect if the connection was lost
                try:
                    # Find the node info
                    node_info = next((node for node in self.nodes if node['node_id'] == node_id), None)
                    if node_info:
                        # Close the old socket
                        try:
                            sock.close()
                        except:
                            pass

                        # Create a new socket
                        new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        new_sock.settimeout(3)

                        # Try to connect
                        host = node_info.get('host', 'localhost')
                        port = node_info.get('port', 8000)
                        logger.info(f"Attempting to reconnect to node {node_id} at {host}:{port}...")

                        try:
                            new_sock.connect((host, port))
                            new_sock.settimeout(None)

                            # Update the socket
                            self.node_sockets[node_id] = new_sock

                            # Create a new response queue
                            self.response_queues[node_id] = queue.Queue()

                            # Start a new response handler thread
                            thread = threading.Thread(
                                target=self._handle_responses,
                                args=(node_id, new_sock)
                            )
                            thread.daemon = True
                            thread.start()

                            logger.info(f"Successfully reconnected to node {node_id}")
                        except Exception as reconnect_error:
                            logger.error(f"Failed to reconnect to node {node_id}: {str(reconnect_error)}")

                            # Create a simulated node as fallback
                            self._create_simulated_node(node_id)
                except Exception as node_error:
                    logger.error(f"Error handling node {node_id} reconnection: {str(node_error)}")

        # For simulated nodes, we don't need to do anything special
        # as they'll pick up the new algorithm from self.current_algorithm

    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.

        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        # Calculate uptime
        uptime = time.time() - self.start_time

        # Calculate frame rate
        if uptime > 0:
            frame_rate = self.frames_sent / uptime
        else:
            frame_rate = 0

        # Calculate drop rate
        if self.frames_sent > 0:
            drop_rate = self.frames_dropped / self.frames_sent
        else:
            drop_rate = 0

        # Count real vs simulated nodes
        real_nodes = 0
        simulated_nodes = 0
        real_node_ids = []
        simulated_node_ids = []

        for node_id, sock in self.node_sockets.items():
            if sock is None:
                simulated_nodes += 1
                simulated_node_ids.append(node_id)
            else:
                real_nodes += 1
                real_node_ids.append(node_id)

        # Check if we're in simulation mode
        # Either explicitly set or determined by node counts
        using_simulation = hasattr(self, 'using_simulation_mode') and self.using_simulation_mode

        # If all nodes are simulated or more than half are simulated, consider it simulation mode
        if not using_simulation and simulated_nodes > 0 and (real_nodes == 0 or simulated_nodes >= real_nodes):
            using_simulation = True
            # Update the flag for future reference
            self.using_simulation_mode = True

        return {
            'algorithm': self.current_algorithm,
            'nodes': len(self.node_sockets),
            'real_nodes': real_nodes,
            'simulated_nodes': simulated_nodes,
            'real_node_ids': real_node_ids,
            'simulated_node_ids': simulated_node_ids,
            'cameras': len(self.camera_counters),
            'frames_sent': self.frames_sent,
            'frames_processed': self.frames_processed,
            'frames_dropped': self.frames_dropped,
            'frame_rate': frame_rate,
            'drop_rate': drop_rate,
            'uptime': uptime,
            'using_simulation': using_simulation
        }


# Create a global instance
node_client = NodeClient()
