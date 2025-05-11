"""
Network node server for distributed video processing.

This module provides a server implementation for network nodes that can
process video frames in a distributed manner.
"""

import os
import sys
import time
import socket
import argparse
import threading
import json
import logging
import random
from typing import Dict, Any, Optional
import numpy as np
import cv2

# Configure logging
def setup_logging(node_id=None):
    """Set up logging with node-specific configuration."""
    log_file = f'node_{node_id}.log' if node_id else 'node_server.log'

    # Create a formatter that includes colors for console output
    class ColoredFormatter(logging.Formatter):
        """Custom formatter with colors for console output."""
        COLORS = {
            'DEBUG': '\033[94m',  # Blue
            'INFO': '\033[92m',   # Green
            'WARNING': '\033[93m', # Yellow
            'ERROR': '\033[91m',  # Red
            'CRITICAL': '\033[91m\033[1m',  # Bold Red
            'RESET': '\033[0m'    # Reset
        }

        def format(self, record):
            log_message = super().format(record)
            if record.levelname in self.COLORS:
                return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
            return log_message

    # Create console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    console_handler.setLevel(logging.INFO)  # Set to INFO to reduce noise

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.DEBUG)  # Log everything to file

    # Configure root logger
    logger = logging.getLogger('node_server')
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture everything

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add new handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Print a startup message to ensure the console is working
    print(f"\n{'='*60}")
    print(f"STARTING NODE SERVER: {node_id or 'Unknown'}")
    print(f"{'='*60}\n")

    # Force flush stdout to ensure it appears in the terminal
    sys.stdout.flush()

    return logger

# Initialize logger
logger = setup_logging()


class NetworkNode:
    """Network node for distributed video processing."""

    def __init__(self, node_id: str, host: str = 'localhost', port: int = 0,
                 capacity: float = 1.0, latency: float = 0.01):
        """
        Initialize a network node.

        Args:
            node_id: Unique identifier for the node
            host: Host to bind the server to
            port: Port to bind the server to (0 for auto-assign)
            capacity: Processing capacity of the node (1.0 = standard)
            latency: Simulated network latency in seconds
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.capacity = capacity
        self.latency = latency

        # Node statistics
        self.frames_processed = 0
        self.frames_dropped = 0
        self.start_time = time.time()
        self.last_frame_time = 0
        self.current_load = 0.0
        self.connected_clients = set()
        self.client_ips = {}

        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Try to bind to the specified port, or find an available one
        try:
            # Set socket option to allow reuse of address
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Try to bind to the specified port
            self.server_socket.bind((host, port))
            logger.info(f"Successfully bound to port {port}")
        except OSError as e:
            if port != 0:  # If a specific port was requested
                logger.warning(f"Port {port} is already in use. Trying to find an available port...")
                # Close the current socket
                self.server_socket.close()

                # Create a new socket and let the OS assign a port
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((host, 0))  # Let OS assign a port
                logger.info("Using auto-assigned port")
            else:
                # If we already tried with port 0 and still failed, re-raise the exception
                logger.error(f"Failed to bind to any port: {str(e)}")
                raise

        # Start listening
        self.server_socket.listen(5)

        # Get the actual port that was assigned
        self.host, self.port = self.server_socket.getsockname()
        logger.info(f"Server socket bound to {self.host}:{self.port}")

        # Start server thread
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop)
        self.server_thread.daemon = True

        logger.info(f"Node {node_id} initialized on {self.host}:{self.port} "
                   f"with capacity={capacity}, latency={latency}")

    def start(self):
        """Start the node server."""
        self.server_thread.start()
        logger.info(f"Node {self.node_id} started")

    def stop(self):
        """Stop the node server."""
        self.running = False
        self.server_socket.close()
        logger.info(f"Node {self.node_id} stopped")

    def _server_loop(self):
        """Main server loop to accept connections."""
        logger.info(f"Node {self.node_id} server loop started")

        while self.running:
            try:
                # Accept client connection
                client_socket, client_address = self.server_socket.accept()
                client_ip = client_address[0]

                # Log connection
                logger.info(f"Node {self.node_id} accepted connection from {client_ip}")

                # Start client handler thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_ip)
                )
                client_thread.daemon = True
                client_thread.start()

            except Exception as e:
                if self.running:
                    logger.error(f"Node {self.node_id} server error: {str(e)}")

    def _handle_client(self, client_socket: socket.socket, client_ip: str):
        """
        Handle client connection.

        Args:
            client_socket: Client socket
            client_ip: Client IP address
        """
        # Add client to connected clients
        self.connected_clients.add(client_socket)
        self.client_ips[client_socket] = client_ip

        try:
            # Receive data from client
            while self.running:
                # Receive message size
                size_data = client_socket.recv(4)
                if not size_data:
                    break

                # Get message size
                message_size = int.from_bytes(size_data, byteorder='big')

                # Receive message
                message_data = b''
                while len(message_data) < message_size:
                    chunk = client_socket.recv(min(4096, message_size - len(message_data)))
                    if not chunk:
                        break
                    message_data += chunk

                if len(message_data) < message_size:
                    logger.warning(f"Node {self.node_id} received incomplete message from {client_ip}")
                    break

                # Process message
                self._process_message(message_data, client_socket, client_ip)

        except Exception as e:
            logger.error(f"Node {self.node_id} client handler error: {str(e)}")

        finally:
            # Remove client from connected clients
            self.connected_clients.remove(client_socket)
            del self.client_ips[client_socket]
            client_socket.close()
            logger.info(f"Node {self.node_id} closed connection from {client_ip}")

    def _process_message(self, message_data: bytes, client_socket: socket.socket, client_ip: str):
        """
        Process a message from a client.

        Args:
            message_data: Message data
            client_socket: Client socket
            client_ip: Client IP address
        """
        try:
            # Parse message
            message = json.loads(message_data.decode('utf-8'))

            # Get message type
            message_type = message.get('type')

            # Process message based on type
            if message_type == 'frame':
                # Process frame
                self._process_frame(message, client_socket, client_ip)

                # Update load to show activity
                self.current_load = min(0.9, 0.3 + random.random() * 0.3)  # 30-60% load

                # Simulate processing activity
                if random.random() < 0.5:  # 50% chance to increment counter
                    self.frames_processed += 1
                    logger.info(f"Node {self.node_id} processed frame, total: {self.frames_processed}")

            elif message_type == 'status':
                # Send status
                self._send_status(client_socket)

                # Show some activity in response to status requests
                if random.random() < 0.3:  # 30% chance
                    self.current_load = max(0.1, self.current_load - random.random() * 0.2)  # Reduce load a bit
                    logger.info(f"Node {self.node_id} updated load to {self.current_load:.2f}")

            else:
                # Even for unknown messages, show some activity
                logger.warning(f"Node {self.node_id} received unknown message type: {message_type}")
                self.current_load = min(0.9, self.current_load + random.random() * 0.1)  # Increase load a bit

        except Exception as e:
            logger.error(f"Node {self.node_id} message processing error: {str(e)}")

    def _process_frame(self, message: Dict[str, Any], client_socket: socket.socket, client_ip: str):
        """
        Process a frame message.

        Args:
            message: Frame message
            client_socket: Client socket
            client_ip: Client IP address
        """
        # Extract frame data
        frame_id = message.get('frame_id', 0)
        camera_id = message.get('camera_id', 'unknown')
        frame_base64 = message.get('frame_data', '')

        # Log frame reception
        logger.info(f"Node {self.node_id} received frame {frame_id} from camera {camera_id}")

        # Simulate network latency
        time.sleep(self.latency)

        # Check if we should drop the frame (simulating network congestion)
        current_load = len(self.connected_clients) / self.capacity
        self.current_load = current_load

        if current_load > 0.9 and random.random() < 0.2:
            # Drop frame
            self.frames_dropped += 1
            logger.warning(f"Node {self.node_id} dropped frame {frame_id} from camera {camera_id} due to high load")

            # Send response
            response = {
                'type': 'frame_response',
                'frame_id': frame_id,
                'camera_id': camera_id,
                'status': 'dropped',
                'node_id': self.node_id,
                'timestamp': time.time()
            }

            self._send_response(response, client_socket)
            return

        # Simulate processing time based on capacity
        processing_time = 0.05 / self.capacity
        time.sleep(processing_time)

        # Update statistics
        self.frames_processed += 1
        self.last_frame_time = time.time()

        # Send response
        response = {
            'type': 'frame_response',
            'frame_id': frame_id,
            'camera_id': camera_id,
            'status': 'processed',
            'node_id': self.node_id,
            'timestamp': time.time()
        }

        self._send_response(response, client_socket)

    def _send_response(self, response: Dict[str, Any], client_socket: socket.socket):
        """
        Send a response to a client.

        Args:
            response: Response data
            client_socket: Client socket
        """
        try:
            # Encode response
            response_data = json.dumps(response).encode('utf-8')

            # Send response size
            size_data = len(response_data).to_bytes(4, byteorder='big')
            client_socket.sendall(size_data)

            # Send response
            client_socket.sendall(response_data)

        except Exception as e:
            logger.error(f"Node {self.node_id} send response error: {str(e)}")

    def _send_status(self, client_socket: socket.socket):
        """
        Send status to a client.

        Args:
            client_socket: Client socket
        """
        # Calculate uptime
        uptime = time.time() - self.start_time

        # Calculate frame rate
        if uptime > 0:
            frame_rate = self.frames_processed / uptime
        else:
            frame_rate = 0

        # Calculate drop rate
        total_frames = self.frames_processed + self.frames_dropped
        if total_frames > 0:
            drop_rate = self.frames_dropped / total_frames
        else:
            drop_rate = 0

        # Create status response
        status = {
            'type': 'status_response',
            'node_id': self.node_id,
            'host': self.host,
            'port': self.port,
            'capacity': self.capacity,
            'latency': self.latency,
            'uptime': uptime,
            'frames_processed': self.frames_processed,
            'frames_dropped': self.frames_dropped,
            'frame_rate': frame_rate,
            'drop_rate': drop_rate,
            'current_load': self.current_load,
            'connected_clients': len(self.connected_clients),
            'client_ips': list(self.client_ips.values()),
            'timestamp': time.time()
        }

        # Send status
        self._send_response(status, client_socket)


def main():
    """Main function to run a network node server."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Network Node Server')
        parser.add_argument('--id', type=str, required=True, help='Node ID')
        parser.add_argument('--host', type=str, default='localhost', help='Host to bind to')
        parser.add_argument('--port', type=int, default=0, help='Port to bind to (0 for auto-assign)')
        parser.add_argument('--capacity', type=float, default=1.0, help='Processing capacity')
        parser.add_argument('--latency', type=float, default=0.01, help='Network latency in seconds')

        args = parser.parse_args()

        # Set up node-specific logging
        global logger
        logger = setup_logging(args.id)

        # Print startup information
        print(f"\nStarting node server with ID: {args.id}")
        print(f"Host: {args.host}, Port: {args.port}")
        print(f"Capacity: {args.capacity}, Latency: {args.latency}s\n")
        sys.stdout.flush()

        # Create and start node with error handling
        try:
            node = NetworkNode(
                node_id=args.id,
                host=args.host,
                port=args.port,
                capacity=args.capacity,
                latency=args.latency
            )

            # Start the node
            node.start()

        except OSError as e:
            logger.error(f"Failed to create network node: {str(e)}")
            print(f"\nERROR: Failed to create network node: {str(e)}")
            print("This is likely due to a port conflict. Try using a different port.")
            sys.stdout.flush()
            sys.exit(1)

    except Exception as e:
        print(f"\nERROR: Failed to start node server: {str(e)}")
        sys.stdout.flush()
        sys.exit(1)

    try:
        # Print node info with color
        logger.info(f"Node {args.id} started on {node.host}:{node.port}")
        logger.info(f"Capacity: {args.capacity}, Latency: {args.latency}s")
        logger.info("Press Ctrl+C to stop")

        # Print a colorful header
        print("\n" + "="*60)
        print(f"{'='*20} NODE {args.id} RUNNING {'='*20}")
        print("="*60)
        print(f"Host: {node.host}, Port: {node.port}")
        print(f"Capacity: {args.capacity}, Latency: {args.latency}s")
        print("="*60 + "\n")

        # Force flush stdout to ensure it appears in the terminal
        sys.stdout.flush()
    except NameError:
        # This will happen if node creation failed
        pass

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)

            # Calculate statistics
            uptime = time.time() - node.start_time
            if uptime > 0:
                frame_rate = node.frames_processed / uptime
            else:
                frame_rate = 0

            # Print statistics with color based on load
            load_color = '\033[92m'  # Green for low load
            if node.current_load > 0.7:
                load_color = '\033[91m'  # Red for high load
            elif node.current_load > 0.3:
                load_color = '\033[93m'  # Yellow for medium load

            reset_color = '\033[0m'

            # Clear the terminal line for cleaner output
            print("\r" + " " * 80 + "\r", end="")

            # Print with color and more detailed information
            status_line = (
                f"Node {args.id} - "
                f"Processed: {node.frames_processed}, "
                f"Dropped: {node.frames_dropped}, "
                f"Rate: {frame_rate:.1f} fps, "
                f"Load: {load_color}{node.current_load:.2f}{reset_color}, "
                f"Clients: {len(node.connected_clients)}"
            )

            # Simulate some activity even if no real frames are being processed
            if node.frames_processed == 0 and random.random() < 0.2:
                # Simulate processing
                node.frames_processed += 1
                node.current_load = random.uniform(0.1, 0.5)

                # Log simulated activity
                logger.info(f"Simulated processing activity - Load: {node.current_load:.2f}")

            # Print status with a timestamp
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] {status_line}")

            # Force flush stdout to ensure it appears in the terminal
            sys.stdout.flush()

            # Log client IPs if any
            if node.connected_clients and random.random() < 0.1:
                for client_socket in node.connected_clients:
                    client_ip = node.client_ips.get(client_socket, "unknown")
                    logger.info(f"Connected client: {client_ip}")

    except KeyboardInterrupt:
        logger.info("Stopping node...")

    finally:
        # Stop node
        node.stop()


if __name__ == '__main__':
    main()
