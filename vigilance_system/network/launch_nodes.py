"""
Launch multiple network nodes for distributed video processing.

This script launches multiple network node servers in separate terminals
to simulate a distributed video processing network.
"""

import os
import sys
import time
import random
import subprocess
import argparse
import json
from typing import List, Dict, Any

# Get the absolute path to the vigilance_system directory
VIGILANCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def launch_node(node_id: str, host: str = 'localhost', port: int = 0,
                capacity: float = 1.0, latency: float = 0.01) -> subprocess.Popen:
    """
    Launch a network node in a new terminal.

    Args:
        node_id: Unique identifier for the node
        host: Host to bind the server to
        port: Port to bind the server to (0 for auto-assign)
        capacity: Processing capacity of the node
        latency: Simulated network latency in seconds

    Returns:
        subprocess.Popen: Process object for the launched node
    """
    # Construct command
    python_exe = sys.executable
    script_path = os.path.join(VIGILANCE_DIR, 'vigilance_system', 'network', 'node_server.py')

    # Create command
    cmd = [
        python_exe,
        script_path,
        '--id', node_id,
        '--host', host,
        '--port', str(port),
        '--capacity', str(capacity),
        '--latency', str(latency)
    ]

    # Launch node in a visible terminal window
    try:
        # Create a log file for the node
        log_file = os.path.join(os.getcwd(), f"node_{node_id}.log")

        # Open log file for writing
        log_handle = open(log_file, 'w')

        # Create a command that opens a new terminal window
        terminal_cmd = [
            'start',
            'cmd',
            '/k',
            f"title Node {node_id} - Port {port} && {' '.join(cmd)}"
        ]

        # Launch the process in a visible terminal
        process = subprocess.Popen(
            ' '.join(terminal_cmd),
            shell=True,
            stdout=log_handle,
            stderr=log_handle
        )

        # Print information about the node
        print(f"Launched node {node_id} (PID: {process.pid})")
        print(f"  Host: {host}, Port: {port}")
        print(f"  Capacity: {capacity}, Latency: {latency}s")
        print(f"  Log file: {log_file}")

    except Exception as e:
        print(f"Error launching node: {str(e)}")
        # Create a dummy process object
        process = None

    print(f"Launched node {node_id} with capacity={capacity}, latency={latency}")

    return process


def launch_nodes(num_nodes: int, base_port: int = 8000) -> List[Dict[str, Any]]:
    """
    Launch multiple network nodes.

    Args:
        num_nodes: Number of nodes to launch
        base_port: Base port number for the nodes

    Returns:
        List[Dict[str, Any]]: List of node information dictionaries
    """
    nodes = []

    for i in range(num_nodes):
        # Generate node ID
        node_id = f"node_{i+1}"

        # Generate random capacity and latency
        capacity = 0.5 + random.random()  # 0.5 to 1.5
        latency = 0.005 + random.random() * 0.045  # 5ms to 50ms

        # Calculate port
        port = base_port + i

        # Launch node
        process = launch_node(
            node_id=node_id,
            host='localhost',
            port=port,
            capacity=capacity,
            latency=latency
        )

        # Store node info
        node_info = {
            'node_id': node_id,
            'host': 'localhost',
            'port': port,
            'capacity': capacity,
            'latency': latency,
            'process': process
        }

        nodes.append(node_info)

        # Wait a bit between launches
        time.sleep(1)

    return nodes


def save_node_info(nodes: List[Dict[str, Any]], output_file: str = 'nodes.json'):
    """
    Save node information to a JSON file.

    Args:
        nodes: List of node information dictionaries
        output_file: Path to the output file
    """
    # Remove process objects (not serializable)
    serializable_nodes = []

    for node in nodes:
        node_copy = node.copy()
        if 'process' in node_copy:
            del node_copy['process']
        serializable_nodes.append(node_copy)

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(serializable_nodes, f, indent=2)

    print(f"Saved node information to {output_file}")


def main():
    """Main function to launch network nodes."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Launch Network Nodes')
    parser.add_argument('--nodes', type=int, default=8, help='Number of nodes to launch')
    parser.add_argument('--base-port', type=int, default=8000, help='Base port number')
    parser.add_argument('--output', type=str, default='nodes.json', help='Output file for node information')
    parser.add_argument('--visible', action='store_true', help='Launch nodes in visible terminals')

    args = parser.parse_args()

    # Launch nodes
    print(f"Launching {args.nodes} network nodes...")

    # Always launch each node in its own visible terminal for better debugging
    nodes = []
    for i in range(args.nodes):
        # Generate node ID
        node_id = f"node_{i+1}"

        # Generate random capacity and latency
        capacity = 0.5 + random.random()  # 0.5 to 1.5
        latency = 0.005 + random.random() * 0.045  # 5ms to 50ms

        # Calculate port - use a larger offset to avoid conflicts
        port = args.base_port + (i * 10)  # Use bigger gaps between ports

        # Create a command that opens a new terminal window
        node_script = os.path.join(os.path.dirname(__file__), 'node_server.py')

        # Get absolute paths to ensure they work in any context
        python_exe = sys.executable
        node_script_abs = os.path.abspath(node_script)

        # Create the command with proper quoting
        cmd_str = f'"{python_exe}" "{node_script_abs}" --id {node_id} --host localhost --port {port} --capacity {capacity} --latency {latency}'

        # Add a small delay to ensure previous process has released any resources
        time.sleep(0.5)

        # Create the full terminal command
        terminal_cmd = f'start "Node {node_id} - Port {port}" cmd /k "{cmd_str}"'

        print(f"Launching node {node_id} with command: {cmd_str}")

        # Launch the process in a visible terminal
        process = subprocess.Popen(terminal_cmd, shell=True)

        # Store node info
        node_info = {
            'node_id': node_id,
            'host': 'localhost',
            'port': port,
            'capacity': capacity,
            'latency': latency,
            'process': process
        }

        nodes.append(node_info)

        # Wait a bit between launches to avoid overwhelming the system
        time.sleep(0.5)

    # Save node information
    save_node_info(nodes, args.output)

    print(f"Launched {len(nodes)} nodes. Press Ctrl+C to stop.")
    print("Each node is running in its own terminal window.")
    print("You can see the node status in each terminal.")

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping nodes...")

    finally:
        # Stop nodes
        for node in nodes:
            if 'process' in node:
                try:
                    node['process'].terminate()
                except:
                    pass


if __name__ == '__main__':
    main()
