"""
Breadth-First Search (BFS) Algorithm Implementation and Visualization.

BFS is a graph traversal algorithm that explores all the vertices of a graph at the
present depth before moving on to vertices at the next depth level.

Time Complexity: O(V + E) where V is the number of vertices and E is the number of edges
Space Complexity: O(V) for the queue and visited set
"""

from typing import Dict, List, Set, Optional, Any
import time
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from algorithms.visualization import GraphVisualizer

def bfs(graph: nx.Graph, start_node: int, visualize: bool = False) -> List[int]:
    """
    Perform breadth-first search on a graph.
    
    Args:
        graph: The graph to traverse
        start_node: The node to start the traversal from
        visualize: Whether to visualize the traversal process
    
    Returns:
        A list of nodes in the order they were visited
    """
    # Check if start node is in the graph
    if start_node not in graph.nodes:
        raise ValueError(f"Start node {start_node} not in graph")
    
    # Create visualizer if needed
    vis = None
    if visualize:
        vis = GraphVisualizer("Breadth-First Search (BFS)")
        vis.set_graph(graph)
        
        # Initial visualization
        node_colors = {start_node: 'red'}
        vis.update(node_colors=node_colors, 
                  title_suffix="Start Node")
    
    # BFS algorithm
    visited = []  # List to keep track of visited nodes
    queue = deque([start_node])  # Initialize a queue
    visited_set = set([start_node])  # Set for O(1) lookups
    
    while queue:
        # Dequeue a vertex from queue
        current_node = queue.popleft()
        visited.append(current_node)
        
        # Visualize the current node being processed
        if visualize:
            node_colors = {node: 'green' for node in visited}
            node_colors[current_node] = 'red'
            
            # Highlight the queue
            for node in queue:
                node_colors[node] = 'yellow'
            
            # Highlight edges to neighbors
            edge_colors = {}
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited_set:
                    edge_colors[(current_node, neighbor)] = 'red'
            
            vis.update(node_colors=node_colors, 
                      edge_colors=edge_colors,
                      title_suffix=f"Processing Node {current_node}")
        
        # Get all adjacent vertices of the dequeued vertex
        # If an adjacent vertex has not been visited, mark it
        # visited and enqueue it
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited_set:
                queue.append(neighbor)
                visited_set.add(neighbor)
                
                # Visualize the neighbor being added to the queue
                if visualize:
                    node_colors = {node: 'green' for node in visited}
                    node_colors[current_node] = 'red'
                    node_colors[neighbor] = 'yellow'
                    
                    # Highlight the queue
                    for node in queue:
                        if node != neighbor:
                            node_colors[node] = 'yellow'
                    
                    # Highlight the edge to the neighbor
                    edge_colors = {(current_node, neighbor): 'red'}
                    
                    vis.update(node_colors=node_colors, 
                              edge_colors=edge_colors,
                              title_suffix=f"Adding {neighbor} to Queue")
    
    # Final visualization
    if visualize:
        node_colors = {node: 'green' for node in visited}
        vis.update(node_colors=node_colors, 
                  title_suffix="Traversal Complete")
        vis.save_animation("bfs")
        vis.show()
    
    return visited

def visualize_bfs_complexity():
    """
    Create a visualization of BFS's time and space complexity.
    """
    plt.figure(figsize=(10, 6))
    
    # Data for the plot
    v_values = np.arange(1, 101)  # Number of vertices
    e_values = v_values * 2  # Assuming average degree of 2 (sparse graph)
    
    # Time complexity: O(V + E)
    time_complexity = v_values + e_values
    
    # Space complexity: O(V)
    space_complexity = v_values
    
    # Plotting
    plt.plot(v_values, time_complexity, label='Time Complexity: O(V + E)', color='blue')
    plt.plot(v_values, space_complexity, label='Space Complexity: O(V)', color='green')
    
    plt.xlabel('Number of Vertices (V)')
    plt.ylabel('Complexity')
    plt.title('BFS Time and Space Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('visualizations/bfs_complexity.png')
    plt.close()
    
    # Create an educational figure explaining the algorithm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create a sample graph for demonstration
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7), (4, 8), (5, 9), (6, 10)
    ])
    
    # Position nodes in a tree-like layout
    pos = {
        1: (0.5, 1.0),
        2: (0.3, 0.8),
        3: (0.7, 0.8),
        4: (0.2, 0.6),
        5: (0.4, 0.6),
        6: (0.6, 0.6),
        7: (0.8, 0.6),
        8: (0.1, 0.4),
        9: (0.3, 0.4),
        10: (0.5, 0.4)
    }
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', ax=ax1)
    nx.draw_networkx_edges(G, pos, ax=ax1)
    nx.draw_networkx_labels(G, pos, ax=ax1)
    
    # Show BFS traversal order
    bfs_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i, node in enumerate(bfs_order):
        ax1.annotate(f"{i+1}", xy=pos[node], xytext=(pos[node][0], pos[node][1] - 0.05),
                    ha='center', va='center', bbox=dict(boxstyle="circle", fc="white", ec="red"))
    
    ax1.set_title("BFS Traversal Order")
    ax1.axis('off')
    
    # Algorithm explanation
    explanation = """
    Breadth-First Search (BFS) Algorithm:
    
    1. Start at a given node (the "source" node)
    2. Explore all neighbor nodes at the present depth before moving on to nodes at the next depth level
    3. Use a queue data structure to keep track of nodes to visit next
    4. Mark nodes as visited to avoid cycles
    
    Implementation Steps:
    1. Create a queue and enqueue the starting node
    2. Mark the starting node as visited
    3. While the queue is not empty:
       a. Dequeue a node from the queue
       b. Process the node (e.g., add to result list)
       c. Enqueue all unvisited neighbors and mark them as visited
    
    Key Characteristics:
    - Time Complexity: O(V + E) where V is the number of vertices and E is the number of edges
    - Space Complexity: O(V) for the queue and visited set
    - Finds shortest paths in unweighted graphs
    - Explores nodes level by level
    
    Applications:
    - Finding shortest path in unweighted graphs
    - Web crawlers
    - Social networking (finding friends within a certain degree of connection)
    - Garbage collection (mark and sweep algorithm)
    - Finding connected components
    """
    
    ax2.text(0.1, 0.5, explanation, fontsize=10, verticalalignment='center')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/bfs_explanation.png')
    plt.close()

if __name__ == "__main__":
    # Create a sample graph
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (4, 7), (5, 7), (6, 7)
    ])
    
    # Run BFS
    start_node = 0
    print(f"Running BFS starting from node {start_node}")
    traversal_order = bfs(G, start_node, visualize=True)
    print(f"BFS traversal order: {traversal_order}")
    
    # Create complexity visualization
    visualize_bfs_complexity()
