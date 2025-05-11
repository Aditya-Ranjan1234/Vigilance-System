"""
Depth-First Search (DFS) Algorithm Implementation and Visualization.

DFS is a graph traversal algorithm that explores as far as possible along each branch
before backtracking. It uses a stack data structure (or recursion) to keep track of nodes.

Time Complexity: O(V + E) where V is the number of vertices and E is the number of edges
Space Complexity: O(V) for the stack and visited set
"""

from typing import Dict, List, Set, Optional, Any
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from algorithms.visualization import GraphVisualizer

def dfs(graph: nx.Graph, start_node: int, visualize: bool = False) -> List[int]:
    """
    Perform depth-first search on a graph.
    
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
        vis = GraphVisualizer("Depth-First Search (DFS)")
        vis.set_graph(graph)
        
        # Initial visualization
        node_colors = {start_node: 'red'}
        vis.update(node_colors=node_colors, 
                  title_suffix="Start Node")
    
    # DFS algorithm
    visited = []  # List to keep track of visited nodes
    stack = [start_node]  # Initialize a stack
    visited_set = set()  # Set for O(1) lookups
    
    while stack:
        # Pop a vertex from stack
        current_node = stack.pop()
        
        # Skip if already visited
        if current_node in visited_set:
            continue
        
        # Mark as visited
        visited.append(current_node)
        visited_set.add(current_node)
        
        # Visualize the current node being processed
        if visualize:
            node_colors = {node: 'green' for node in visited}
            node_colors[current_node] = 'red'
            
            # Highlight the stack
            for i, node in enumerate(reversed(stack)):
                node_colors[node] = 'yellow'
            
            # Highlight edges to neighbors
            edge_colors = {}
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited_set:
                    edge_colors[(current_node, neighbor)] = 'red'
            
            vis.update(node_colors=node_colors, 
                      edge_colors=edge_colors,
                      title_suffix=f"Processing Node {current_node}")
        
        # Get all adjacent vertices of the popped vertex
        # If an adjacent vertex has not been visited, push it to the stack
        # We use reversed to maintain the same order as recursive DFS
        neighbors = list(graph.neighbors(current_node))
        for neighbor in reversed(neighbors):
            if neighbor not in visited_set:
                stack.append(neighbor)
                
                # Visualize the neighbor being added to the stack
                if visualize:
                    node_colors = {node: 'green' for node in visited}
                    node_colors[current_node] = 'red'
                    node_colors[neighbor] = 'yellow'
                    
                    # Highlight the stack
                    for i, node in enumerate(reversed(stack)):
                        if node != neighbor:
                            node_colors[node] = 'yellow'
                    
                    # Highlight the edge to the neighbor
                    edge_colors = {(current_node, neighbor): 'red'}
                    
                    vis.update(node_colors=node_colors, 
                              edge_colors=edge_colors,
                              title_suffix=f"Adding {neighbor} to Stack")
    
    # Final visualization
    if visualize:
        node_colors = {node: 'green' for node in visited}
        vis.update(node_colors=node_colors, 
                  title_suffix="Traversal Complete")
        vis.save_animation("dfs")
        vis.show()
    
    return visited

def dfs_recursive(graph: nx.Graph, start_node: int, visualize: bool = False) -> List[int]:
    """
    Perform depth-first search on a graph using recursion.
    
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
        vis = GraphVisualizer("Recursive Depth-First Search (DFS)")
        vis.set_graph(graph)
        
        # Initial visualization
        node_colors = {start_node: 'red'}
        vis.update(node_colors=node_colors, 
                  title_suffix="Start Node")
    
    # Initialize visited list and set
    visited = []
    visited_set = set()
    
    # Recursive DFS function
    def dfs_util(node, visited, visited_set):
        # Mark the current node as visited
        visited.append(node)
        visited_set.add(node)
        
        # Visualize the current node being processed
        if visualize:
            node_colors = {n: 'green' for n in visited}
            node_colors[node] = 'red'
            
            # Highlight edges to neighbors
            edge_colors = {}
            for neighbor in graph.neighbors(node):
                if neighbor not in visited_set:
                    edge_colors[(node, neighbor)] = 'red'
            
            vis.update(node_colors=node_colors, 
                      edge_colors=edge_colors,
                      title_suffix=f"Processing Node {node}")
        
        # Recur for all the adjacent vertices
        for neighbor in graph.neighbors(node):
            if neighbor not in visited_set:
                # Visualize the neighbor before recursion
                if visualize:
                    node_colors = {n: 'green' for n in visited}
                    node_colors[node] = 'red'
                    node_colors[neighbor] = 'yellow'
                    
                    # Highlight the edge to the neighbor
                    edge_colors = {(node, neighbor): 'red'}
                    
                    vis.update(node_colors=node_colors, 
                              edge_colors=edge_colors,
                              title_suffix=f"Exploring {neighbor}")
                
                # Recursive call
                dfs_util(neighbor, visited, visited_set)
    
    # Call the recursive function
    dfs_util(start_node, visited, visited_set)
    
    # Final visualization
    if visualize:
        node_colors = {node: 'green' for node in visited}
        vis.update(node_colors=node_colors, 
                  title_suffix="Traversal Complete")
        vis.save_animation("dfs_recursive")
        vis.show()
    
    return visited

def visualize_dfs_complexity():
    """
    Create a visualization of DFS's time and space complexity.
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
    plt.title('DFS Time and Space Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('visualizations/dfs_complexity.png')
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
    
    # Show DFS traversal order
    dfs_order = [1, 2, 4, 8, 5, 9, 3, 6, 10, 7]
    for i, node in enumerate(dfs_order):
        ax1.annotate(f"{i+1}", xy=pos[node], xytext=(pos[node][0], pos[node][1] - 0.05),
                    ha='center', va='center', bbox=dict(boxstyle="circle", fc="white", ec="red"))
    
    ax1.set_title("DFS Traversal Order")
    ax1.axis('off')
    
    # Algorithm explanation
    explanation = """
    Depth-First Search (DFS) Algorithm:
    
    1. Start at a given node (the "source" node)
    2. Explore as far as possible along each branch before backtracking
    3. Use a stack data structure (or recursion) to keep track of nodes to visit next
    4. Mark nodes as visited to avoid cycles
    
    Implementation Steps:
    1. Create a stack and push the starting node
    2. While the stack is not empty:
       a. Pop a node from the stack
       b. If the node has not been visited:
          i. Mark it as visited
          ii. Push all its unvisited neighbors to the stack
    
    Recursive Implementation:
    1. Mark the current node as visited
    2. Recursively visit all unvisited neighbors
    
    Key Characteristics:
    - Time Complexity: O(V + E) where V is the number of vertices and E is the number of edges
    - Space Complexity: O(V) for the stack and visited set
    - Goes deep into the graph before exploring siblings
    - Can be implemented using recursion or an explicit stack
    
    Applications:
    - Topological sorting
    - Finding connected components
    - Maze generation and solving
    - Cycle detection
    - Path finding
    """
    
    ax2.text(0.1, 0.5, explanation, fontsize=10, verticalalignment='center')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/dfs_explanation.png')
    plt.close()

if __name__ == "__main__":
    # Create a sample graph
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (4, 7), (5, 7), (6, 7)
    ])
    
    # Run DFS
    start_node = 0
    print(f"Running DFS starting from node {start_node}")
    traversal_order = dfs(G, start_node, visualize=True)
    print(f"DFS traversal order: {traversal_order}")
    
    # Run recursive DFS
    print(f"Running recursive DFS starting from node {start_node}")
    traversal_order_recursive = dfs_recursive(G, start_node, visualize=True)
    print(f"Recursive DFS traversal order: {traversal_order_recursive}")
    
    # Create complexity visualization
    visualize_dfs_complexity()
