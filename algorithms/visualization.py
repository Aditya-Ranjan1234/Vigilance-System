"""
Visualization utilities for algorithm demonstrations.

This module provides helper functions for visualizing algorithm execution.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Callable, Union
import time
import os

# Create a directory for saving visualizations
os.makedirs('visualizations', exist_ok=True)

class ArrayVisualizer:
    """Visualizer for array-based algorithms like sorting and searching."""
    
    def __init__(self, title: str = "Algorithm Visualization"):
        """
        Initialize the array visualizer.
        
        Args:
            title: Title for the visualization
        """
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.bar_container = None
        self.frames = []
        self.text_objects = {}
        
    def update(self, array: List[int], highlights: Dict[int, str] = None, 
               text: Dict[str, Any] = None, clear: bool = True) -> None:
        """
        Update the visualization with a new state.
        
        Args:
            array: Current state of the array
            highlights: Dictionary mapping indices to highlight colors
            text: Dictionary of text annotations to display
            clear: Whether to clear previous text annotations
        """
        if clear and self.text_objects:
            for text_obj in self.text_objects.values():
                text_obj.remove()
            self.text_objects = {}
            
        self.ax.clear()
        bars = self.ax.bar(range(len(array)), array, align='center', alpha=0.7)
        
        # Apply highlights
        if highlights:
            for idx, color in highlights.items():
                if 0 <= idx < len(bars):
                    bars[idx].set_color(color)
        
        # Add text annotations
        if text:
            y_pos = max(array) * 1.1 if array else 1
            x_offset = 0
            
            for key, value in text.items():
                text_obj = self.ax.text(len(array) // 2 + x_offset, y_pos, 
                                       f"{key}: {value}", ha='center')
                self.text_objects[key] = text_obj
                x_offset += len(array) // 4
        
        self.ax.set_title(self.title)
        self.ax.set_xlabel('Index')
        self.ax.set_ylabel('Value')
        self.ax.set_xticks(range(len(array)))
        
        # Capture the current state for animation
        self.fig.canvas.draw()
        frame = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.frames.append(frame)
    
    def save_animation(self, filename: str, fps: int = 2) -> None:
        """
        Save the animation to a file.
        
        Args:
            filename: Name of the file to save (without extension)
            fps: Frames per second for the animation
        """
        if not self.frames:
            print("No frames to save")
            return
            
        path = os.path.join('visualizations', f"{filename}.gif")
        
        # Create animation
        ani = animation.ArtistAnimation(self.fig, 
                                       [[plt.imshow(frame)] for frame in self.frames], 
                                       interval=1000//fps, blit=True)
        
        # Save animation
        ani.save(path, writer='pillow', fps=fps)
        print(f"Animation saved to {path}")
    
    def show(self) -> None:
        """Display the current visualization."""
        plt.tight_layout()
        plt.show()


class GraphVisualizer:
    """Visualizer for graph-based algorithms."""
    
    def __init__(self, title: str = "Graph Algorithm Visualization"):
        """
        Initialize the graph visualizer.
        
        Args:
            title: Title for the visualization
        """
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.frames = []
        self.pos = None
        self.G = None
        
    def set_graph(self, G: nx.Graph, pos: Dict = None) -> None:
        """
        Set the graph to visualize.
        
        Args:
            G: NetworkX graph
            pos: Optional positions for the nodes
        """
        self.G = G
        if pos is None:
            self.pos = nx.spring_layout(G, seed=42)
        else:
            self.pos = pos
    
    def update(self, node_colors: Dict[int, str] = None, 
               edge_colors: Dict[Tuple[int, int], str] = None,
               node_labels: Dict[int, str] = None,
               edge_labels: Dict[Tuple[int, int], str] = None,
               title_suffix: str = "") -> None:
        """
        Update the visualization with a new state.
        
        Args:
            node_colors: Dictionary mapping nodes to colors
            edge_colors: Dictionary mapping edges to colors
            node_labels: Dictionary mapping nodes to labels
            edge_labels: Dictionary mapping edges to labels
            title_suffix: Additional text to add to the title
        """
        if self.G is None:
            print("No graph set. Call set_graph() first.")
            return
            
        self.ax.clear()
        
        # Default colors
        default_node_color = '#1f78b4'
        default_edge_color = '#888888'
        
        # Prepare node colors
        node_color_map = [node_colors.get(node, default_node_color) if node_colors else default_node_color 
                         for node in self.G.nodes()]
        
        # Prepare edge colors
        edge_color_map = []
        for u, v in self.G.edges():
            if edge_colors and (u, v) in edge_colors:
                edge_color_map.append(edge_colors[(u, v)])
            elif edge_colors and (v, u) in edge_colors:
                edge_color_map.append(edge_colors[(v, u)])
            else:
                edge_color_map.append(default_edge_color)
        
        # Draw the graph
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_color_map, 
                              node_size=500, alpha=0.8, ax=self.ax)
        
        nx.draw_networkx_edges(self.G, self.pos, edge_color=edge_color_map, 
                              width=2, alpha=0.7, ax=self.ax)
        
        # Draw labels
        if node_labels:
            nx.draw_networkx_labels(self.G, self.pos, labels=node_labels, font_size=10, 
                                   font_color='white', ax=self.ax)
        else:
            nx.draw_networkx_labels(self.G, self.pos, font_size=10, font_color='white', ax=self.ax)
        
        if edge_labels:
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, 
                                        font_size=8, ax=self.ax)
        
        # Set title
        full_title = self.title
        if title_suffix:
            full_title += f" - {title_suffix}"
        self.ax.set_title(full_title)
        
        # Remove axis
        self.ax.axis('off')
        
        # Capture the current state for animation
        self.fig.canvas.draw()
        frame = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.frames.append(frame)
    
    def save_animation(self, filename: str, fps: int = 2) -> None:
        """
        Save the animation to a file.
        
        Args:
            filename: Name of the file to save (without extension)
            fps: Frames per second for the animation
        """
        if not self.frames:
            print("No frames to save")
            return
            
        path = os.path.join('visualizations', f"{filename}.gif")
        
        # Create animation
        ani = animation.ArtistAnimation(self.fig, 
                                       [[plt.imshow(frame)] for frame in self.frames], 
                                       interval=1000//fps, blit=True)
        
        # Save animation
        ani.save(path, writer='pillow', fps=fps)
        print(f"Animation saved to {path}")
    
    def show(self) -> None:
        """Display the current visualization."""
        plt.tight_layout()
        plt.show()
