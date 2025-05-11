"""
Bubble Sort Algorithm Implementation and Visualization.

Bubble Sort is a simple sorting algorithm that repeatedly steps through the list,
compares adjacent elements, and swaps them if they are in the wrong order.
The pass through the list is repeated until the list is sorted.

Time Complexity:
- Best Case: O(n) when the array is already sorted
- Average Case: O(n²)
- Worst Case: O(n²)

Space Complexity: O(1)
"""

from typing import List, Optional
import time
import matplotlib.pyplot as plt
import numpy as np
from algorithms.visualization import ArrayVisualizer

def bubble_sort(arr: List[int], visualize: bool = False) -> List[int]:
    """
    Sort an array using the bubble sort algorithm.
    
    Args:
        arr: The array to sort
        visualize: Whether to visualize the sorting process
    
    Returns:
        The sorted array
    """
    # Create a copy of the array to avoid modifying the original
    array = arr.copy()
    n = len(array)
    
    # Create visualizer if needed
    vis = None
    if visualize:
        vis = ArrayVisualizer("Bubble Sort")
        vis.update(array, text={"Algorithm": "Bubble Sort", 
                               "Time Complexity": "O(n²)",
                               "Space Complexity": "O(1)"})
    
    # Bubble sort algorithm
    for i in range(n):
        # Flag to optimize if no swaps occur in a pass
        swapped = False
        
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Highlight elements being compared
            if visualize:
                highlights = {j: 'red', j + 1: 'red'}
                vis.update(array, highlights, 
                          text={"Pass": i+1, "Comparing": f"{array[j]} and {array[j+1]}"})
            
            # Compare adjacent elements
            if array[j] > array[j + 1]:
                # Swap elements
                array[j], array[j + 1] = array[j + 1], array[j]
                swapped = True
                
                # Visualize the swap
                if visualize:
                    highlights = {j: 'green', j + 1: 'green'}
                    vis.update(array, highlights, 
                              text={"Pass": i+1, "Swapped": f"{array[j+1]} and {array[j]}"})
            
        # If no swapping occurred in this pass, array is sorted
        if not swapped:
            break
    
    # Final visualization
    if visualize:
        vis.update(array, text={"Status": "Sorted!", 
                               "Time Complexity": "O(n²)",
                               "Space Complexity": "O(1)"})
        vis.save_animation("bubble_sort")
        vis.show()
    
    return array

def visualize_bubble_sort_complexity():
    """
    Create a visualization of Bubble Sort's time complexity.
    """
    plt.figure(figsize=(10, 6))
    
    # Data for the plot
    n_values = np.arange(1, 101)
    best_case = n_values  # O(n)
    worst_case = n_values**2  # O(n²)
    average_case = n_values**2  # O(n²)
    
    # Plotting
    plt.plot(n_values, best_case, label='Best Case: O(n)', color='green', linestyle='--')
    plt.plot(n_values, average_case, label='Average Case: O(n²)', color='blue')
    plt.plot(n_values, worst_case, label='Worst Case: O(n²)', color='red')
    
    plt.xlabel('Input Size (n)')
    plt.ylabel('Number of Operations')
    plt.title('Bubble Sort Time Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('visualizations/bubble_sort_complexity.png')
    plt.close()
    
    # Create an educational figure explaining the algorithm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Example array for demonstration
    example = [5, 1, 4, 2, 8]
    
    # First pass visualization
    ax1.bar(range(len(example)), example, color=['blue', 'blue', 'blue', 'blue', 'blue'])
    ax1.set_xticks(range(len(example)))
    ax1.set_xticklabels(range(len(example)))
    ax1.set_title("Bubble Sort: First Pass")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Value")
    
    # Add arrows and explanations
    ax1.annotate("Compare and swap if needed", xy=(0.5, 5.5), xytext=(0.5, 7),
                arrowprops=dict(arrowstyle="->", color='red'))
    ax1.annotate("Move to next pair", xy=(1.5, 4.5), xytext=(1.5, 6),
                arrowprops=dict(arrowstyle="->", color='green'))
    
    # Algorithm explanation
    explanation = """
    Bubble Sort Algorithm:
    
    1. Start at the beginning of the array
    2. Compare adjacent elements, swap if they are in wrong order
    3. Move to the next pair of adjacent elements
    4. After one pass, the largest element is at the end
    5. Repeat for remaining elements (excluding sorted ones)
    6. Continue until no swaps are needed
    
    Key Characteristics:
    - Simple implementation
    - O(n²) time complexity
    - O(1) space complexity
    - Stable sorting algorithm
    - Not suitable for large datasets
    """
    
    ax2.text(0.1, 0.5, explanation, fontsize=10, verticalalignment='center')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/bubble_sort_explanation.png')
    plt.close()

if __name__ == "__main__":
    # Example usage
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original array: {test_array}")
    sorted_array = bubble_sort(test_array, visualize=True)
    print(f"Sorted array: {sorted_array}")
    
    # Create complexity visualization
    visualize_bubble_sort_complexity()
