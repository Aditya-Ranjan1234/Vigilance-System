"""
Insertion Sort Algorithm Implementation and Visualization.

Insertion sort is a simple sorting algorithm that builds the final sorted array
one item at a time. It is efficient for small data sets and is often used as
part of more sophisticated algorithms.

Time Complexity:
- Best Case: O(n) when the array is already sorted
- Average Case: O(n²)
- Worst Case: O(n²) when the array is sorted in reverse order

Space Complexity: O(1)
"""

from typing import List, Optional
import time
import matplotlib.pyplot as plt
import numpy as np
from algorithms.visualization import ArrayVisualizer

def insertion_sort(arr: List[int], visualize: bool = False) -> List[int]:
    """
    Sort an array using the insertion sort algorithm.
    
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
        vis = ArrayVisualizer("Insertion Sort")
        vis.update(array, text={"Algorithm": "Insertion Sort", 
                               "Time Complexity": "O(n²)",
                               "Space Complexity": "O(1)"})
    
    # Insertion sort algorithm
    for i in range(1, n):
        key = array[i]
        j = i - 1
        
        # Highlight the current element being inserted
        if visualize:
            highlights = {i: 'red'}
            vis.update(array, highlights, 
                      text={"Pass": i, "Current Element": key})
        
        # Move elements greater than key one position ahead
        while j >= 0 and array[j] > key:
            array[j + 1] = array[j]
            
            # Visualize the shift
            if visualize:
                highlights = {j: 'blue', j + 1: 'green'}
                vis.update(array, highlights, 
                          text={"Pass": i, "Shifting": f"{array[j]} to position {j+1}"})
            
            j -= 1
        
        # Place the key in its correct position
        array[j + 1] = key
        
        # Visualize the insertion
        if visualize:
            highlights = {j + 1: 'green'}
            vis.update(array, highlights, 
                      text={"Pass": i, "Inserted": f"{key} at position {j+1}"})
    
    # Final visualization
    if visualize:
        vis.update(array, text={"Status": "Sorted!", 
                               "Time Complexity": "O(n²)",
                               "Space Complexity": "O(1)"})
        vis.save_animation("insertion_sort")
        vis.show()
    
    return array

def visualize_insertion_sort_complexity():
    """
    Create a visualization of Insertion Sort's time complexity.
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
    plt.title('Insertion Sort Time Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('visualizations/insertion_sort_complexity.png')
    plt.close()
    
    # Create an educational figure explaining the algorithm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Example array for demonstration
    example = [5, 1, 4, 2, 8]
    
    # Visualization of insertion process
    ax1.bar(range(len(example)), example, color=['green', 'red', 'blue', 'blue', 'blue'])
    ax1.set_xticks(range(len(example)))
    ax1.set_xticklabels(range(len(example)))
    ax1.set_title("Insertion Sort: Process")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Value")
    
    # Add arrows and explanations
    ax1.annotate("Sorted portion", xy=(0.5, 5.5), xytext=(0.5, 7),
                arrowprops=dict(arrowstyle="->", color='green'))
    ax1.annotate("Current element\nto insert", xy=(1, 1), xytext=(1, -1),
                arrowprops=dict(arrowstyle="->", color='red'))
    ax1.annotate("Unsorted portion", xy=(3, 4), xytext=(3, 6),
                arrowprops=dict(arrowstyle="->", color='blue'))
    
    # Algorithm explanation
    explanation = """
    Insertion Sort Algorithm:
    
    1. Start with the second element (index 1)
    2. Compare it with elements in the sorted portion
    3. Shift elements greater than the key to the right
    4. Insert the key in its correct position
    5. Move to the next unsorted element
    6. Repeat until the entire array is sorted
    
    Key Characteristics:
    - Efficient for small datasets
    - O(n²) time complexity
    - O(1) space complexity
    - Stable sorting algorithm
    - Adaptive: O(n) when nearly sorted
    - Works well for nearly sorted data
    """
    
    ax2.text(0.1, 0.5, explanation, fontsize=10, verticalalignment='center')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/insertion_sort_explanation.png')
    plt.close()

if __name__ == "__main__":
    # Example usage
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original array: {test_array}")
    sorted_array = insertion_sort(test_array, visualize=True)
    print(f"Sorted array: {sorted_array}")
    
    # Create complexity visualization
    visualize_insertion_sort_complexity()
