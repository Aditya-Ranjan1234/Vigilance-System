"""
Quick Sort Algorithm Implementation and Visualization.

Quick Sort is a divide-and-conquer algorithm that picks an element as a pivot
and partitions the array around the pivot. There are different versions of quicksort
that pick pivot in different ways:
1. First element as pivot
2. Last element as pivot
3. Random element as pivot
4. Median as pivot

Time Complexity:
- Best Case: O(n log n)
- Average Case: O(n log n)
- Worst Case: O(n²) when the array is already sorted and pivot is always the smallest/largest element

Space Complexity: O(log n) due to the recursive call stack
"""

from typing import List, Optional, Dict, Any, Tuple
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from algorithms.visualization import ArrayVisualizer

def quick_sort(arr: List[int], visualize: bool = False) -> List[int]:
    """
    Sort an array using the quick sort algorithm.
    
    Args:
        arr: The array to sort
        visualize: Whether to visualize the sorting process
    
    Returns:
        The sorted array
    """
    # Create a copy of the array to avoid modifying the original
    array = arr.copy()
    
    # Create visualizer if needed
    vis = None
    if visualize:
        vis = ArrayVisualizer("Quick Sort")
        vis.update(array, text={"Algorithm": "Quick Sort", 
                               "Time Complexity": "O(n log n) average",
                               "Space Complexity": "O(log n)"})
    
    # Helper function to partition the array
    def partition(arr: List[int], low: int, high: int) -> int:
        # Choose the rightmost element as pivot
        pivot = arr[high]
        
        # Visualize the pivot selection
        if visualize:
            highlights = {high: 'red'}
            vis.update(array, highlights, 
                      text={"Partitioning": f"Indices {low} to {high}",
                           "Pivot": pivot})
        
        # Index of smaller element
        i = low - 1
        
        # Traverse through all elements
        # compare each element with pivot
        for j in range(low, high):
            # Visualize the comparison
            if visualize:
                highlights = {j: 'yellow', high: 'red'}
                if i >= low:
                    highlights[i] = 'blue'
                vis.update(array, highlights, 
                          text={"Comparing": f"{arr[j]} with pivot {pivot}",
                               "Current i": i})
            
            # If current element is smaller than or equal to pivot
            if arr[j] <= pivot:
                # Increment index of smaller element
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                
                # Visualize the swap
                if visualize and i != j:
                    highlights = {i: 'green', j: 'green', high: 'red'}
                    vis.update(array, highlights, 
                              text={"Swapped": f"{arr[i]} and {arr[j]}",
                                   "Current i": i})
        
        # Swap the pivot element with the element at i+1
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        
        # Visualize the final pivot placement
        if visualize:
            highlights = {i + 1: 'purple', high: 'green'}
            vis.update(array, highlights, 
                      text={"Pivot Placed": f"Pivot {pivot} placed at index {i+1}"})
        
        return i + 1
    
    # Helper function to implement quick sort recursively
    def quick_sort_recursive(arr: List[int], low: int, high: int) -> None:
        if low < high:
            # Visualize the current subarray
            if visualize:
                highlights = {i: 'cyan' for i in range(low, high + 1)}
                vis.update(array, highlights, 
                          text={"Sorting": f"Subarray from index {low} to {high}"})
            
            # Partition the array and get the pivot index
            pi = partition(arr, low, high)
            
            # Recursively sort elements before and after partition
            quick_sort_recursive(arr, low, pi - 1)
            quick_sort_recursive(arr, pi + 1, high)
    
    # Start the quick sort
    quick_sort_recursive(array, 0, len(array) - 1)
    
    # Final visualization
    if visualize:
        vis.update(array, text={"Status": "Sorted!", 
                               "Time Complexity": "O(n log n) average",
                               "Space Complexity": "O(log n)"})
        vis.save_animation("quick_sort")
        vis.show()
    
    return array

def visualize_quick_sort_complexity():
    """
    Create a visualization of Quick Sort's time complexity.
    """
    plt.figure(figsize=(10, 6))
    
    # Data for the plot
    n_values = np.arange(1, 101)
    best_case = n_values * np.log2(n_values)  # O(n log n)
    average_case = n_values * np.log2(n_values)  # O(n log n)
    worst_case = n_values**2  # O(n²)
    
    # For comparison, show O(n)
    n_linear = n_values
    
    # Plotting
    plt.plot(n_values, best_case, label='Best Case: O(n log n)', color='green')
    plt.plot(n_values, average_case, label='Average Case: O(n log n)', color='blue')
    plt.plot(n_values, worst_case, label='Worst Case: O(n²)', color='red')
    plt.plot(n_values, n_linear, label='O(n)', color='purple', linestyle='--', alpha=0.5)
    
    plt.xlabel('Input Size (n)')
    plt.ylabel('Number of Operations')
    plt.title('Quick Sort Time Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('visualizations/quick_sort_complexity.png')
    plt.close()
    
    # Create an educational figure explaining the algorithm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Example of quick sort partitioning
    example = [7, 2, 1, 6, 8, 5, 3, 4]
    pivot = example[-1]  # 4
    
    # Visualize the partitioning process
    ax1.set_title("Quick Sort: Partitioning Process")
    
    # Original array with pivot highlighted
    bars = ax1.bar(range(len(example)), example, color=['blue'] * (len(example) - 1) + ['red'])
    
    # Add annotations
    ax1.text(len(example) - 1, pivot + 0.5, "Pivot", ha='center', color='red')
    
    # Show the partitioning result
    partitioned = [2, 1, 3, 4, 8, 5, 6, 7]  # After partitioning
    ax1.plot(range(len(partitioned)), partitioned, 'go--', alpha=0.7, label='After partitioning')
    
    # Add a line to show elements less than pivot and greater than pivot
    ax1.axhline(y=pivot, color='r', linestyle='--', alpha=0.3)
    ax1.text(len(example) / 2, pivot + 0.2, "Elements < pivot", ha='center', color='green')
    ax1.text(len(example) / 2, pivot - 0.2, "Elements > pivot", ha='center', color='red')
    
    ax1.set_xticks(range(len(example)))
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Value")
    ax1.legend()
    
    # Algorithm explanation
    explanation = """
    Quick Sort Algorithm:
    
    1. Choose a pivot element from the array
    2. Partition the array around the pivot:
       - Elements less than pivot go to the left
       - Elements greater than pivot go to the right
       - Pivot is placed in its final sorted position
    3. Recursively apply the above steps to the sub-arrays
    
    Partitioning Process:
    - Start from the leftmost element
    - Keep track of index of smaller elements as i
    - If current element is smaller than pivot, swap it with element at i
    - After processing all elements, swap pivot with element at i+1
    - Return the position of the pivot
    
    Key Characteristics:
    - Efficient for large datasets
    - O(n log n) average time complexity
    - O(n²) worst-case time complexity
    - O(log n) space complexity
    - Not stable by default
    - In-place sorting algorithm
    - Quicksort is often faster in practice than other O(n log n) algorithms
    """
    
    ax2.text(0.1, 0.5, explanation, fontsize=10, verticalalignment='center')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/quick_sort_explanation.png')
    plt.close()

if __name__ == "__main__":
    # Example usage
    test_array = [10, 7, 8, 9, 1, 5]
    print(f"Original array: {test_array}")
    sorted_array = quick_sort(test_array, visualize=True)
    print(f"Sorted array: {sorted_array}")
    
    # Create complexity visualization
    visualize_quick_sort_complexity()
