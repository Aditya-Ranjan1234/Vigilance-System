"""
Merge Sort Algorithm Implementation and Visualization.

Merge Sort is a divide-and-conquer algorithm that divides the input array into two halves,
recursively sorts them, and then merges the sorted halves.

Time Complexity:
- Best Case: O(n log n)
- Average Case: O(n log n)
- Worst Case: O(n log n)

Space Complexity: O(n)
"""

from typing import List, Optional, Dict, Any
import time
import matplotlib.pyplot as plt
import numpy as np
from algorithms.visualization import ArrayVisualizer

def merge_sort(arr: List[int], visualize: bool = False) -> List[int]:
    """
    Sort an array using the merge sort algorithm.
    
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
        vis = ArrayVisualizer("Merge Sort")
        vis.update(array, text={"Algorithm": "Merge Sort", 
                               "Time Complexity": "O(n log n)",
                               "Space Complexity": "O(n)"})
    
    # Helper function to merge two sorted subarrays
    def merge(arr: List[int], left: int, mid: int, right: int) -> None:
        # Create temporary arrays
        L = arr[left:mid+1]
        R = arr[mid+1:right+1]
        
        # Visualize the subarrays
        if visualize:
            highlights = {i: 'yellow' for i in range(left, mid+1)}
            highlights.update({i: 'cyan' for i in range(mid+1, right+1)})
            vis.update(array, highlights, 
                      text={"Merging": f"Indices {left} to {right}",
                           "Left subarray": str(L),
                           "Right subarray": str(R)})
        
        # Merge the temporary arrays back into arr[left..right]
        i = j = 0
        k = left
        
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            
            # Visualize the merge step
            if visualize:
                highlights = {k: 'green'}
                vis.update(array, highlights, 
                          text={"Merging": f"Placing element at index {k}"})
            
            k += 1
        
        # Copy the remaining elements of L[], if any
        while i < len(L):
            arr[k] = L[i]
            
            # Visualize
            if visualize:
                highlights = {k: 'green'}
                vis.update(array, highlights, 
                          text={"Merging": f"Copying remaining left elements"})
            
            i += 1
            k += 1
        
        # Copy the remaining elements of R[], if any
        while j < len(R):
            arr[k] = R[j]
            
            # Visualize
            if visualize:
                highlights = {k: 'green'}
                vis.update(array, highlights, 
                          text={"Merging": f"Copying remaining right elements"})
            
            j += 1
            k += 1
        
        # Visualize the merged subarray
        if visualize:
            highlights = {i: 'blue' for i in range(left, right+1)}
            vis.update(array, highlights, 
                      text={"Merged": f"Indices {left} to {right}"})
    
    # Helper function to implement merge sort recursively
    def merge_sort_recursive(arr: List[int], left: int, right: int) -> None:
        if left < right:
            # Find the middle point
            mid = (left + right) // 2
            
            # Visualize the division
            if visualize:
                highlights = {i: 'red' for i in range(left, mid+1)}
                highlights.update({i: 'blue' for i in range(mid+1, right+1)})
                vis.update(array, highlights, 
                          text={"Dividing": f"Indices {left} to {right}",
                               "Mid point": mid})
            
            # Sort first and second halves
            merge_sort_recursive(arr, left, mid)
            merge_sort_recursive(arr, mid + 1, right)
            
            # Merge the sorted halves
            merge(arr, left, mid, right)
    
    # Start the merge sort
    merge_sort_recursive(array, 0, len(array) - 1)
    
    # Final visualization
    if visualize:
        vis.update(array, text={"Status": "Sorted!", 
                               "Time Complexity": "O(n log n)",
                               "Space Complexity": "O(n)"})
        vis.save_animation("merge_sort")
        vis.show()
    
    return array

def visualize_merge_sort_complexity():
    """
    Create a visualization of Merge Sort's time complexity.
    """
    plt.figure(figsize=(10, 6))
    
    # Data for the plot
    n_values = np.arange(1, 101)
    best_case = n_values * np.log2(n_values)  # O(n log n)
    worst_case = n_values * np.log2(n_values)  # O(n log n)
    average_case = n_values * np.log2(n_values)  # O(n log n)
    
    # For comparison, show O(n²) and O(n)
    n_squared = n_values**2
    n_linear = n_values
    
    # Plotting
    plt.plot(n_values, best_case, label='Best Case: O(n log n)', color='green')
    plt.plot(n_values, average_case, label='Average Case: O(n log n)', color='blue')
    plt.plot(n_values, worst_case, label='Worst Case: O(n log n)', color='red')
    plt.plot(n_values, n_squared, label='O(n²)', color='black', linestyle='--', alpha=0.5)
    plt.plot(n_values, n_linear, label='O(n)', color='purple', linestyle='--', alpha=0.5)
    
    plt.xlabel('Input Size (n)')
    plt.ylabel('Number of Operations')
    plt.title('Merge Sort Time Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('visualizations/merge_sort_complexity.png')
    plt.close()
    
    # Create an educational figure explaining the algorithm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Example of merge sort process
    ax1.set_title("Merge Sort: Divide and Conquer")
    
    # Draw a tree-like structure to illustrate the divide and conquer approach
    # Level 1: Original array
    ax1.text(0.5, 0.9, "[38, 27, 43, 3, 9, 82, 10]", ha='center', fontsize=10, bbox=dict(facecolor='lightblue', alpha=0.5))
    
    # Level 2: First division
    ax1.text(0.3, 0.7, "[38, 27, 43, 3]", ha='center', fontsize=10, bbox=dict(facecolor='lightgreen', alpha=0.5))
    ax1.text(0.7, 0.7, "[9, 82, 10]", ha='center', fontsize=10, bbox=dict(facecolor='lightgreen', alpha=0.5))
    
    # Level 3: Further divisions
    ax1.text(0.2, 0.5, "[38, 27]", ha='center', fontsize=10, bbox=dict(facecolor='lightyellow', alpha=0.5))
    ax1.text(0.4, 0.5, "[43, 3]", ha='center', fontsize=10, bbox=dict(facecolor='lightyellow', alpha=0.5))
    ax1.text(0.6, 0.5, "[9]", ha='center', fontsize=10, bbox=dict(facecolor='lightyellow', alpha=0.5))
    ax1.text(0.8, 0.5, "[82, 10]", ha='center', fontsize=10, bbox=dict(facecolor='lightyellow', alpha=0.5))
    
    # Level 4: Individual elements
    ax1.text(0.15, 0.3, "[38]", ha='center', fontsize=10, bbox=dict(facecolor='lightpink', alpha=0.5))
    ax1.text(0.25, 0.3, "[27]", ha='center', fontsize=10, bbox=dict(facecolor='lightpink', alpha=0.5))
    ax1.text(0.35, 0.3, "[43]", ha='center', fontsize=10, bbox=dict(facecolor='lightpink', alpha=0.5))
    ax1.text(0.45, 0.3, "[3]", ha='center', fontsize=10, bbox=dict(facecolor='lightpink', alpha=0.5))
    ax1.text(0.75, 0.3, "[82]", ha='center', fontsize=10, bbox=dict(facecolor='lightpink', alpha=0.5))
    ax1.text(0.85, 0.3, "[10]", ha='center', fontsize=10, bbox=dict(facecolor='lightpink', alpha=0.5))
    
    # Level 5: Merging back
    ax1.text(0.2, 0.1, "[27, 38]", ha='center', fontsize=10, bbox=dict(facecolor='lightsalmon', alpha=0.5))
    ax1.text(0.4, 0.1, "[3, 43]", ha='center', fontsize=10, bbox=dict(facecolor='lightsalmon', alpha=0.5))
    ax1.text(0.8, 0.1, "[10, 82]", ha='center', fontsize=10, bbox=dict(facecolor='lightsalmon', alpha=0.5))
    
    # Add connecting lines
    ax1.plot([0.5, 0.3], [0.9, 0.7], 'k-', alpha=0.3)
    ax1.plot([0.5, 0.7], [0.9, 0.7], 'k-', alpha=0.3)
    ax1.plot([0.3, 0.2], [0.7, 0.5], 'k-', alpha=0.3)
    ax1.plot([0.3, 0.4], [0.7, 0.5], 'k-', alpha=0.3)
    ax1.plot([0.7, 0.6], [0.7, 0.5], 'k-', alpha=0.3)
    ax1.plot([0.7, 0.8], [0.7, 0.5], 'k-', alpha=0.3)
    
    ax1.plot([0.2, 0.15], [0.5, 0.3], 'k-', alpha=0.3)
    ax1.plot([0.2, 0.25], [0.5, 0.3], 'k-', alpha=0.3)
    ax1.plot([0.4, 0.35], [0.5, 0.3], 'k-', alpha=0.3)
    ax1.plot([0.4, 0.45], [0.5, 0.3], 'k-', alpha=0.3)
    ax1.plot([0.8, 0.75], [0.5, 0.3], 'k-', alpha=0.3)
    ax1.plot([0.8, 0.85], [0.5, 0.3], 'k-', alpha=0.3)
    
    ax1.plot([0.15, 0.2], [0.3, 0.1], 'k-', alpha=0.3)
    ax1.plot([0.25, 0.2], [0.3, 0.1], 'k-', alpha=0.3)
    ax1.plot([0.35, 0.4], [0.3, 0.1], 'k-', alpha=0.3)
    ax1.plot([0.45, 0.4], [0.3, 0.1], 'k-', alpha=0.3)
    ax1.plot([0.75, 0.8], [0.3, 0.1], 'k-', alpha=0.3)
    ax1.plot([0.85, 0.8], [0.3, 0.1], 'k-', alpha=0.3)
    
    ax1.axis('off')
    
    # Algorithm explanation
    explanation = """
    Merge Sort Algorithm:
    
    1. Divide the unsorted array into n subarrays, each containing one element
       (an array of one element is considered sorted)
    2. Repeatedly merge subarrays to produce new sorted subarrays until there is
       only one subarray remaining - this will be the sorted array
    
    Merge Process:
    - Compare the elements of two subarrays starting from the first element
    - Pick the smaller element and put it in the merged array
    - Move to the next element in the subarray from which the element was taken
    - Repeat until all elements are processed
    
    Key Characteristics:
    - Stable sorting algorithm
    - O(n log n) time complexity in all cases
    - O(n) space complexity
    - Not an in-place sorting algorithm
    - Efficient for large datasets
    - Used in external sorting
    """
    
    ax2.text(0.1, 0.5, explanation, fontsize=10, verticalalignment='center')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/merge_sort_explanation.png')
    plt.close()

if __name__ == "__main__":
    # Example usage
    test_array = [38, 27, 43, 3, 9, 82, 10]
    print(f"Original array: {test_array}")
    sorted_array = merge_sort(test_array, visualize=True)
    print(f"Sorted array: {sorted_array}")
    
    # Create complexity visualization
    visualize_merge_sort_complexity()
