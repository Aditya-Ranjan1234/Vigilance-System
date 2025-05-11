"""
Binary Search Algorithm Implementation and Visualization.

Binary search is a search algorithm that finds the position of a target value
within a sorted array. It works by repeatedly dividing the search interval in half.

Time Complexity:
- Best Case: O(1) when the element is at the middle of the array
- Average Case: O(log n)
- Worst Case: O(log n) when the element is at the extremes or not present

Space Complexity: 
- O(1) for iterative implementation
- O(log n) for recursive implementation due to the call stack
"""

from typing import List, Optional, Union, Dict, Any
import time
import matplotlib.pyplot as plt
import numpy as np
from algorithms.visualization import ArrayVisualizer

def binary_search(arr: List[int], target: int, visualize: bool = False) -> Union[int, None]:
    """
    Search for a target value in a sorted array using binary search.
    
    Args:
        arr: The sorted array to search in
        target: The value to search for
        visualize: Whether to visualize the search process
    
    Returns:
        The index of the target if found, None otherwise
    """
    # Check if array is sorted
    if not all(arr[i] <= arr[i+1] for i in range(len(arr)-1)):
        raise ValueError("Array must be sorted for binary search")
    
    # Create visualizer if needed
    vis = None
    if visualize:
        vis = ArrayVisualizer("Binary Search")
        vis.update(arr, text={"Algorithm": "Binary Search", 
                             "Target": target,
                             "Time Complexity": "O(log n)",
                             "Space Complexity": "O(1)"})
    
    # Binary search algorithm
    left, right = 0, len(arr) - 1
    
    while left <= right:
        # Calculate middle index
        mid = (left + right) // 2
        
        # Visualize the current search interval
        if visualize:
            highlights = {i: 'lightblue' for i in range(left, right + 1)}
            highlights[mid] = 'yellow'
            vis.update(arr, highlights, 
                      text={"Searching": f"Interval [{left}, {right}]",
                           "Middle": f"Index {mid}, Value {arr[mid]}",
                           "Target": target})
        
        # Check if target is present at mid
        if arr[mid] == target:
            # Visualize the found element
            if visualize:
                highlights = {mid: 'green'}
                vis.update(arr, highlights, 
                          text={"Found": f"Target {target} at index {mid}",
                               "Status": "Success!"})
                vis.save_animation("binary_search_found")
                vis.show()
            
            return mid
        
        # If target is greater, ignore left half
        elif arr[mid] < target:
            left = mid + 1
            
            # Visualize the updated search interval
            if visualize:
                highlights = {i: 'gray' for i in range(0, mid + 1)}
                highlights.update({i: 'lightblue' for i in range(mid + 1, right + 1)})
                vis.update(arr, highlights, 
                          text={"Update": f"Target > {arr[mid]}, search right half",
                               "New Interval": f"[{left}, {right}]"})
        
        # If target is smaller, ignore right half
        else:
            right = mid - 1
            
            # Visualize the updated search interval
            if visualize:
                highlights = {i: 'lightblue' for i in range(left, mid)}
                highlights.update({i: 'gray' for i in range(mid, len(arr))})
                vis.update(arr, highlights, 
                          text={"Update": f"Target < {arr[mid]}, search left half",
                               "New Interval": f"[{left}, {right}]"})
    
    # Target not found
    if visualize:
        vis.update(arr, text={"Status": "Target not found!",
                             "Target": target})
        vis.save_animation("binary_search_not_found")
        vis.show()
    
    return None

def visualize_binary_search_complexity():
    """
    Create a visualization of Binary Search's time complexity.
    """
    plt.figure(figsize=(10, 6))
    
    # Data for the plot
    n_values = np.arange(1, 101)
    best_case = np.ones_like(n_values)  # O(1)
    average_case = np.log2(n_values)  # O(log n)
    worst_case = np.log2(n_values)  # O(log n)
    
    # For comparison, show O(n)
    linear = n_values
    
    # Plotting
    plt.plot(n_values, best_case, label='Best Case: O(1)', color='green')
    plt.plot(n_values, average_case, label='Average Case: O(log n)', color='blue')
    plt.plot(n_values, worst_case, label='Worst Case: O(log n)', color='red')
    plt.plot(n_values, linear, label='Linear Search: O(n)', color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Input Size (n)')
    plt.ylabel('Number of Operations')
    plt.title('Binary Search Time Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('visualizations/binary_search_complexity.png')
    plt.close()
    
    # Create an educational figure explaining the algorithm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Example array for demonstration
    example = [1, 2, 4, 5, 7, 8, 10, 12, 14, 15, 18, 20]
    target = 14
    
    # Visualization of binary search process
    ax1.set_title("Binary Search Process")
    
    # Initial state
    bars = ax1.bar(range(len(example)), example, color='lightblue')
    
    # Show the search steps
    left, right = 0, len(example) - 1
    steps = []
    
    while left <= right:
        mid = (left + right) // 2
        
        if example[mid] == target:
            steps.append((left, mid, right, "Found"))
            break
        elif example[mid] < target:
            steps.append((left, mid, right, "Go Right"))
            left = mid + 1
        else:
            steps.append((left, mid, right, "Go Left"))
            right = mid - 1
    
    # Annotate the steps
    y_pos = max(example) + 2
    for i, (l, m, r, action) in enumerate(steps):
        # Draw the search interval
        ax1.annotate(f"Step {i+1}: [{l}, {r}]", xy=(0, y_pos - i*2), 
                    xytext=(0, y_pos - i*2), fontsize=8)
        
        # Draw the middle point
        ax1.annotate(f"mid={m}, value={example[m]}", xy=(m, example[m]), 
                    xytext=(m, example[m] + 1), fontsize=8, ha='center',
                    arrowprops=dict(arrowstyle="->", color='red'))
        
        # Draw the action
        ax1.annotate(action, xy=(len(example) - 3, y_pos - i*2), 
                    xytext=(len(example) - 3, y_pos - i*2), fontsize=8, 
                    color='green' if action == "Found" else 'blue')
    
    # Highlight the target
    target_idx = example.index(target)
    bars[target_idx].set_color('green')
    
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Value")
    ax1.set_xticks(range(len(example)))
    
    # Algorithm explanation
    explanation = """
    Binary Search Algorithm:
    
    1. Compare the target value to the middle element of the array
    2. If they are equal, return the index of the middle element
    3. If the target is less than the middle element, search the left half
    4. If the target is greater than the middle element, search the right half
    5. Repeat steps 1-4 until the element is found or the search interval is empty
    
    Key Characteristics:
    - Requires a sorted array
    - O(log n) time complexity in worst and average cases
    - O(1) space complexity for iterative implementation
    - Much more efficient than linear search for large datasets
    - Divide and conquer approach
    
    When to use Binary Search:
    - Sorted arrays
    - Large datasets
    - When search operations are frequent
    - When the cost of sorting is amortized over many searches
    
    Comparison with Linear Search:
    - For n=1,000,000 elements:
      * Linear search: up to 1,000,000 comparisons
      * Binary search: at most 20 comparisons (log₂ 1,000,000 ≈ 20)
    """
    
    ax2.text(0.1, 0.5, explanation, fontsize=10, verticalalignment='center')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/binary_search_explanation.png')
    plt.close()

if __name__ == "__main__":
    # Example usage
    test_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target = 7
    print(f"Array: {test_array}")
    print(f"Searching for: {target}")
    result = binary_search(test_array, target, visualize=True)
    
    if result is not None:
        print(f"Element found at index: {result}")
    else:
        print("Element not found")
    
    # Create complexity visualization
    visualize_binary_search_complexity()
