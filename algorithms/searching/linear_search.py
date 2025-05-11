"""
Linear Search Algorithm Implementation and Visualization.

Linear search is the simplest search algorithm that checks each element of the list
until the desired element is found or the list ends.

Time Complexity:
- Best Case: O(1) when the element is at the beginning of the array
- Average Case: O(n)
- Worst Case: O(n) when the element is at the end or not present

Space Complexity: O(1)
"""

from typing import List, Optional, Union, Dict, Any
import time
import matplotlib.pyplot as plt
import numpy as np
from algorithms.visualization import ArrayVisualizer

def linear_search(arr: List[int], target: int, visualize: bool = False) -> Union[int, None]:
    """
    Search for a target value in an array using linear search.
    
    Args:
        arr: The array to search in
        target: The value to search for
        visualize: Whether to visualize the search process
    
    Returns:
        The index of the target if found, None otherwise
    """
    # Create visualizer if needed
    vis = None
    if visualize:
        vis = ArrayVisualizer("Linear Search")
        vis.update(arr, text={"Algorithm": "Linear Search", 
                             "Target": target,
                             "Time Complexity": "O(n)",
                             "Space Complexity": "O(1)"})
    
    # Linear search algorithm
    for i in range(len(arr)):
        # Visualize the current element being checked
        if visualize:
            highlights = {i: 'yellow'}
            vis.update(arr, highlights, 
                      text={"Checking": f"Index {i}, Value {arr[i]}",
                           "Target": target})
        
        # Check if current element is the target
        if arr[i] == target:
            # Visualize the found element
            if visualize:
                highlights = {i: 'green'}
                vis.update(arr, highlights, 
                          text={"Found": f"Target {target} at index {i}",
                               "Status": "Success!"})
                vis.save_animation("linear_search_found")
                vis.show()
            
            return i
    
    # Target not found
    if visualize:
        vis.update(arr, text={"Status": "Target not found!",
                             "Target": target})
        vis.save_animation("linear_search_not_found")
        vis.show()
    
    return None

def visualize_linear_search_complexity():
    """
    Create a visualization of Linear Search's time complexity.
    """
    plt.figure(figsize=(10, 6))
    
    # Data for the plot
    n_values = np.arange(1, 101)
    best_case = np.ones_like(n_values)  # O(1)
    average_case = n_values / 2  # O(n/2) which is O(n)
    worst_case = n_values  # O(n)
    
    # Plotting
    plt.plot(n_values, best_case, label='Best Case: O(1)', color='green')
    plt.plot(n_values, average_case, label='Average Case: O(n)', color='blue')
    plt.plot(n_values, worst_case, label='Worst Case: O(n)', color='red')
    
    plt.xlabel('Input Size (n)')
    plt.ylabel('Number of Operations')
    plt.title('Linear Search Time Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('visualizations/linear_search_complexity.png')
    plt.close()
    
    # Create an educational figure explaining the algorithm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Example array for demonstration
    example = [5, 1, 4, 2, 8, 10, 7, 6]
    target = 8
    
    # Visualization of linear search process
    bars = ax1.bar(range(len(example)), example, color=['blue'] * len(example))
    
    # Highlight the target
    target_idx = example.index(target)
    bars[target_idx].set_color('green')
    
    # Show the search path
    for i in range(target_idx + 1):
        ax1.annotate("", xy=(i, example[i]), xytext=(i-0.5, example[i]+1),
                    arrowprops=dict(arrowstyle="->", color='red' if i < target_idx else 'green'))
    
    ax1.set_title("Linear Search Process")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Value")
    ax1.set_xticks(range(len(example)))
    
    # Add annotations
    ax1.text(target_idx, example[target_idx] + 2, "Target Found!", ha='center', color='green')
    ax1.text(len(example) / 2, max(example) + 3, "Sequential Search Path", ha='center', color='red')
    
    # Algorithm explanation
    explanation = """
    Linear Search Algorithm:
    
    1. Start from the leftmost element of the array
    2. Compare each element with the target value
    3. If the element matches the target, return its index
    4. If the element doesn't match, move to the next element
    5. If no match is found after checking all elements, return -1 or None
    
    Key Characteristics:
    - Simplest search algorithm
    - O(n) time complexity in worst and average cases
    - O(1) space complexity
    - Works on unsorted arrays
    - Inefficient for large datasets
    - Best for small arrays or when the target is likely near the beginning
    
    When to use Linear Search:
    - Small datasets
    - Unsorted arrays where sorting would be more expensive
    - When the target is likely at the beginning
    - When simplicity is more important than efficiency
    """
    
    ax2.text(0.1, 0.5, explanation, fontsize=10, verticalalignment='center')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/linear_search_explanation.png')
    plt.close()

if __name__ == "__main__":
    # Example usage
    test_array = [64, 34, 25, 12, 22, 11, 90]
    target = 22
    print(f"Array: {test_array}")
    print(f"Searching for: {target}")
    result = linear_search(test_array, target, visualize=True)
    
    if result is not None:
        print(f"Element found at index: {result}")
    else:
        print("Element not found")
    
    # Create complexity visualization
    visualize_linear_search_complexity()
