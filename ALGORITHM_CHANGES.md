# Algorithm Changes

This document outlines the changes made to the Vigilance System to replace deep learning algorithms with traditional computer vision and machine learning approaches.

## Overview

The system has been updated to use non-deep learning algorithms for all components:

- **Object Detection**: Replaced deep learning models with classical computer vision techniques
- **Object Tracking**: Replaced deep learning trackers with feature-based and statistical approaches
- **Loitering Detection**: Replaced deep learning prediction with rule-based and statistical methods
- **Crowd Detection**: Replaced deep learning density maps with clustering and counting approaches
- **Video Preprocessing**: Replaced deep learning stabilization with feature-based techniques

## Detailed Changes

### Object Detection

| Previous Algorithm | New Algorithm | Description |
|-------------------|---------------|-------------|
| YOLOv5 | Background Subtraction | Simple background subtraction for detecting moving objects |
| YOLOv8 | MOG2 | Background subtraction with Gaussian mixture models |
| SSD | KNN | K-nearest neighbors background subtraction |
| Faster R-CNN | SVM Classifier | Support Vector Machine classifier with HOG features |

### Object Tracking

| Previous Algorithm | New Algorithm | Description |
|-------------------|---------------|-------------|
| SORT | KLT Tracker | Kanade-Lucas-Tomasi feature tracker |
| DeepSORT | Kalman Filter | Kalman filter for motion prediction |
| IoU | Optical Flow | Dense optical flow for tracking objects |

### Loitering Detection

| Previous Algorithm | New Algorithm | Description |
|-------------------|---------------|-------------|
| Time Threshold | Rule-based | Simple rules based on time spent and movement patterns |
| Trajectory Heatmap | Timer Threshold | Zone-based timer for detecting loitering |
| LSTM Prediction | Decision Tree | Decision tree based on trajectory features |

### Crowd Detection

| Previous Algorithm | New Algorithm | Description |
|-------------------|---------------|-------------|
| Count Threshold | Blob Counting | Simple blob counting for detecting crowds |
| Density Map | Contour Counting | Contour-based approach for crowd detection |
| Clustering (DBSCAN) | K-means Clustering | K-means clustering for grouping people into crowds |

### Video Preprocessing

| Previous Algorithm | New Algorithm | Description |
|-------------------|---------------|-------------|
| Optical Flow | Feature Matching | General feature matching for video stabilization |
| Feature-based | ORB | ORB feature detector and descriptor for stabilization |
| Deep Learning | SIFT | SIFT feature detector and descriptor for stabilization |
| - | Affine Transform | Affine transform based stabilization |

## Benefits of the Changes

1. **Reduced Resource Requirements**: The new algorithms require less computational resources and can run efficiently on CPUs without requiring GPUs.

2. **Faster Processing**: Many of the traditional computer vision techniques are faster than their deep learning counterparts, allowing for better real-time performance.

3. **Simpler Implementation**: The new algorithms are simpler to understand and implement, making the system more maintainable.

4. **Reduced Dependencies**: The system no longer depends on large deep learning frameworks like PyTorch, reducing the installation size and complexity.

5. **Better Explainability**: Traditional algorithms often provide better explainability of their decisions compared to deep learning "black boxes."

## Performance Comparison

The Analysis Dashboard allows you to compare the performance of different algorithms. In general, you can expect:

- **Higher FPS**: The non-deep learning algorithms typically achieve higher frames per second.
- **Lower Precision/Recall**: The traditional methods may have lower precision and recall compared to deep learning approaches.
- **Comparable Tracking Performance**: For simple scenes, the tracking performance is comparable to deep learning approaches.
- **Adequate Alert Generation**: The alert generation (loitering, crowd detection) is adequate for most use cases.

## Configuration

The system has been updated to use the new algorithms by default. You can still specify which algorithm to use in the configuration file or via command-line arguments:

```bash
python -m vigilance_system --detection-algorithm background_subtraction --tracking-algorithm klt_tracker --loitering-algorithm rule_based --crowd-algorithm blob_counting --preprocessing-algorithm feature_matching
```

See the [README.md](README.md) and [QUICK_START.md](QUICK_START.md) for more detailed configuration options.
