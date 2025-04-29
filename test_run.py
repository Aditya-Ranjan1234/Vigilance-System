#!/usr/bin/env python
"""
Test script to run the Vigilance System with our new algorithms.
"""

import os
import sys
import argparse

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run the Vigilance System with non-deep learning algorithms')
    parser.add_argument('--detection', type=str, default='background_subtraction',
                        choices=['background_subtraction', 'mog2', 'knn', 'svm_classifier'],
                        help='Detection algorithm to use')
    parser.add_argument('--tracking', type=str, default='klt_tracker',
                        choices=['klt_tracker', 'kalman_filter', 'optical_flow'],
                        help='Tracking algorithm to use')
    parser.add_argument('--loitering', type=str, default='rule_based',
                        choices=['rule_based', 'timer_threshold', 'decision_tree'],
                        help='Loitering detection algorithm to use')
    parser.add_argument('--crowd', type=str, default='blob_counting',
                        choices=['blob_counting', 'contour_counting', 'kmeans_clustering'],
                        help='Crowd detection algorithm to use')
    parser.add_argument('--preprocessing', type=str, default='feature_matching',
                        choices=['feature_matching', 'orb', 'sift', 'affine_transform'],
                        help='Preprocessing algorithm to use')
    args = parser.parse_args()
    
    # Build the command
    cmd = [
        'python', '-m', 'vigilance_system',
        f'--detection-algorithm={args.detection}',
        f'--tracking-algorithm={args.tracking}',
        f'--loitering-algorithm={args.loitering}',
        f'--crowd-algorithm={args.crowd}',
        f'--preprocessing-algorithm={args.preprocessing}'
    ]
    
    # Print the command
    print("Running command:")
    print(' '.join(cmd))
    
    # Run the command
    os.system(' '.join(cmd))
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
