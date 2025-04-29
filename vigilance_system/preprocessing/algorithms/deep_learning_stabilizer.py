"""
Deep learning-based video stabilizer implementation.

This module provides a video stabilizer based on deep learning.
"""

import os
import time
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.preprocessing.algorithms.base_stabilizer import BaseStabilizer

logger = get_logger(__name__)


class DeepLearningStabilizer(BaseStabilizer):
    """
    Deep learning-based video stabilizer implementation.
    
    This class implements a video stabilizer that uses deep learning
    to predict and correct camera motion.
    """
    
    def __init__(self):
        """Initialize the deep learning stabilizer."""
        super().__init__()
        
        # Get deep learning specific configuration
        self.model_name = config.get(f'{self.algorithm_config}.model', 'DeepStab')
        self.batch_size = config.get(f'{self.algorithm_config}.batch_size', 1)
        self.model_path = config.get(f'{self.algorithm_config}.model_path', 'models/deep_stab.h5')
        
        # Initialize model
        self.model = None
        self._load_model()
        
        # Initialize state
        self.frame_buffer = []
        
        logger.info(f"Initialized deep learning stabilizer with "
                   f"model={self.model_name}, batch_size={self.batch_size}")
    
    def get_name(self) -> str:
        """
        Get the name of the stabilizer.
        
        Returns:
            Name of the stabilizer
        """
        return 'deep_learning'
    
    def _load_model(self) -> None:
        """Load the deep learning model."""
        try:
            # Try to import TensorFlow
            import tensorflow as tf
            
            # Check if model file exists
            if os.path.exists(self.model_path):
                # Load model
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded deep learning model from {self.model_path}")
            else:
                # Create a simple model for demonstration
                logger.warning(f"Model file not found: {self.model_path}, using optical flow fallback")
                self.model = None
        
        except ImportError:
            logger.warning("TensorFlow not found, using optical flow fallback")
            self.model = None
        
        except Exception as e:
            logger.error(f"Error loading deep learning model: {str(e)}")
            self.model = None
    
    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        Stabilize a frame using deep learning.
        
        Args:
            frame: Input frame
        
        Returns:
            Stabilized frame
        """
        # Record start time for metrics
        start_time = time.time()
        
        # Increment frame count
        self.frame_count += 1
        
        # Check if stabilization is enabled
        if not self.enabled:
            # Record metrics
            self._record_metrics(start_time, 1.0)
            return frame
        
        # Check if we have a model
        if self.model is None:
            # Fall back to optical flow stabilization
            return self._optical_flow_stabilize(frame, start_time)
        
        try:
            # Add frame to buffer
            self.frame_buffer.append(frame.copy())
            
            # Keep buffer size limited
            max_buffer_size = max(self.batch_size, self.smoothing_radius)
            if len(self.frame_buffer) > max_buffer_size:
                self.frame_buffer.pop(0)
            
            # Check if we have enough frames
            if len(self.frame_buffer) < 2:
                # Not enough frames, return as is
                # Record metrics
                self._record_metrics(start_time, 1.0)
                return frame
            
            # Prepare input for model
            if len(self.frame_buffer) >= self.batch_size:
                # Use batch of frames
                batch = self.frame_buffer[-self.batch_size:]
            else:
                # Use available frames
                batch = self.frame_buffer
            
            # Preprocess frames
            processed_batch = self._preprocess_frames(batch)
            
            # Predict transformation
            transform = self._predict_transform(processed_batch)
            
            if transform is not None:
                # Add to transforms list
                self.transforms.append(transform)
                
                # Smooth transforms
                smoothed_transform = self._smooth_transform(transform)
                
                # Apply smoothed transform
                stabilized = cv2.warpAffine(
                    frame, smoothed_transform, (frame.shape[1], frame.shape[0])
                )
                
                # Calculate stability score
                stability_score = self._calculate_stability_score(transform)
                
                # Record metrics
                self._record_metrics(start_time, stability_score)
                
                return stabilized
            
            # Fallback: return original frame
            # Record metrics
            self._record_metrics(start_time, 1.0)
            return frame
        
        except Exception as e:
            logger.error(f"Error in deep learning stabilization: {str(e)}")
            return self._optical_flow_stabilize(frame, start_time)
    
    def _optical_flow_stabilize(self, frame: np.ndarray, start_time: float) -> np.ndarray:
        """
        Fallback to optical flow stabilization.
        
        Args:
            frame: Input frame
            start_time: Start time for metrics
        
        Returns:
            Stabilized frame
        """
        # Import here to avoid circular imports
        from vigilance_system.preprocessing.algorithms.optical_flow_stabilizer import OpticalFlowStabilizer
        
        # Create optical flow stabilizer if not exists
        if not hasattr(self, 'optical_flow_stabilizer'):
            self.optical_flow_stabilizer = OpticalFlowStabilizer()
            self.optical_flow_stabilizer.set_camera_name(self.camera_name)
        
        # Stabilize using optical flow
        stabilized = self.optical_flow_stabilizer.stabilize(frame)
        
        # Record metrics
        self._record_metrics(start_time, 0.8)  # Assume slightly worse stability than deep learning
        
        return stabilized
    
    def _preprocess_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess frames for the deep learning model.
        
        Args:
            frames: List of frames
        
        Returns:
            Preprocessed frames as a numpy array
        """
        # Resize frames
        resized_frames = [cv2.resize(frame, (224, 224)) for frame in frames]
        
        # Convert to float32 and normalize
        normalized_frames = [frame.astype(np.float32) / 255.0 for frame in resized_frames]
        
        # Stack frames
        stacked_frames = np.stack(normalized_frames, axis=0)
        
        return stacked_frames
    
    def _predict_transform(self, processed_batch: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict transformation using the deep learning model.
        
        Args:
            processed_batch: Preprocessed batch of frames
        
        Returns:
            Transformation matrix or None if prediction failed
        """
        try:
            import tensorflow as tf
            
            # Make prediction
            prediction = self.model.predict(processed_batch, verbose=0)
            
            # Extract transformation parameters
            # This is a simplified example, actual implementation would depend on the model
            if isinstance(prediction, list):
                prediction = prediction[0]
            
            # Assume prediction contains [dx, dy, angle, scale]
            dx = prediction[0, 0] * 100  # Scale to reasonable range
            dy = prediction[0, 1] * 100
            angle = prediction[0, 2] * 10  # In degrees
            scale = 1.0 + prediction[0, 3] * 0.2  # Scale factor around 1.0
            
            # Create transformation matrix
            transform = cv2.getRotationMatrix2D((processed_batch.shape[2] / 2, processed_batch.shape[1] / 2), angle, scale)
            transform[0, 2] += dx
            transform[1, 2] += dy
            
            return transform
        
        except Exception as e:
            logger.error(f"Error predicting transform: {str(e)}")
            return None
    
    def _smooth_transform(self, transform: np.ndarray) -> np.ndarray:
        """
        Smooth a transformation using a moving average.
        
        Args:
            transform: Current transformation matrix
        
        Returns:
            Smoothed transformation matrix
        """
        # Limit transforms list size
        if len(self.transforms) > self.smoothing_radius * 2:
            self.transforms.pop(0)
        
        # Get window of transforms
        window_size = min(len(self.transforms), self.smoothing_radius)
        window = self.transforms[-window_size:]
        
        # Calculate average transform
        avg_transform = np.zeros_like(transform)
        for t in window:
            avg_transform += t
        avg_transform /= len(window)
        
        return avg_transform
