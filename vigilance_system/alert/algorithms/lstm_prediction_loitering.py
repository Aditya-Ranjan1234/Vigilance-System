"""
LSTM-based loitering detector implementation.

This module provides a loitering detector based on LSTM prediction of movement patterns.
"""

import time
import os
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import Detection
from vigilance_system.alert.algorithms.base_loitering_detector import BaseLoiteringDetector, LoiteringEvent

logger = get_logger(__name__)


class LSTMPredictionLoiteringDetector(BaseLoiteringDetector):
    """
    LSTM-based loitering detector implementation.
    
    This class implements a loitering detector that uses LSTM to predict
    movement patterns and detect suspicious behavior.
    """
    
    def __init__(self):
        """Initialize the LSTM prediction loitering detector."""
        super().__init__()
        
        # Get configuration
        self.sequence_length = config.get(f'{self.algorithm_config}.sequence_length', 20)
        self.prediction_threshold = config.get(f'{self.algorithm_config}.prediction_threshold', 0.8)
        self.model_path = config.get(f'{self.algorithm_config}.model_path', 'models/lstm_loitering.h5')
        
        # Initialize track history
        self.track_history = {}
        
        # Initialize LSTM model
        self.model = None
        self._load_model()
        
        logger.info(f"Initialized LSTM prediction loitering detector with "
                   f"sequence_length={self.sequence_length}, "
                   f"prediction_threshold={self.prediction_threshold}")
    
    def get_name(self) -> str:
        """
        Get the name of the loitering detector.
        
        Returns:
            Name of the loitering detector
        """
        return 'lstm_prediction'
    
    def _load_model(self) -> None:
        """Load the LSTM model."""
        try:
            # Try to import TensorFlow
            import tensorflow as tf
            
            # Check if model file exists
            if os.path.exists(self.model_path):
                # Load model
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded LSTM model from {self.model_path}")
            else:
                # Create a simple model for demonstration
                logger.warning(f"Model file not found: {self.model_path}, creating a dummy model")
                self._create_dummy_model()
        
        except ImportError:
            logger.warning("TensorFlow not found, using dummy prediction")
            self.model = None
        
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
            self.model = None
    
    def _create_dummy_model(self) -> None:
        """Create a dummy LSTM model for demonstration."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            # Create a simple model
            model = Sequential()
            model.add(LSTM(64, input_shape=(self.sequence_length, 2), return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(32))
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model.save(self.model_path)
            
            self.model = model
            logger.info(f"Created and saved dummy LSTM model to {self.model_path}")
        
        except Exception as e:
            logger.error(f"Error creating dummy LSTM model: {str(e)}")
            self.model = None
    
    def update(self, detections: List[Detection], frame: np.ndarray, frame_id: int) -> List[LoiteringEvent]:
        """
        Update the loitering detector with new detections.
        
        Args:
            detections: List of detections with tracking IDs
            frame: Current frame
            frame_id: ID of the current frame
        
        Returns:
            List of active loitering events
        """
        # Record start time for metrics
        start_time = time.time()
        
        # Increment frame count
        self.frame_count += 1
        
        # Filter person detections
        person_detections = [d for d in detections if d.class_name.lower() == 'person' and hasattr(d, 'tracking_id')]
        
        # Update track history
        current_time = time.time()
        current_track_ids = []
        
        for detection in person_detections:
            track_id = detection.tracking_id
            current_track_ids.append(track_id)
            
            # Calculate center point
            center_x = (detection.bbox[0] + detection.bbox[2]) / 2
            center_y = (detection.bbox[1] + detection.bbox[3]) / 2
            center = (center_x, center_y)
            
            if track_id not in self.track_history:
                # New track
                self.track_history[track_id] = {
                    'positions': [center],
                    'timestamps': [current_time],
                    'start_time': current_time,
                    'last_update_time': current_time,
                    'loitering_score': 0.0,
                    'event_id': None
                }
            else:
                # Existing track
                history = self.track_history[track_id]
                
                # Update history
                history['positions'].append(center)
                history['timestamps'].append(current_time)
                history['last_update_time'] = current_time
                
                # Limit history size
                if len(history['positions']) > max(100, self.sequence_length * 2):
                    history['positions'].pop(0)
                    history['timestamps'].pop(0)
                
                # Check if we have enough history for prediction
                if len(history['positions']) >= self.sequence_length:
                    # Predict loitering score
                    loitering_score = self._predict_loitering(history['positions'][-self.sequence_length:])
                    history['loitering_score'] = loitering_score
                    
                    # Check if score exceeds threshold
                    if loitering_score >= self.prediction_threshold:
                        # Loitering detected
                        if history['event_id'] is None:
                            # Create new loitering event
                            event = LoiteringEvent(
                                track_id=track_id,
                                location=center,
                                bbox=detection.bbox,
                                frame_id=frame_id,
                                confidence=loitering_score
                            )
                            event.id = self.next_event_id
                            self.next_event_id += 1
                            
                            # Add to events list
                            self.loitering_events.append(event)
                            
                            # Update history
                            history['event_id'] = event.id
                        else:
                            # Update existing event
                            event_id = history['event_id']
                            event = next((e for e in self.loitering_events if e.id == event_id), None)
                            
                            if event is not None and event.is_active:
                                event.update(center, detection.bbox, frame_id, loitering_score)
                    else:
                        # Score below threshold
                        if history['event_id'] is not None:
                            # End event
                            event_id = history['event_id']
                            event = next((e for e in self.loitering_events if e.id == event_id), None)
                            
                            if event is not None and event.is_active:
                                event.end()
                            
                            history['event_id'] = None
        
        # Remove old tracks
        tracks_to_remove = []
        
        for track_id, history in self.track_history.items():
            if track_id not in current_track_ids:
                # Track lost
                if current_time - history['last_update_time'] > 5:  # 5 seconds timeout
                    tracks_to_remove.append(track_id)
                    
                    # End event if exists
                    if history['event_id'] is not None:
                        event_id = history['event_id']
                        event = next((e for e in self.loitering_events if e.id == event_id), None)
                        
                        if event is not None and event.is_active:
                            event.end()
        
        # Remove tracks
        for track_id in tracks_to_remove:
            del self.track_history[track_id]
        
        # Get active events
        active_events = [e for e in self.loitering_events if e.is_active]
        
        # Record metrics
        self._record_metrics(start_time, active_events)
        
        return active_events
    
    def _predict_loitering(self, positions: List[Tuple[float, float]]) -> float:
        """
        Predict loitering score from a sequence of positions.
        
        Args:
            positions: List of positions (x, y)
        
        Returns:
            Loitering score (0-1)
        """
        if self.model is None:
            # Use a simple heuristic if model is not available
            return self._heuristic_prediction(positions)
        
        try:
            import tensorflow as tf
            import numpy as np
            
            # Normalize positions
            positions_array = np.array(positions)
            min_vals = np.min(positions_array, axis=0)
            max_vals = np.max(positions_array, axis=0)
            range_vals = max_vals - min_vals
            
            # Avoid division by zero
            range_vals = np.maximum(range_vals, 1e-5)
            
            normalized_positions = (positions_array - min_vals) / range_vals
            
            # Reshape for LSTM input
            X = normalized_positions.reshape(1, self.sequence_length, 2)
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)[0][0]
            
            return float(prediction)
        
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {str(e)}")
            return self._heuristic_prediction(positions)
    
    def _heuristic_prediction(self, positions: List[Tuple[float, float]]) -> float:
        """
        Use a simple heuristic to predict loitering.
        
        Args:
            positions: List of positions (x, y)
        
        Returns:
            Loitering score (0-1)
        """
        # Calculate total distance traveled
        total_distance = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            distance = (dx ** 2 + dy ** 2) ** 0.5
            total_distance += distance
        
        # Calculate bounding box of positions
        min_x = min(p[0] for p in positions)
        min_y = min(p[1] for p in positions)
        max_x = max(p[0] for p in positions)
        max_y = max(p[1] for p in positions)
        
        # Calculate diagonal of bounding box
        diagonal = ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5
        
        # Calculate ratio of total distance to diagonal
        if diagonal < 1e-5:
            ratio = 0.0
        else:
            ratio = total_distance / diagonal
        
        # Convert to loitering score (high ratio = low loitering)
        score = max(0.0, min(1.0, 2.0 - ratio / 5.0))
        
        return score
