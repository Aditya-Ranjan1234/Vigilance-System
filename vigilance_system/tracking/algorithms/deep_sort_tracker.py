"""
DeepSORT (Deep Simple Online and Realtime Tracking) implementation.

This module provides an implementation of the DeepSORT tracking algorithm.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import Detection
from vigilance_system.tracking.algorithms.base_tracker import BaseTracker, Track
from vigilance_system.tracking.algorithms.sort_tracker import KalmanTrack

logger = get_logger(__name__)


class DeepSORTTrack(KalmanTrack):
    """
    Track class for DeepSORT tracker.
    
    This class extends the KalmanTrack class with appearance features.
    """
    
    def __init__(self, detection: Detection, track_id: int, feature: np.ndarray = None):
        """
        Initialize a DeepSORT track.
        
        Args:
            detection: Initial detection for this track
            track_id: Unique ID for this track
            feature: Appearance feature vector
        """
        super().__init__(detection, track_id)
        
        # Initialize feature vector
        self.features = []
        if feature is not None:
            self.features.append(feature)
    
    def update(self, detection: Detection, feature: np.ndarray = None) -> None:
        """
        Update the track with a new detection.
        
        Args:
            detection: New detection to update the track with
            feature: New appearance feature vector
        """
        # Update track
        super().update(detection)
        
        # Update feature vector
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > 100:  # Limit feature history
                self.features.pop(0)
    
    def get_feature(self) -> np.ndarray:
        """
        Get the average feature vector for this track.
        
        Returns:
            Average feature vector
        """
        if not self.features:
            return None
        
        return np.mean(self.features, axis=0)


class FeatureExtractor:
    """
    Feature extractor for DeepSORT.
    
    This class extracts appearance features from detections.
    """
    
    def __init__(self, model_name: str = 'osnet_x0_25'):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: Name of the feature extraction model
        """
        self.model_name = model_name
        self.model = None
        self.device = config.get('tracking.algorithms.deep_sort.device', 'cpu')
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the feature extraction model."""
        try:
            import torch
            import torchvision.transforms as transforms
            
            # Try to import torchreid
            try:
                import torchreid
                
                # Load model
                self.model = torchreid.models.build_model(
                    name=self.model_name,
                    num_classes=1000,
                    pretrained=True
                )
                
                # Set model to evaluation mode
                self.model.eval()
                
                # Move model to device
                if self.device != 'cpu' and torch.cuda.is_available():
                    self.model.to(self.device)
                else:
                    self.model.to('cpu')
                    logger.warning("CUDA not available, using CPU for feature extraction")
                
                # Define preprocessing
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                logger.info(f"Successfully loaded feature extraction model: {self.model_name}")
            
            except ImportError:
                logger.warning("torchreid not found, using dummy feature extractor")
                self.model = None
        
        except Exception as e:
            logger.error(f"Error loading feature extraction model: {str(e)}")
            self.model = None
    
    def extract_features(self, frame: np.ndarray, detections: List[Detection]) -> List[np.ndarray]:
        """
        Extract features from detections.
        
        Args:
            frame: Input frame
            detections: List of detections
        
        Returns:
            List of feature vectors
        """
        if self.model is None:
            # Return dummy features
            return [np.random.rand(512).astype(np.float32) for _ in detections]
        
        try:
            import torch
            
            # Extract image patches
            patches = []
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection.bbox)
                patch = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if patch.size == 0:
                    # Empty patch, use dummy feature
                    patches.append(None)
                else:
                    patches.append(patch)
            
            # Extract features
            features = []
            for patch in patches:
                if patch is None:
                    # Use dummy feature
                    features.append(np.random.rand(512).astype(np.float32))
                else:
                    # Preprocess patch
                    tensor = self.transform(patch).unsqueeze(0)
                    
                    # Move tensor to device
                    if self.device != 'cpu' and torch.cuda.is_available():
                        tensor = tensor.to(self.device)
                    
                    # Extract feature
                    with torch.no_grad():
                        feature = self.model(tensor)
                    
                    # Convert to numpy
                    feature = feature.cpu().numpy().flatten()
                    
                    # Normalize
                    norm = np.linalg.norm(feature)
                    if norm > 0:
                        feature = feature / norm
                    
                    features.append(feature)
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return [np.random.rand(512).astype(np.float32) for _ in detections]


class DeepSORTTracker(BaseTracker):
    """
    DeepSORT (Deep Simple Online and Realtime Tracking) implementation.
    
    This class implements the DeepSORT tracking algorithm.
    """
    
    def __init__(self):
        """Initialize the DeepSORT tracker."""
        super().__init__()
        
        # Get DeepSORT specific configuration
        self.feature_model = config.get('tracking.algorithms.deep_sort.feature_model', 'osnet_x0_25')
        self.max_cosine_distance = config.get('tracking.algorithms.deep_sort.max_cosine_distance', 0.2)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.feature_model)
        
        logger.info(f"Initialized DeepSORT tracker with feature model: {self.feature_model}, "
                   f"max cosine distance: {self.max_cosine_distance}")
    
    def get_name(self) -> str:
        """
        Get the name of the tracker.
        
        Returns:
            Name of the tracker
        """
        return 'deep_sort'
    
    def update(self, detections: List[Detection], frame: np.ndarray = None) -> List[Detection]:
        """
        Update the tracker with new detections.
        
        Args:
            detections: List of new detections
            frame: Input frame (required for feature extraction)
        
        Returns:
            List of tracked detections
        """
        # Record start time for metrics
        start_time = time.time()
        
        # Increment frame count
        self.frame_count += 1
        
        # Extract features
        if frame is not None:
            features = self.feature_extractor.extract_features(frame, detections)
        else:
            features = [None] * len(detections)
        
        # Predict new locations of tracks
        predicted_tracks = []
        for track in self.tracks:
            bbox = track.predict()
            predicted_tracks.append(track)
        
        # Associate detections with tracks
        if len(predicted_tracks) > 0 and len(detections) > 0:
            # Calculate IoU between all tracks and detections
            iou_matrix = np.zeros((len(predicted_tracks), len(detections)))
            for i, track in enumerate(predicted_tracks):
                for j, detection in enumerate(detections):
                    iou_matrix[i, j] = self._calculate_iou(track.detection.bbox, detection.bbox)
            
            # Calculate feature distance
            feature_matrix = np.zeros((len(predicted_tracks), len(detections)))
            for i, track in enumerate(predicted_tracks):
                if isinstance(track, DeepSORTTrack):
                    track_feature = track.get_feature()
                    if track_feature is not None:
                        for j, feature in enumerate(features):
                            if feature is not None:
                                # Calculate cosine distance
                                cosine_distance = 1.0 - np.dot(track_feature, feature)
                                feature_matrix[i, j] = cosine_distance
            
            # Combine IoU and feature distance
            distance_matrix = (1.0 - iou_matrix) * 0.5 + feature_matrix * 0.5
            
            # Use Hungarian algorithm to find best matches
            from scipy.optimize import linear_sum_assignment
            track_indices, detection_indices = linear_sum_assignment(distance_matrix)
            
            # Update matched tracks
            for track_idx, det_idx in zip(track_indices, detection_indices):
                if iou_matrix[track_idx, det_idx] >= self.iou_threshold and \
                   feature_matrix[track_idx, det_idx] <= self.max_cosine_distance:
                    self.tracks[track_idx].update(detections[det_idx], features[det_idx])
                else:
                    # IoU or feature distance too low, treat as unmatched
                    pass
            
            # Get unmatched tracks and detections
            unmatched_tracks = [i for i in range(len(predicted_tracks)) if i not in track_indices]
            unmatched_detections = [i for i in range(len(detections)) if i not in detection_indices]
            
            # Add new tracks for unmatched detections
            for i in unmatched_detections:
                self.tracks.append(DeepSORTTrack(detections[i], self.next_id, features[i]))
                self.next_id += 1
        else:
            # No tracks or no detections
            if len(detections) > 0:
                # Create new tracks for all detections
                for i, detection in enumerate(detections):
                    self.tracks.append(DeepSORTTrack(detection, self.next_id, features[i]))
                    self.next_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Get tracked detections
        tracked_detections = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update == 0:
                tracked_detection = track.to_detection()
                tracked_detections.append(tracked_detection)
        
        # Record metrics
        self._record_metrics(start_time, self.tracks)
        
        return tracked_detections
