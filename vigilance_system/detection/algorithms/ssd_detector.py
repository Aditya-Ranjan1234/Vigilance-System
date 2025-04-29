"""
SSD (Single Shot Detector) implementation.

This module provides an implementation of the SSD object detection algorithm.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import BaseDetector, Detection

logger = get_logger(__name__)


class SSDDetector(BaseDetector):
    """
    SSD (Single Shot Detector) implementation.
    
    This class implements the SSD object detection algorithm using PyTorch.
    """
    
    def __init__(self):
        """Initialize the SSD detector."""
        super().__init__()
        
        # Get SSD specific configuration
        self.backbone = config.get('detection.algorithms.ssd.backbone', 'mobilenet_v2')
        self.img_size = config.get('detection.algorithms.ssd.img_size', 300)
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Load the model
        self.load_model()
    
    def load_model(self) -> None:
        """
        Load the SSD model.
        
        This method loads the SSD model from torchvision.
        """
        try:
            logger.info(f"Loading SSD with {self.backbone} backbone")
            
            # Load pre-trained model
            if self.backbone == 'mobilenet_v2':
                self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
            else:
                self.model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Set device
            if self.device != 'cpu' and torch.cuda.is_available():
                self.model.to(self.device)
            else:
                self.model.to('cpu')
                logger.warning("CUDA not available, using CPU for inference")
            
            logger.info(f"Successfully loaded SSD model with {len(self.class_names)} classes")
        
        except Exception as e:
            logger.error(f"Error loading SSD model: {str(e)}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame using SSD.
        
        Args:
            frame: Input frame (BGR image)
        
        Returns:
            List of Detection objects
        """
        # Record start time for metrics
        start_time = time.time()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize image
        if self.img_size != frame.shape[0] or self.img_size != frame.shape[1]:
            rgb_frame = cv2.resize(rgb_frame, (self.img_size, self.img_size))
        
        # Convert to tensor
        tensor = torch.from_numpy(rgb_frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        
        # Move tensor to device
        if self.device != 'cpu' and torch.cuda.is_available():
            tensor = tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            predictions = self.model(tensor)
        
        # Process results
        detections = []
        
        # Get detections
        for i in range(len(predictions[0]['boxes'])):
            box = predictions[0]['boxes'][i].cpu().numpy()
            score = predictions[0]['scores'][i].cpu().numpy()
            class_id = predictions[0]['labels'][i].cpu().numpy()
            
            # Skip if confidence is below threshold
            if score < self.confidence_threshold:
                continue
            
            # Skip if class is not of interest
            if self.classes_of_interest is not None and int(class_id) not in self.classes_of_interest:
                continue
            
            # Scale box coordinates to original image size
            if self.img_size != frame.shape[0] or self.img_size != frame.shape[1]:
                scale_x = frame.shape[1] / self.img_size
                scale_y = frame.shape[0] / self.img_size
                box[0] *= scale_x
                box[1] *= scale_y
                box[2] *= scale_x
                box[3] *= scale_y
            
            # Create detection object
            detection = Detection(
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                class_id=int(class_id - 1),  # SSD uses 1-indexed classes, convert to 0-indexed
                class_name=self.class_names[int(class_id - 1)],
                confidence=float(score)
            )
            
            detections.append(detection)
        
        # Record metrics
        self._record_metrics(start_time, detections)
        
        return detections
