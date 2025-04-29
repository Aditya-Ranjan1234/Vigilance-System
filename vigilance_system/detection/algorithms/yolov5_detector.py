"""
YOLOv5 detector implementation.

This module provides an implementation of the YOLOv5 object detection algorithm.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import BaseDetector, Detection

logger = get_logger(__name__)


class YOLOv5Detector(BaseDetector):
    """
    YOLOv5 object detector implementation.
    
    This class implements the YOLOv5 object detection algorithm using PyTorch.
    """
    
    def __init__(self):
        """Initialize the YOLOv5 detector."""
        super().__init__()
        
        # Get YOLOv5 specific configuration
        self.model_size = config.get('detection.algorithms.yolov5.model_size', 's')
        self.img_size = config.get('detection.algorithms.yolov5.img_size', 640)
        
        # Load the model
        self.load_model()
    
    def load_model(self) -> None:
        """
        Load the YOLOv5 model.
        
        This method loads the YOLOv5 model from PyTorch Hub or a local file.
        """
        try:
            # Try to load from PyTorch Hub
            model_name = f"yolov5{self.model_size}"
            logger.info(f"Loading {model_name} from PyTorch Hub")
            
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            
            # Set model parameters
            self.model.conf = self.confidence_threshold
            self.model.iou = self.nms_threshold
            
            # Set device
            if self.device != 'cpu' and torch.cuda.is_available():
                self.model.to(self.device)
            else:
                self.model.to('cpu')
                logger.warning("CUDA not available, using CPU for inference")
            
            # Get class names
            self.class_names = self.model.names
            
            logger.info(f"Successfully loaded {model_name} with {len(self.class_names)} classes")
        
        except Exception as e:
            logger.error(f"Error loading YOLOv5 model: {str(e)}")
            logger.info("Attempting to load from local file")
            
            try:
                # Try to load from local file
                model_path = os.path.join('models', f"yolov5{self.model_size}.pt")
                
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    logger.info("Downloading model...")
                    
                    # Create models directory if it doesn't exist
                    os.makedirs('models', exist_ok=True)
                    
                    # Download model
                    torch.hub.download_url_to_file(
                        f"https://github.com/ultralytics/yolov5/releases/download/v6.1/{model_name}.pt",
                        model_path
                    )
                
                # Load model
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                
                # Set model parameters
                self.model.conf = self.confidence_threshold
                self.model.iou = self.nms_threshold
                
                # Set device
                if self.device != 'cpu' and torch.cuda.is_available():
                    self.model.to(self.device)
                else:
                    self.model.to('cpu')
                    logger.warning("CUDA not available, using CPU for inference")
                
                # Get class names
                self.class_names = self.model.names
                
                logger.info(f"Successfully loaded {model_name} from local file with {len(self.class_names)} classes")
            
            except Exception as e:
                logger.error(f"Error loading YOLOv5 model from local file: {str(e)}")
                raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame using YOLOv5.
        
        Args:
            frame: Input frame (BGR image)
        
        Returns:
            List of Detection objects
        """
        # Record start time for metrics
        start_time = time.time()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = self.model(rgb_frame, size=self.img_size)
        
        # Process results
        detections = []
        
        # Get detections
        for pred in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, class_id = pred
            
            # Skip if confidence is below threshold
            if conf < self.confidence_threshold:
                continue
            
            # Skip if class is not of interest
            if self.classes_of_interest is not None and int(class_id) not in self.classes_of_interest:
                continue
            
            # Create detection object
            detection = Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                class_id=int(class_id),
                class_name=self.class_names[int(class_id)],
                confidence=float(conf)
            )
            
            detections.append(detection)
        
        # Record metrics
        self._record_metrics(start_time, detections)
        
        return detections
