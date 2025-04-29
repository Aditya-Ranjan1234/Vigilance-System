"""
YOLOv8 detector implementation.

This module provides an implementation of the YOLOv8 object detection algorithm.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

from vigilance_system.utils.config import config
from vigilance_system.utils.logger import get_logger
from vigilance_system.detection.algorithms.base_detector import BaseDetector, Detection

logger = get_logger(__name__)


class YOLOv8Detector(BaseDetector):
    """
    YOLOv8 object detector implementation.
    
    This class implements the YOLOv8 object detection algorithm using Ultralytics YOLOv8.
    """
    
    def __init__(self):
        """Initialize the YOLOv8 detector."""
        super().__init__()
        
        # Get YOLOv8 specific configuration
        self.model_size = config.get('detection.algorithms.yolov8.model_size', 's')
        self.img_size = config.get('detection.algorithms.yolov8.img_size', 640)
        
        # Load the model
        self.load_model()
    
    def load_model(self) -> None:
        """
        Load the YOLOv8 model.
        
        This method loads the YOLOv8 model from Ultralytics.
        """
        try:
            # Try to import ultralytics
            from ultralytics import YOLO
            
            model_name = f"yolov8{self.model_size}"
            logger.info(f"Loading {model_name} from Ultralytics")
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Check if model exists locally
            model_path = os.path.join('models', f"{model_name}.pt")
            
            if os.path.exists(model_path):
                logger.info(f"Loading {model_name} from local file: {model_path}")
                self.model = YOLO(model_path)
            else:
                logger.info(f"Downloading {model_name} from Ultralytics")
                self.model = YOLO(model_name)
                
                # Save model for future use
                self.model.export(format="onnx")
                logger.info(f"Saved {model_name} to {model_path}")
            
            # Set device
            if self.device != 'cpu' and self.device.startswith('cuda'):
                self.model.to(self.device)
            else:
                self.model.to('cpu')
                logger.warning("CUDA not available or not specified, using CPU for inference")
            
            # Get class names
            self.class_names = self.model.names
            
            logger.info(f"Successfully loaded {model_name} with {len(self.class_names)} classes")
        
        except ImportError:
            logger.error("Ultralytics package not found. Please install it with: pip install ultralytics")
            raise
        
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {str(e)}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame using YOLOv8.
        
        Args:
            frame: Input frame (BGR image)
        
        Returns:
            List of Detection objects
        """
        # Record start time for metrics
        start_time = time.time()
        
        # Perform inference
        results = self.model(frame, conf=self.confidence_threshold, iou=self.nms_threshold, verbose=False)
        
        # Process results
        detections = []
        
        # Get detections from first result
        if results and len(results) > 0:
            result = results[0]
            
            # Process each detection
            for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy.cpu().numpy(), 
                                                   result.boxes.conf.cpu().numpy(),
                                                   result.boxes.cls.cpu().numpy())):
                
                # Skip if confidence is below threshold
                if conf < self.confidence_threshold:
                    continue
                
                # Skip if class is not of interest
                class_id = int(cls)
                if self.classes_of_interest is not None and class_id not in self.classes_of_interest:
                    continue
                
                # Create detection object
                detection = Detection(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    confidence=float(conf)
                )
                
                detections.append(detection)
        
        # Record metrics
        self._record_metrics(start_time, detections)
        
        return detections
