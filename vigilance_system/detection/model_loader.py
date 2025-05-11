"""
Model loader module for loading and managing detection models.

This module provides functionality to load and manage different object detection models.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from vigilance_system.utils.logger import get_logger
from vigilance_system.utils.config import config

# Initialize logger
logger = get_logger(__name__)


class ModelLoader:
    """
    Loads and manages object detection models.

    Supports different model architectures and handles model loading, caching,
    and device management.
    """

    _instance = None

    def __new__(cls):
        """
        Singleton pattern implementation to ensure only one model loader exists.

        Returns:
            ModelLoader: The singleton ModelLoader instance
        """
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the model loader."""
        if self._initialized:
            return

        self.models = {}
        self.device = self._get_device()
        self.model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'models'
        )
        os.makedirs(self.model_dir, exist_ok=True)

        self._initialized = True
        logger.info(f"Model loader initialized with device: {self.device}")

    def _get_device(self) -> torch.device:
        """
        Determine the device to use for model inference.

        Returns:
            torch.device: Device to use (CUDA or CPU)
        """
        device_name = config.get('detection.device', 'cuda:0')

        # Check if CUDA is available
        cuda_available = False
        try:
            cuda_available = torch.cuda.is_available()
        except AssertionError:
            # This happens when PyTorch is not compiled with CUDA support
            cuda_available = False

        if device_name.startswith('cuda') and not cuda_available:
            logger.warning("CUDA requested but not available (PyTorch may not be installed with CUDA support), falling back to CPU")
            logger.info("To use CUDA, reinstall PyTorch with CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            return torch.device('cpu')

        return torch.device(device_name)

    def load_model(self, model_name: str) -> Any:
        """
        Load a model by name.

        Args:
            model_name: Name of the model to load

        Returns:
            Any: Loaded model

        Raises:
            ValueError: If the model is not supported
        """
        if model_name in self.models:
            logger.info(f"Using cached model: {model_name}")
            return self.models[model_name]

        logger.info(f"Loading model: {model_name}")

        try:
            if model_name.startswith('yolov5'):
                model = self._load_yolov5_model(model_name)
            elif model_name in ['background_subtraction', 'hog_svm']:
                # These models are handled directly in the ObjectDetector class
                # Return a dummy model to indicate it's supported
                model = model_name
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            self.models[model_name] = model
            return model

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def _load_yolov5_model(self, model_name: str) -> Any:
        """
        Load a YOLOv5 model.

        Args:
            model_name: Name of the YOLOv5 model (e.g., 'yolov5s', 'yolov5m')

        Returns:
            Any: Loaded YOLOv5 model
        """
        try:
            # Import here to avoid dependency if not using YOLOv5
            import torch

            # Try to load from torch hub
            model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

            # Move model to device
            model.to(self.device)

            # Set inference mode
            model.eval()

            return model

        except Exception as e:
            logger.error(f"Error loading YOLOv5 model: {str(e)}")
            raise

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_name: Name of the model to unload

        Returns:
            bool: True if model was unloaded, False if not found
        """
        if model_name in self.models:
            del self.models[model_name]
            # Force garbage collection to free GPU memory
            import gc
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {model_name}")
            return True
        return False

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.

        Args:
            model_name: Name of the model

        Returns:
            Dict[str, Any]: Dictionary with model information
        """
        if model_name.startswith('yolov5'):
            size = model_name[6:]  # Extract size (s, m, l, x)
            return {
                'name': model_name,
                'type': 'yolov5',
                'size': size,
                'input_size': (640, 640),
                'loaded': model_name in self.models
            }
        elif model_name == 'background_subtraction':
            return {
                'name': model_name,
                'type': 'traditional',
                'description': 'Background subtraction using MOG2 algorithm',
                'loaded': model_name in self.models
            }
        elif model_name == 'hog_svm':
            return {
                'name': model_name,
                'type': 'traditional',
                'description': 'Histogram of Oriented Gradients with SVM classifier',
                'loaded': model_name in self.models
            }
        return {
            'name': model_name,
            'type': 'unknown',
            'loaded': model_name in self.models
        }

    def get_available_models(self) -> List[str]:
        """
        Get list of available models.

        Returns:
            List[str]: List of available model names
        """
        return ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'background_subtraction', 'hog_svm']

    def get_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded models.

        Returns:
            List[str]: List of loaded model names
        """
        return list(self.models.keys())


# Create a default instance
model_loader = ModelLoader()
