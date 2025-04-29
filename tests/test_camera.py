"""
Tests for the camera module.
"""

import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch

from vigilance_system.video_acquisition.camera import Camera, RTSPCamera, HTTPCamera, WebcamCamera


class TestCamera:
    """Tests for the Camera base class and its implementations."""
    
    def test_camera_abstract_class(self):
        """Test that Camera is an abstract class that cannot be instantiated."""
        with pytest.raises(TypeError):
            Camera("test", "test_url")
    
    @patch('cv2.VideoCapture')
    def test_rtsp_camera_connect(self, mock_video_capture):
        """Test connecting to an RTSP camera."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_video_capture.return_value = mock_instance
        
        # Create camera and connect
        camera = RTSPCamera("test_rtsp", "rtsp://example.com/stream")
        result = camera.connect()
        
        # Verify
        assert result is True
        assert camera.is_connected is True
        mock_video_capture.assert_called_once()
        mock_instance.isOpened.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_rtsp_camera_connect_failure(self, mock_video_capture):
        """Test connecting to an RTSP camera that fails."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_video_capture.return_value = mock_instance
        
        # Create camera and connect
        camera = RTSPCamera("test_rtsp", "rtsp://example.com/stream")
        result = camera.connect()
        
        # Verify
        assert result is False
        assert camera.is_connected is False
    
    @patch('cv2.VideoCapture')
    def test_rtsp_camera_read_frame(self, mock_video_capture):
        """Test reading a frame from an RTSP camera."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_instance
        
        # Create camera and connect
        camera = RTSPCamera("test_rtsp", "rtsp://example.com/stream")
        camera.connect()
        
        # Read frame
        success, frame = camera.read_frame()
        
        # Verify
        assert success is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        mock_instance.read.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_rtsp_camera_read_frame_failure(self, mock_video_capture):
        """Test reading a frame from an RTSP camera that fails."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (False, None)
        mock_video_capture.return_value = mock_instance
        
        # Create camera and connect
        camera = RTSPCamera("test_rtsp", "rtsp://example.com/stream")
        camera.connect()
        
        # Read frame
        success, frame = camera.read_frame()
        
        # Verify
        assert success is False
        assert frame is None
    
    @patch('cv2.VideoCapture')
    def test_rtsp_camera_disconnect(self, mock_video_capture):
        """Test disconnecting from an RTSP camera."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_video_capture.return_value = mock_instance
        
        # Create camera and connect
        camera = RTSPCamera("test_rtsp", "rtsp://example.com/stream")
        camera.connect()
        
        # Disconnect
        camera.disconnect()
        
        # Verify
        assert camera.is_connected is False
        mock_instance.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_camera_start_stop(self, mock_video_capture):
        """Test starting and stopping a camera."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_instance
        
        # Create camera and start
        camera = RTSPCamera("test_rtsp", "rtsp://example.com/stream")
        camera.start()
        
        # Verify camera is running
        assert camera.is_running is True
        assert hasattr(camera, 'acquisition_thread')
        
        # Let the thread run for a bit
        time.sleep(0.1)
        
        # Stop camera
        camera.stop()
        
        # Verify camera is stopped
        assert camera.is_running is False
        mock_instance.release.assert_called_once()
