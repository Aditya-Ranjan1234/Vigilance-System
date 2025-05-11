# Vigilance System Dependencies

This document explains the dependencies required for the Vigilance System and how to install them.

## Core Dependencies

The Vigilance System relies on several Python packages to function properly. These are categorized by their purpose:

### Core Dependencies
- **numpy**: For numerical operations and array handling
- **opencv-python**: For image processing and computer vision
- **opencv-contrib-python**: For advanced OpenCV features like video stabilization
- **pillow**: For image handling

### Deep Learning
- **torch**: PyTorch for deep learning
- **torchvision**: Computer vision utilities for PyTorch
- **ultralytics**: For YOLOv5 object detection models

### Video Streaming
- **av**: PyAV for video decoding
- **rtsp**: RTSP client for streaming

### Web Dashboard
- **flask**: Web framework for the dashboard
- **flask-socketio**: Real-time communication for the dashboard
- **bidict**: Required by flask-socketio
- **python-engineio**: Required by flask-socketio
- **python-socketio**: Required by flask-socketio

### Notification Services (Optional)
- **twilio**: For SMS notifications
- **boto3**: For AWS SNS notifications

### Utilities
- **pyyaml**: For YAML configuration files
- **python-dotenv**: For environment variable management
- **loguru**: For enhanced logging
- **requests**: For HTTP requests

### Testing
- **pytest**: For unit testing
- **pytest-cov**: For test coverage reports

## Installation

### Using the Setup Scripts

The easiest way to install all dependencies is to use the provided setup scripts:

#### Windows
```bash
setup.bat
```

#### Linux/Mac
```bash
chmod +x setup.sh
./setup.sh
```

These scripts will:
1. Create a virtual environment
2. Install all dependencies from requirements.txt
3. Install the package in development mode with all extras

### Manual Installation

If you prefer to install dependencies manually:

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install core dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Install the package with all extras:
   ```bash
   pip install -e .[notifications,dev]
   ```

### Installing Specific Extras

If you only need certain features, you can install specific extras:

```bash
# For notification features only
pip install -e .[notifications]

# For development and testing only
pip install -e .[dev]
```

## System Dependencies

Some Python packages require system-level dependencies:

### OpenCV Dependencies (Linux)
```bash
sudo apt-get update
sudo apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx
```

### PyTorch with CUDA (Optional, for GPU acceleration)
If you want to use GPU acceleration, make sure you have:
1. A CUDA-compatible GPU
2. CUDA Toolkit installed (version compatible with your PyTorch version)
3. cuDNN installed

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'cv2'**
   - Solution: `pip install opencv-python opencv-contrib-python`

2. **ImportError: DLL load failed while importing _C**
   - Solution: Make sure you have the Visual C++ Redistributable installed (Windows)

3. **No CUDA GPUs are available**
   - Solution: Check that your GPU is CUDA-compatible and that you have the correct CUDA version installed

4. **Error loading YOLOv5 model**
   - Solution: `pip install ultralytics`

### Checking Installed Packages

To check which packages are installed in your environment:
```bash
pip list
```

To check if a specific package is installed:
```bash
pip show package_name
```

## Updating Dependencies

To update all dependencies to their latest versions:
```bash
pip install --upgrade -r requirements.txt
```

To update a specific package:
```bash
pip install --upgrade package_name
```
