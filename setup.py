from setuptools import setup, find_packages

setup(
    name="vigilance_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "numpy>=1.22.0",
        "opencv-python>=4.5.5",
        "opencv-contrib-python>=4.5.5",
        "pillow>=9.0.0",

        # Deep learning
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "ultralytics>=8.0.0",  # For YOLOv5 models

        # Video streaming
        "av>=9.0.0",  # PyAV for video decoding
        "rtsp>=1.1.8",  # RTSP client

        # Web dashboard
        "flask>=2.0.0",
        "flask-socketio>=5.1.0",
        "bidict>=0.22.0",  # Required by flask-socketio
        "python-engineio>=4.3.0",  # Required by flask-socketio
        "python-socketio>=5.7.0",  # Required by flask-socketio

        # Utilities
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
        "loguru>=0.6.0",
        "requests>=2.28.0",  # For HTTP requests
        "tqdm>=4.64.0",      # For progress bars
    ],
    extras_require={
        "notifications": [
            "twilio>=7.0.0",  # For SMS notifications
            "boto3>=1.26.0",  # For AWS SNS notifications
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A camera-only vigilance system for security monitoring",
    keywords="security, camera, monitoring, computer vision",
    python_requires=">=3.10",
)
