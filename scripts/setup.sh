#!/bin/bash

# Setup script for Vigilance System

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 10 ]); then
    echo "Error: Python 3.10 or higher is required. Found Python $python_version"
    exit 1
fi

echo "Using Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
if [ -f "venv/Scripts/activate" ]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install package in development mode with all extras
echo "Installing package in development mode..."
pip install -e .[notifications,dev]

# Create necessary directories
mkdir -p logs
mkdir -p alerts
mkdir -p models

echo "Setup complete!"
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate  # On Linux/Mac"
echo "  venv\\Scripts\\activate    # On Windows"
echo ""
echo "To start the system, run:"
echo "  python -m vigilance_system"
