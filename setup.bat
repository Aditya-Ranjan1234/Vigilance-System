@echo off
REM Setup script for Vigilance System on Windows

REM Check Python version
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set python_version=%%I
for /f "tokens=1 delims=." %%I in ("%python_version%") do set python_major=%%I
for /f "tokens=2 delims=." %%I in ("%python_version%") do set python_minor=%%I

if %python_major% LSS 3 (
    echo Error: Python 3.10 or higher is required. Found Python %python_version%
    exit /b 1
)

if %python_major% EQU 3 (
    if %python_minor% LSS 10 (
        echo Error: Python 3.10 or higher is required. Found Python %python_version%
        exit /b 1
    )
)

echo Using Python %python_version%

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Install package in development mode with all extras
echo Installing package in development mode...
pip install -e .[notifications,dev]

REM Create necessary directories
if not exist logs mkdir logs
if not exist alerts mkdir alerts
if not exist models mkdir models

echo Setup complete!
echo To activate the virtual environment, run:
echo   venv\Scripts\activate
echo.
echo To start the system, run:
echo   python -m vigilance_system
