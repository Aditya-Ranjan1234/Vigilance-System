# PowerShell script to reorganize the Vigilance System directory structure

# Create a clean directory structure
$cleanDir = "clean_structure"
New-Item -Path $cleanDir -ItemType Directory -Force | Out-Null
Write-Host "Created directory: $cleanDir"

# Create the main package directory
$mainPackageDir = "$cleanDir/vigilance_system"
New-Item -Path $mainPackageDir -ItemType Directory -Force | Out-Null
Write-Host "Created directory: $mainPackageDir"

# Create subdirectories
$subdirs = @(
    "alert",
    "dashboard",
    "dashboard/templates",
    "dashboard/static",
    "dashboard/static/css",
    "dashboard/static/js",
    "dashboard/static/img",
    "dashboard/static/audio",
    "detection",
    "preprocessing",
    "utils",
    "video_acquisition",
    "videos"
)

foreach ($dir in $subdirs) {
    $path = "$mainPackageDir/$dir"
    New-Item -Path $path -ItemType Directory -Force | Out-Null
    Write-Host "Created directory: $path"
}

# Create other top-level directories
$topDirs = @(
    "tests",
    "examples"
)

foreach ($dir in $topDirs) {
    $path = "$cleanDir/$dir"
    New-Item -Path $path -ItemType Directory -Force | Out-Null
    Write-Host "Created directory: $path"
}

# Create __init__.py files
$initDirs = @(
    "",
    "alert",
    "dashboard",
    "detection",
    "preprocessing",
    "utils",
    "video_acquisition"
)

foreach ($dir in $initDirs) {
    $path = "$mainPackageDir/$dir/__init__.py"
    New-Item -Path $path -ItemType File -Force | Out-Null
    Write-Host "Created file: $path"
}

# Find and copy files
Write-Host "Searching for files to copy..."

# Function to find and copy a file
function Find-And-Copy {
    param (
        [string]$pattern,
        [string]$destination
    )
    
    $files = Get-ChildItem -Path . -Recurse -File -Filter $pattern | Where-Object { $_.FullName -notlike "*clean_structure*" -and $_.FullName -notlike "*backup*" }
    
    if ($files.Count -gt 0) {
        $sourceFile = $files[0].FullName
        $destPath = "$cleanDir/$destination"
        
        # Create the destination directory if it doesn't exist
        $destDir = Split-Path -Path $destPath -Parent
        if (-not (Test-Path -Path $destDir)) {
            New-Item -Path $destDir -ItemType Directory -Force | Out-Null
        }
        
        # Copy the file
        Copy-Item -Path $sourceFile -Destination $destPath -Force
        Write-Host "Copied: $sourceFile -> $destPath"
        return $true
    }
    
    Write-Host "Warning: Could not find file matching pattern: $pattern"
    return $false
}

# Copy main package files
Find-And-Copy -pattern "__main__.py" -destination "vigilance_system/__main__.py"

# Copy alert module files
Find-And-Copy -pattern "decision_maker.py" -destination "vigilance_system/alert/decision_maker.py"
Find-And-Copy -pattern "notifier.py" -destination "vigilance_system/alert/notifier.py"

# Copy dashboard module files
Find-And-Copy -pattern "app.py" -destination "vigilance_system/dashboard/app.py"
Find-And-Copy -pattern "index.html" -destination "vigilance_system/dashboard/templates/index.html"
Find-And-Copy -pattern "login.html" -destination "vigilance_system/dashboard/templates/login.html"
Find-And-Copy -pattern "dashboard.css" -destination "vigilance_system/dashboard/static/css/dashboard.css"
Find-And-Copy -pattern "dashboard.js" -destination "vigilance_system/dashboard/static/js/dashboard.js"

# Copy detection module files
Find-And-Copy -pattern "model_loader.py" -destination "vigilance_system/detection/model_loader.py"
Find-And-Copy -pattern "object_detector.py" -destination "vigilance_system/detection/object_detector.py"

# Copy preprocessing module files
Find-And-Copy -pattern "frame_extractor.py" -destination "vigilance_system/preprocessing/frame_extractor.py"
Find-And-Copy -pattern "video_stabilizer.py" -destination "vigilance_system/preprocessing/video_stabilizer.py"

# Copy utils module files
Find-And-Copy -pattern "config.py" -destination "vigilance_system/utils/config.py"
Find-And-Copy -pattern "logger.py" -destination "vigilance_system/utils/logger.py"

# Copy video acquisition module files
Find-And-Copy -pattern "camera.py" -destination "vigilance_system/video_acquisition/camera.py"
Find-And-Copy -pattern "stream_manager.py" -destination "vigilance_system/video_acquisition/stream_manager.py"

# Copy videos directory files
Find-And-Copy -pattern "videos\README.md" -destination "vigilance_system/videos/README.md"

# Copy test files
Find-And-Copy -pattern "test_config.py" -destination "tests/test_config.py"
Find-And-Copy -pattern "test_camera.py" -destination "tests/test_camera.py"
Find-And-Copy -pattern "tests\README.md" -destination "tests/README.md"

# Copy example files
Find-And-Copy -pattern "simple_detection.py" -destination "examples/simple_detection.py"
Find-And-Copy -pattern "examples\README.md" -destination "examples/README.md"

# Copy root files
Find-And-Copy -pattern "config.yaml" -destination "config.yaml"
Find-And-Copy -pattern "requirements.txt" -destination "requirements.txt"
Find-And-Copy -pattern "setup.py" -destination "setup.py"
Find-And-Copy -pattern "setup.sh" -destination "setup.sh"
Find-And-Copy -pattern "setup.bat" -destination "setup.bat"
Find-And-Copy -pattern "download_sample_videos.py" -destination "download_sample_videos.py"
Find-And-Copy -pattern "README.md" -destination "README.md"
Find-And-Copy -pattern "COMPONENTS_GUIDE.md" -destination "COMPONENTS_GUIDE.md"
Find-And-Copy -pattern "DEPENDENCIES.md" -destination "DEPENDENCIES.md"

# Copy video files
$videoExtensions = @(".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")
foreach ($ext in $videoExtensions) {
    $videoFiles = Get-ChildItem -Path . -Recurse -File -Filter "*$ext" | Where-Object { $_.FullName -notlike "*clean_structure*" -and $_.FullName -notlike "*backup*" }
    foreach ($file in $videoFiles) {
        $destPath = "$cleanDir/vigilance_system/videos/$($file.Name)"
        Copy-Item -Path $file.FullName -Destination $destPath -Force
        Write-Host "Copied video: $($file.FullName) -> $destPath"
    }
}

Write-Host "Reorganization complete! The clean structure is in the '$cleanDir' directory."
Write-Host "Please review the files and then replace the old structure with the new one."
