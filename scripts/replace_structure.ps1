# PowerShell script to replace the old structure with the new one

# Create a backup directory
$backupDir = "backup_old_structure"
New-Item -Path $backupDir -ItemType Directory -Force | Out-Null
Write-Host "Created backup directory: $backupDir"

# Move the old files to backup
Write-Host "Moving old files to backup..."

# List of directories and files to back up
$itemsToBackup = @(
    "vigilance_system",
    "tests",
    "examples",
    "config.yaml",
    "requirements.txt",
    "setup.py",
    "setup.sh",
    "setup.bat",
    "download_sample_videos.py",
    "README.md",
    "COMPONENTS_GUIDE.md",
    "DEPENDENCIES.md"
)

foreach ($item in $itemsToBackup) {
    if (Test-Path $item) {
        Move-Item -Path $item -Destination "$backupDir/$item" -Force
        Write-Host "Moved $item to backup"
    }
}

# Move the new structure to the root
Write-Host "Moving new structure to root..."
$items = Get-ChildItem -Path "clean_structure" -Force
foreach ($item in $items) {
    Move-Item -Path $item.FullName -Destination "." -Force
    Write-Host "Moved $($item.Name) to root"
}

# Remove the empty clean_structure directory
Remove-Item -Path "clean_structure" -Force
Write-Host "Removed empty clean_structure directory"

Write-Host "Structure replacement complete!"
Write-Host "The old structure has been backed up to $backupDir"
