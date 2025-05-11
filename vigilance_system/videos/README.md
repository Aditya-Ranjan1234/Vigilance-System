# Video Files for Vigilance System

This directory is used to store video files that can be used by the Vigilance System when no cameras are configured.

## Supported Video Formats

The system supports the following video formats:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)

## How to Use

1. Place your surveillance video files in this directory
2. The system will automatically detect and use these videos if no cameras are configured in `config.yaml`
3. Each video file will be treated as a separate camera source
4. Videos will loop by default

## Example

If you have the following files in this directory:
- `store_entrance.mp4`
- `parking_lot.mp4`

The system will create two virtual cameras named "store_entrance" and "parking_lot" that will continuously play these videos.

## Finding Sample Videos

If you don't have your own surveillance videos, you can find sample videos from various sources:

1. Public datasets:
   - [VIRAT Video Dataset](https://viratdata.org/)
   - [PETS Dataset](https://www.cvg.reading.ac.uk/PETS2009/a.html)

2. Stock video websites:
   - [Pexels](https://www.pexels.com/videos/)
   - [Pixabay](https://pixabay.com/videos/)

3. YouTube (download with permission)

## Configuration

If you want more control over how video files are used, you can explicitly configure them in the `config.yaml` file:

```yaml
cameras:
  - name: custom_name
    url: videos/your_video.mp4
    type: video
    fps: null  # null means use the video's native FPS
    loop: true  # Loop the video when it reaches the end
```
