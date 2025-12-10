# Video Processing with SAM 3D Body

This guide explains how to use the `process_video.py` script to perform 3D body estimation on video files and generate result videos with synthetic model overlays.

## Overview

The `process_video.py` script:
- Processes video files frame by frame
- Detects and estimates 3D body pose for each frame
- Renders visualization showing:
  - Original frame
  - 2D keypoints overlay
  - 3D mesh overlay on original
  - 3D mesh side view
- Saves the result as a new video file
- Assumes there is only one human per frame in the video

## Quick Start

### Basic Usage

```bash
python process_video.py \
    --input_video path/to/your/video.mp4 \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

### Complete Example with All Options

```bash
python process_video.py \
    --input_video input.mp4 \
    --output_video output/result_3d_body.mp4 \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
    --detector_name vitdet \
    --segmentor_name sam2 \
    --fov_name moge2 \
    --bbox_thresh 0.8 \
    --use_mask
```

## Prerequisites

1. **Install SAM 3D Body**: Follow the installation instructions in [INSTALL.md](INSTALL.md)

2. **Download Model Checkpoints**: 
   ```bash
   # Download the model from HuggingFace (requires access approval)
   huggingface-cli download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
   ```

3. **Prepare Input Video**: Ensure your video file is in a supported format (mp4, avi, mov, etc.)

## Command Line Arguments

### Required Arguments

- `--input_video`: Path to the input video file
- `--checkpoint_path`: Path to SAM 3D Body model checkpoint file

### Output Arguments

- `--output_video`: Path for output video (default: `./output/<input_name>_3d_body.mp4`)
- `--output_folder`: Output folder when output_video is not specified (default: `./output`)
- `--save_frames`: Save individual processed frames as images

### Model Configuration

- `--detector_name`: Human detection model (default: `vitdet`)
- `--segmentor_name`: Human segmentation model (default: `sam2`)
- `--fov_name`: FOV estimation model (default: `moge2`)
- `--detector_path`: Path to detection model folder (or set `SAM3D_DETECTOR_PATH` env var)
- `--segmentor_path`: Path to segmentation model folder (or set `SAM3D_SEGMENTOR_PATH` env var)
- `--fov_path`: Path to FOV estimation model folder (or set `SAM3D_FOV_PATH` env var)
- `--mhr_path`: Path to MHR assets folder (or set `SAM3D_MHR_PATH` env var)

### Processing Options

- `--bbox_thresh`: Bounding box detection threshold (default: `0.8`)
- `--use_mask`: Enable mask-conditioned prediction
- `--max_frames`: Maximum number of frames to process (default: `0` for all frames)

## Usage Examples

### 1. Process Full Video

Process an entire video with default settings:

```bash
python process_video.py \
    --input_video dance_performance.mp4 \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt
```

### 2. Test with Limited Frames

Process only the first 100 frames for quick testing:

```bash
python process_video.py \
    --input_video long_video.mp4 \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --max_frames 100
```

### 3. Save Individual Frames

Save each processed frame as a separate image file:

```bash
python process_video.py \
    --input_video workout.mp4 \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --save_frames
```

This creates a folder with frames named `frame_000000.jpg`, `frame_000001.jpg`, etc.

### 4. Specify Custom Output Location

```bash
python process_video.py \
    --input_video sports_clip.mp4 \
    --output_video results/sports_analysis.mp4 \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt
```

### 5. Enable Mask-Conditioned Prediction

For better accuracy, enable mask-based prediction:

```bash
python process_video.py \
    --input_video yoga_session.mp4 \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --use_mask
```

## Output Format

The output video will have 4x the width of the input video, showing:

1. **Column 1**: Original input frame
2. **Column 2**: 2D keypoints overlay with bounding box
3. **Column 3**: 3D mesh overlay on the original frame
4. **Column 4**: 3D mesh side view on white background

Example output layout:
```
[Original] [2D Keypoints] [3D Mesh Overlay] [3D Side View]
```

## Performance Considerations

- **GPU Recommended**: Processing is significantly faster with CUDA-enabled GPU
- **Processing Time**: Depends on:
  - Video resolution
  - Number of frames
  - Available GPU/CPU resources
  - Whether mask prediction is enabled
- **Memory Usage**: Each frame requires model inference; GPU memory requirements scale with resolution

### Typical Processing Times (on NVIDIA A100)

- 720p video (30 fps): ~0.5-1 seconds per frame
- 1080p video (30 fps): ~1-2 seconds per frame
- 4K video (30 fps): ~3-5 seconds per frame

## Troubleshooting

### No Human Detected in Frame

If humans are not detected in some frames, the script will:
- Print a warning message
- Output the original frame with blank visualization panels
- Continue processing remaining frames

To improve detection:
- Lower `--bbox_thresh` value (e.g., `0.5`)
- Ensure good video quality and lighting
- Check that the person is clearly visible in the frame

### Multiple Humans Detected

If multiple humans are detected (despite the single-human assumption):
- The script uses the first detected person
- A warning is printed
- Consider pre-processing the video to focus on one person

### Out of Memory Errors

If you encounter GPU memory errors:
- Reduce video resolution before processing
- Process fewer frames at a time using `--max_frames`
- Close other GPU-intensive applications
- Use a smaller model if available

### Slow Processing

To speed up processing:
- Use a GPU instead of CPU
- Process at lower resolution
- Disable `--use_mask` if not needed
- Use fewer detector/segmentor models

## Environment Variables

Instead of passing paths as arguments, you can set environment variables:

```bash
export SAM3D_MHR_PATH="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
export SAM3D_DETECTOR_PATH="./checkpoints/detector"
export SAM3D_SEGMENTOR_PATH="./checkpoints/sam2"
export SAM3D_FOV_PATH="./checkpoints/moge2"

python process_video.py \
    --input_video video.mp4 \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt
```

## Integration with Other Tools

### Extract Specific Frames

Use FFmpeg to extract specific segments before processing:

```bash
# Extract 10 seconds starting at 30 seconds
ffmpeg -i input.mp4 -ss 00:00:30 -t 00:00:10 segment.mp4

# Process the segment
python process_video.py --input_video segment.mp4 --checkpoint_path ./checkpoints/model.ckpt
```

### Convert Video Format

Convert your video to a compatible format:

```bash
ffmpeg -i input.avi -c:v libx264 -preset fast -crf 22 input.mp4
```

### Combine with Other Analysis

The script can be used as part of a larger pipeline:

```python
# Example: Custom processing pipeline
from process_video import extract_video_info, process_video_frames
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

# Load model
model, cfg = load_sam_3d_body("checkpoint.ckpt")
estimator = SAM3DBodyEstimator(model, cfg)

# Process and extract pose data
for frame_idx, rendered_frame in process_video_frames("input.mp4", estimator):
    # Your custom analysis here
    pass
```

## Citation

If you use this video processing script in your research, please cite SAM 3D Body:

```bibtex
@article{yang2025sam3dbody,
  title={SAM 3D Body: Robust Full-Body Human Mesh Recovery},
  author={Yang, Xitong and Kukreja, Devansh and Pinkus, Don and others},
  journal={arXiv preprint},
  year={2025}
}
```

## Support

For issues or questions:
- Check the main [README.md](README.md)
- Review [INSTALL.md](INSTALL.md) for setup issues
- Open an issue on the GitHub repository
