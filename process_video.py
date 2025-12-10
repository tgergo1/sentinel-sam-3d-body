# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Video Processing Script for SAM 3D Body

This script processes video files frame by frame, applies SAM 3D Body estimation,
and renders the results with synthetic model overlay to create an output video.

Assumes there is only one human per frame in the video.
"""

import argparse
import os
import tempfile
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample_together
from tqdm import tqdm


def extract_video_info(video_path):
    """Extract video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return fps, width, height, frame_count


def process_video_frames(video_path, estimator, bbox_thr=0.8, use_mask=False, max_frames=None):
    """
    Process video frames and yield processed frames.
    
    Args:
        video_path: Path to input video
        estimator: SAM3DBodyEstimator instance
        bbox_thr: Bounding box detection threshold
        use_mask: Whether to use mask-conditioned prediction
        max_frames: Maximum number of frames to process (None for all)
    
    Yields:
        Tuple of (frame_idx, rendered_frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_idx >= max_frames:
                break
            
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with SAM 3D Body
            outputs = estimator.process_one_image(
                frame_rgb,
                bbox_thr=bbox_thr,
                use_mask=use_mask,
            )
            
            # Visualize results - note that visualize_sample_together expects BGR input
            if len(outputs) > 0:
                # If multiple people detected, take only the first one (assumption: single human)
                if len(outputs) > 1:
                    print(f"Warning: Multiple humans detected in frame {frame_idx}, using the first one")
                    outputs = [outputs[0]]
                
                rend_img = visualize_sample_together(frame, outputs, estimator.faces)
            else:
                # No human detected, return original frame with all views as white
                print(f"Warning: No human detected in frame {frame_idx}")
                # Create a blank frame showing original + 3 white views
                white = np.ones_like(frame) * 255
                rend_img = np.concatenate([frame, white, white, white], axis=1)
            
            yield frame_idx, rend_img.astype(np.uint8)
            frame_idx += 1
            
    finally:
        cap.release()


def main(args):
    # Validate input video
    if not os.path.isfile(args.input_video):
        raise FileNotFoundError(f"Input video not found: {args.input_video}")
    
    # Set up output path
    if args.output_video == "":
        input_name = Path(args.input_video).stem
        output_folder = Path(args.output_folder or "./output")
        output_folder.mkdir(parents=True, exist_ok=True)
        output_video = output_folder / f"{input_name}_3d_body.mp4"
    else:
        output_video = Path(args.output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Input video: {args.input_video}")
    print(f"Output video: {output_video}")
    
    # Extract video info
    fps, width, height, frame_count = extract_video_info(args.input_video)
    print(f"Video info: {width}x{height} @ {fps} fps, {frame_count} frames")
    
    # Use command-line args or environment variables for model paths
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")
    
    # Initialize SAM 3D Body model
    print("Loading SAM 3D Body model...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )
    
    # Initialize optional modules
    human_detector, human_segmentor, fov_estimator = None, None, None
    
    if args.detector_name:
        print(f"Loading human detector: {args.detector_name}...")
        from tools.build_detector import HumanDetector
        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    
    if len(segmentor_path):
        print(f"Loading human segmentor: {args.segmentor_name}...")
        from tools.build_sam import HumanSegmentor
        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    
    if args.fov_name:
        print(f"Loading FOV estimator: {args.fov_name}...")
        from tools.build_fov_estimator import FOVEstimator
        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)
    
    # Create estimator
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )
    
    # The output width is 4x the input width (original + keypoints + mesh + side view)
    output_width = width * 4
    output_height = height
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (output_width, output_height)
    )
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_video}")
    
    print("Processing video frames...")
    
    # Process frames
    max_frames = args.max_frames if args.max_frames > 0 else None
    total_frames = min(frame_count, max_frames) if max_frames else frame_count
    
    try:
        for frame_idx, rendered_frame in tqdm(
            process_video_frames(
                args.input_video,
                estimator,
                bbox_thr=args.bbox_thresh,
                use_mask=args.use_mask,
                max_frames=max_frames
            ),
            total=total_frames,
            desc="Processing frames"
        ):
            out.write(rendered_frame)
            
            # Optionally save individual frames
            if args.save_frames:
                frames_dir = output_video.parent / f"{output_video.stem}_frames"
                frames_dir.mkdir(exist_ok=True)
                frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), rendered_frame)
    
    finally:
        out.release()
    
    print(f"\nVideo processing complete!")
    print(f"Output saved to: {output_video}")
    
    if args.save_frames:
        print(f"Individual frames saved to: {frames_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Video Processing - Process video and render 3D body estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with checkpoint
  python process_video.py --input_video input.mp4 --checkpoint_path ./checkpoints/model.ckpt

  # Specify output location
  python process_video.py --input_video input.mp4 --output_video output/result.mp4 --checkpoint_path ./checkpoints/model.ckpt

  # Process only first 100 frames for testing
  python process_video.py --input_video input.mp4 --checkpoint_path ./checkpoints/model.ckpt --max_frames 100

  # Save individual frames
  python process_video.py --input_video input.mp4 --checkpoint_path ./checkpoints/model.ckpt --save_frames

Environment Variables:
  SAM3D_MHR_PATH: Path to MHR asset
  SAM3D_DETECTOR_PATH: Path to human detection model folder
  SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
  SAM3D_FOV_PATH: Path to FOV estimation model folder
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--input_video",
        required=True,
        type=str,
        help="Path to input video file",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_video",
        default="",
        type=str,
        help="Path to output video file (default: ./output/<input_name>_3d_body.mp4)",
    )
    parser.add_argument(
        "--output_folder",
        default="./output",
        type=str,
        help="Output folder when output_video is not specified (default: ./output)",
    )
    
    # Model arguments
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model (default: vitdet)",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model (default: sam2)",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model (default: moge2)",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to FOV estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MHR assets folder (or set SAM3D_MHR_PATH)",
    )
    
    # Processing arguments
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold (default: 0.8)",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (mask automatically generated from bbox)",
    )
    parser.add_argument(
        "--max_frames",
        default=0,
        type=int,
        help="Maximum number of frames to process (0 for all frames, useful for testing)",
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        default=False,
        help="Save individual processed frames as images",
    )
    
    args = parser.parse_args()
    main(args)
