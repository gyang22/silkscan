#!/usr/bin/env python3
import argparse
import os
import sys

# Add the current directory to sys.path so silkscan can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import silkscan
from silkscan import SweepProcessor

def main():
    parser = argparse.ArgumentParser(description="Process a single video file to create a 3D point cloud using the Silkscan pipeline.")
    parser.add_argument("video_path", type=str, help="Path to the input video file (e.g., scan.mp4).")
    parser.add_argument("-o", "--output", type=str, default="output.pcd", help="Path to save the resulting .pcd file. Default is 'output.pcd'.")
    parser.add_argument("--mm-per-frame", type=float, default=0.1, help="Millimeters traveled per frame. Default is 0.1.")
    parser.add_argument("--pixels-per-mm", type=float, default=3.4, help="Pixels per millimeter scale. Default is 3.4.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame to start processing from. Default is 0.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process. Default is all frames.")
    
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: The video file '{args.video_path}' was not found.")
        sys.exit(1)

    print(f"Processing video: {args.video_path}")
    print("Using highly optimized default settings (Steger Line Detection)...")

    # Set up optimized config
    config = silkscan.Config(
        method='steger', 
        intensity_threshold=0.05, 
        strength_threshold=0.002,
        high_intensity_threshold=0.15, 
        high_strength_threshold=0.015,
        sigma=0.5, 
        persistence_min_frames=5, 
        temporal_spatial_radius=2.0,
        spatial_2d_min_length_px=20, 
        temporal_stack_frames=5
    )

    processor = SweepProcessor(config)

    # Basic manifest and sweep info
    manifest = {
        'pixels_per_mm': args.pixels_per_mm,
        'mm_per_frame': args.mm_per_frame,
        'crop_box_mm': {} # No crop by default
    }
    
    sweep_info = {
        'id': 'sweep1',
        'video_path': args.video_path,
        'rotation_angle_deg': 0.0
    }

    try:
        print("Starting processing. This may take a moment depending on the video length...")
        pcd_data = processor.process_video(
            args.video_path, 
            sweep_info, 
            manifest, 
            start_frame=args.start_frame, 
            max_frames=args.max_frames,
            override_mask_poly=None # Process the whole image
        )

        print(f"Processing complete! Found {len(pcd_data)} points.")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        silkscan.save_pcd(pcd_data, args.output)
        print(f"Successfully saved 3D point cloud to: {args.output}")

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        print("Please check your video file and parameters.")
        sys.exit(1)

if __name__ == "__main__":
    main()
