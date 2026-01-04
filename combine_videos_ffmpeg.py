#!/usr/bin/env python3
"""
Combine three model output videos using ffmpeg for better reliability.
"""

import subprocess
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    
    # Paths to high-quality output videos
    video1 = base_dir / 'src' / 'yolov8' / 'Results_high' / 'output_video.mp4'
    video2 = base_dir / 'src' / 'yolov9' / 'Results_high' / 'output_video.mp4'
    video3 = base_dir / 'src' / 'coco' / 'faster R_CNN' / 'Results_high' / 'output_video.mp4'
    
    output_path = base_dir / 'comparison_grid_high_quality.mp4'
    
    print("=" * 60)
    print("Combining Videos into Grid using FFmpeg")
    print("=" * 60)
    
    # Check if videos exist
    for v in [video1, video2, video3]:
        if not v.exists():
            print(f"❌ Error: Video not found: {v}")
            return
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: ffmpeg not found. Please install ffmpeg:")
        print("  brew install ffmpeg")
        return
    
    print(f"\nInput videos:")
    print(f"  1. YOLOv8: {video1}")
    print(f"  2. YOLOv9: {video2}")
    print(f"  3. Faster R-CNN: {video3}")
    print(f"\nOutput: {output_path}")
    print("\nProcessing...")
    
    # FFmpeg command to combine videos side by side with labels
    # Scale videos to 1280x720 for better compatibility
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', str(video1),
        '-i', str(video2),
        '-i', str(video3),
        '-filter_complex',
        '[0:v]scale=1280:720,setpts=PTS-STARTPTS,drawtext=text=\'YOLOv8\':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=20:box=1:boxcolor=black@0.5[v0];'
        '[1:v]scale=1280:720,setpts=PTS-STARTPTS,drawtext=text=\'YOLOv9\':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=20:box=1:boxcolor=black@0.5[v1];'
        '[2:v]scale=1280:720,setpts=PTS-STARTPTS,drawtext=text=\'Faster R-CNN\':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=20:box=1:boxcolor=black@0.5[v2];'
        '[v0][v1][v2]hstack=inputs=3',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output file
        str(output_path)
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        print("\n" + "=" * 60)
        print("✓ Video combination completed successfully!")
        print(f"✓ Output saved to: {output_path}")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: FFmpeg failed")
        print(f"Return code: {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr[:500]}")  # First 500 chars
        return

if __name__ == '__main__':
    main()

