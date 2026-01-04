#!/usr/bin/env python3
"""
Combine three model output videos (YOLOv8, YOLOv9, Faster R-CNN) 
into a side-by-side grid for comparison.
"""

import cv2
import numpy as np
from pathlib import Path

def get_video_properties(video_path):
    """Get video properties (width, height, fps, frame count)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    return {'width': width, 'height': height, 'fps': fps, 'frame_count': frame_count}

def resize_frame(frame, target_width, target_height):
    """Resize frame to target dimensions while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    
    # Calculate scaling factor
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create black canvas and center the resized frame
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def add_label(frame, text, position='top'):
    """Add model label to frame."""
    h, w = frame.shape[:2]
    
    # Create a semi-transparent overlay
    overlay = frame.copy()
    
    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position: top center
    if position == 'top':
        x = (w - text_width) // 2
        y = text_height + 20
        # Draw background rectangle
        cv2.rectangle(overlay, 
                     (x - 10, y - text_height - 10), 
                     (x + text_width + 10, y + baseline + 10),
                     (0, 0, 0), -1)
        # Draw text
        cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    # Blend overlay with original frame
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return frame

def combine_videos_grid(video_paths, output_path, labels=None):
    """Combine multiple videos into a side-by-side grid."""
    if labels is None:
        labels = [f"Video {i+1}" for i in range(len(video_paths))]
    
    # Open all video files
    caps = []
    properties = []
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Error: Could not open {video_path}")
            return False
        caps.append(cap)
        
        props = get_video_properties(video_path)
        properties.append(props)
        print(f"✓ Loaded: {video_path.name}")
        print(f"  Resolution: {props['width']}x{props['height']}, FPS: {props['fps']:.2f}, Frames: {props['frame_count']}")
    
    # Use the minimum frame count and fps
    min_frames = min(p['frame_count'] for p in properties)
    min_fps = min(p['fps'] for p in properties)
    
    # Determine target dimensions (scale down if too large for better compatibility)
    target_height = min(p['height'] for p in properties)
    target_width = min(p['width'] for p in properties)
    
    # Scale down if dimensions are too large (max width 1920 per video for 3 videos = 5760 total)
    max_per_video_width = 1280  # Scale down to 1280 per video for better compatibility
    if target_width > max_per_video_width:
        scale = max_per_video_width / target_width
        target_width = max_per_video_width
        target_height = int(target_height * scale)
    
    # For 3 videos side by side
    num_videos = len(video_paths)
    grid_width = target_width * num_videos
    grid_height = target_height + 60  # Extra space for labels
    
    print(f"\nGrid dimensions: {grid_width}x{grid_height}")
    print(f"Target frame size per video: {target_width}x{target_height}")
    print(f"Total frames to process: {min_frames}")
    print(f"Output FPS: {min_fps:.2f}")
    
    # Create video writer - try different codecs for compatibility
    codecs_to_try = [
        ('avc1', '.mp4'),  # H.264
        ('mp4v', '.mp4'),  # MPEG-4
        ('XVID', '.avi'),  # XVID
    ]
    
    out = None
    final_output_path = output_path
    
    for codec, ext in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_path = output_path.with_suffix(ext)
            test_writer = cv2.VideoWriter(str(test_path), fourcc, min_fps, (grid_width, grid_height))
            if test_writer.isOpened():
                out = test_writer
                final_output_path = test_path
                print(f"✓ Using codec: {codec}")
                break
            test_writer.release()
        except:
            continue
    
    if out is None or not out.isOpened():
        print(f"❌ Error: Could not create video writer with any codec")
        return False
    
    frame_num = 0
    
    print("\nProcessing frames...")
    while True:
        frames = []
        all_read = True
        
        # Read one frame from each video
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                all_read = False
                break
            
            # Resize frame to target dimensions
            resized = resize_frame(frame, target_width, target_height)
            
            # Add label
            labeled = add_label(resized, labels[i], position='top')
            frames.append(labeled)
        
        if not all_read:
            break
        
        # Combine frames horizontally
        combined_frame = np.hstack(frames)
        
        # Write combined frame
        out.write(combined_frame)
        frame_num += 1
        
        if frame_num % 50 == 0:
            progress = (frame_num / min_frames) * 100
            print(f"Progress: {frame_num}/{min_frames} ({progress:.1f}%)", end='\r')
    
    # Release everything
    for cap in caps:
        cap.release()
    out.release()
    
    print(f"\n✓ Combined video saved to: {final_output_path}")
    print(f"✓ Total frames: {frame_num}")
    
    return True

def main():
    base_dir = Path(__file__).parent
    
    # Paths to high-quality output videos
    video_paths = [
        base_dir / 'src' / 'yolov8' / 'Results_high' / 'output_video.mp4',
        base_dir / 'src' / 'yolov9' / 'Results_high' / 'output_video.mp4',
        base_dir / 'src' / 'coco' / 'faster R_CNN' / 'Results_high' / 'output_video.mp4'
    ]
    
    # Model labels
    labels = ['YOLOv8', 'YOLOv9', 'Faster R-CNN']
    
    # Output path
    output_path = base_dir / 'comparison_grid_high_quality.mp4'
    
    print("=" * 60)
    print("Combining Videos into Grid")
    print("=" * 60)
    print(f"Input videos:")
    for i, (path, label) in enumerate(zip(video_paths, labels), 1):
        print(f"  {i}. {label}: {path}")
    print(f"\nOutput: {output_path}")
    print("=" * 60)
    
    # Check if all videos exist
    missing = [p for p in video_paths if not p.exists()]
    if missing:
        print("\n❌ Error: Missing video files:")
        for p in missing:
            print(f"  - {p}")
        return
    
    # Combine videos
    success = combine_videos_grid(video_paths, output_path, labels)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Video combination completed successfully!")
        print(f"✓ Output saved to: {output_path}")
        print("=" * 60)
    else:
        print("\n❌ Error: Failed to combine videos")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

