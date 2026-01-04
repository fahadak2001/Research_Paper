"""
Unified Faster R-CNN Testing Script
Combines video processing and metrics generation
Usage: python faster_rcnn.py --video <path> [--model <path>] [--conf <0.5>] [--classes <6>]
"""

import argparse
import os
import json
import time
import cv2
import ssl
import warnings

# CRITICAL: Disable SSL verification BEFORE importing torchvision
# This prevents certificate errors when torchvision tries to download models
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torchvision.transforms.functional as F
from pathlib import Path
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Class names (matching YOLO format)
CLASS_NAMES = ['Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck']

def parse_args():
    parser = argparse.ArgumentParser(description='Faster R-CNN Testing and Evaluation')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default=None, help='Path to model file (default: Model/faster_rcnn_best.pth)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--classes', type=int, default=6, help='Number of classes including background (default: 6)')
    parser.add_argument('--output-dir', type=str, default='Results', help='Output directory (default: Results)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set default model path if not provided
    if args.model is None:
        script_dir = Path(__file__).parent
        default_model = script_dir / 'Model' / 'faster_rcnn_best.pth'
        if default_model.exists():
            args.model = str(default_model)
        else:
            raise FileNotFoundError(f"Model not found. Please provide --model path or place model at {default_model}")
    
    # Validate inputs
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Faster R-CNN Testing and Evaluation")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print(f"Confidence: {args.conf}")
    print(f"Classes: {args.classes}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)
    
    device = torch.device("cpu")
    
    # Load model
    print("\n[1/4] Loading model...")
    start_time = time.time()
    
    # Create model architecture without pretrained weights
    # The full model (including backbone) will be loaded from checkpoint
    try:
        # Try new API (torchvision >= 0.13) - weights=None prevents downloads
        model = fasterrcnn_resnet50_fpn(weights=None)
    except (TypeError, ValueError):
        # Fallback for older torchvision versions
        model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.classes)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    print(f"✓ Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Open video
    print("\n[2/4] Processing video...")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open video")
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video - standardized name
    output_video_path = output_dir / 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))
    
    print(f"  - Total frames: {total_frames}")
    print(f"  - Resolution: {w}x{h}")
    print(f"  - FPS: {fps:.2f}")
    
    # Process frames
    frame_detections = {}
    total_detections = {name: 0 for name in CLASS_NAMES}
    frame_num = 0
    video_start = time.time()
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB tensor
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = F.to_tensor(rgb).unsqueeze(0).to(device)
            
            # Run inference
            preds = model(img)[0]
            
            # Count detections per class in this frame
            class_counts = {}
            
            for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
                if score < args.conf:
                    continue
                
                label_idx = int(label.item())
                
                # Skip background (label 0)
                if label_idx == 0:
                    continue
                
                # Map to class name (label 1 = Bus, label 2 = Car, etc.)
                if 1 <= label_idx <= len(CLASS_NAMES):
                    class_name = CLASS_NAMES[label_idx - 1]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    total_detections[class_name] += 1
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{class_name} {score:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
            
            frame_detections[f'frame_{frame_num}'] = class_counts
            out.write(frame)
            frame_num += 1
            
            if frame_num % 50 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {frame_num}/{total_frames} ({progress:.1f}%)", end='\r')
    
    cap.release()
    out.release()
    
    video_time = time.time() - video_start
    print(f"\n✓ Video processed in {video_time:.2f} seconds")
    print(f"✓ Output video saved to: {output_video_path}")
    
    # Standardized JSON format - total objects detected
    print("\n[3/4] Saving detection counts...")
    detections_file = output_dir / 'detections.json'
    
    # Ensure all classes are present (even if 0)
    standardized_detections = {cls: total_detections.get(cls, 0) for cls in CLASS_NAMES}
    
    detection_data = {
        'model': 'Faster R-CNN',
        'video_path': str(args.video),
        'total_detections': standardized_detections,
        'total_vehicles': sum(standardized_detections.values()),
        'frames_analyzed': frame_num,
        'confidence_threshold': args.conf
    }
    
    with open(detections_file, 'w') as f:
        json.dump(detection_data, f, indent=2)
    
    print(f"✓ Detection counts saved to: {detections_file}")
    
    # Standardized metrics format
    print("\n[4/4] Generating metrics...")
    total_frames = len(frame_detections)
    total_det = sum(standardized_detections.values())
    
    metrics = {
        'precision': 0.0,  # Not available without ground truth
        'recall': 0.0,     # Not available without ground truth
        'mAP50': 0.0,      # Not available without ground truth
        'mAP50_95': 0.0,   # Not available without ground truth
        'fitness': 0.0,    # Not available without ground truth
        'total_frames': total_frames,
        'total_detections': total_det,
        'avg_detections_per_frame': total_det / total_frames if total_frames > 0 else 0,
        'processing_time_seconds': video_time,
        'fps': total_frames / video_time if video_time > 0 else 0
    }
    
    # Save metrics - standardized format
    metrics_file = output_dir / 'metrics.json'
    metrics_data = {
        'model': 'Faster R-CNN',
        'video_path': str(args.video),
        'confidence_threshold': args.conf,
        **metrics
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Frames: {frame_num}")
    print(f"Total Vehicles: {sum(standardized_detections.values())}")
    print("\nDetections by Class:")
    for class_name in CLASS_NAMES:
        count = standardized_detections[class_name]
        print(f"  {class_name}: {count}")
    print("\n" + "=" * 60)
    print("✓ All results saved to:", output_dir)
    print("  - output_video.mp4")
    print("  - detections.json")
    print("  - metrics.json")
    print("=" * 60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()