"""
Unified YOLOv9 Testing Script
Combines video processing and metrics generation
Usage: python yolov9.py --video <path> [--model <path>] [--conf <0.5>] [--data <path>]
"""

import argparse
import os
import json
import time
import cv2
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv9 Testing and Evaluation')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default=None, help='Path to model file (default: Model/best.pt)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--data', type=str, default=None, help='Path to data.yaml for validation (optional)')
    parser.add_argument('--output-dir', type=str, default='Results', help='Output directory (default: Results)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set default model path if not provided
    if args.model is None:
        script_dir = Path(__file__).parent
        default_model = script_dir / 'Model' / 'best.pt'
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
    print("YOLOv9 Testing and Evaluation")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print(f"Confidence: {args.conf}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading model...")
    start_time = time.time()
    model = YOLO(args.model)
    print(f"✓ Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Open video for processing
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
    
    print(f"\n[2/4] Processing video...")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Resolution: {w}x{h}")
    print(f"  - FPS: {fps:.2f}")
    
    # Process frames and count detections
    frame_detections = {}
    total_detections = {}
    frame_num = 0
    video_start = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        det_results = model(frame, verbose=False, conf=args.conf)
        
        # Draw boxes and count objects per class
        class_counts = {}
        for result in det_results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_detections[class_name] = total_detections.get(class_name, 0) + 1
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{class_name} {confidence:.2f}",
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
    all_classes = ['Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck']
    standardized_detections = {cls: total_detections.get(cls, 0) for cls in all_classes}
    
    detection_data = {
        'model': 'YOLOv9',
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
    
    # Try to find validation data automatically
    validation_data = args.data
    if not validation_data:
        # Auto-detect validation data.yaml
        script_dir = Path(__file__).parent
        possible_data_paths = [
            script_dir.parent.parent / 'drive-download-20251215T134537Z-3-001' / 'YOLOv9e' / 'data' / 'data.yaml',
            script_dir / 'data' / 'data.yaml',
            script_dir.parent.parent / 'YOLOv9e' / 'data' / 'data.yaml'
        ]
        for path in possible_data_paths:
            if path.exists():
                validation_data = str(path)
                print(f"Auto-detected validation data: {validation_data}")
                break
    
    metrics = {}
    validation_metrics = {}
    
    if validation_data and os.path.exists(validation_data):
        # Run validation to get precision, recall, mAP
        print("Running validation on validation dataset...")
        try:
            val_results = model.val(data=validation_data, batch=8, verbose=False)
            
            if hasattr(val_results, 'results_dict'):
                validation_metrics = val_results.results_dict
            elif hasattr(val_results, 'metrics'):
                metrics_obj = val_results.metrics
                validation_metrics = {
                    'precision': float(getattr(metrics_obj, 'mp', 0)),
                    'recall': float(getattr(metrics_obj, 'mr', 0)),
                    'mAP50': float(getattr(metrics_obj, 'map50', 0)),
                    'mAP50_95': float(getattr(metrics_obj, 'map', 0)),
                    'fitness': float(getattr(metrics_obj, 'fitness', 0))
                }
            print("✓ Validation metrics obtained")
        except Exception as e:
            print(f"⚠ Could not run validation: {e}")
            validation_metrics = {}
    else:
        print("ℹ No validation data provided - precision/recall/mAP require ground truth")
        validation_metrics = {}
    
    # Video inference metrics (always available)
    total_frames = len(frame_detections)
    total_det = sum(standardized_detections.values())
    
    metrics = {
        **validation_metrics,  # Include validation metrics if available
        'total_frames': total_frames,
        'total_detections': total_det,
        'avg_detections_per_frame': total_det / total_frames if total_frames > 0 else 0,
        'processing_time_seconds': video_time,
        'fps': total_frames / video_time if video_time > 0 else 0
    }
    
    # Set defaults if validation metrics not available
    if 'precision' not in metrics:
        metrics['precision'] = 0.0
    if 'recall' not in metrics:
        metrics['recall'] = 0.0
    if 'mAP50' not in metrics:
        metrics['mAP50'] = 0.0
    if 'mAP50_95' not in metrics:
        metrics['mAP50_95'] = 0.0
    if 'fitness' not in metrics:
        metrics['fitness'] = 0.0
    
    # Save metrics - standardized format
    metrics_file = output_dir / 'metrics.json'
    metrics_data = {
        'model': 'YOLOv9',
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
    for class_name in all_classes:
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
