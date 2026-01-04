#!/usr/bin/env python3
"""
Calculate unified detection accuracy metrics (mAP50, mAP50-95) for all three models
on the same test dataset for fair comparison.
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from collections import defaultdict

# Class names mapping (YOLO format: 0=Bus, 1=Car, 2=Motorcycle, 3=Pickup, 4=Truck)
CLASS_NAMES = ['Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck']
NUM_CLASSES = len(CLASS_NAMES)

# IoU thresholds for mAP calculation
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)  # 0.5 to 0.95 in steps of 0.05

def load_yolo_label(label_path):
    """Load YOLO format label file and return list of bounding boxes."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append({
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f'Class_{class_id}',
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
    return boxes

def yolo_to_xyxy(box, img_width, img_height):
    """Convert YOLO format (normalized center, width, height) to xyxy format."""
    x_center = box['x_center'] * img_width
    y_center = box['y_center'] * img_height
    width = box['width'] * img_width
    height = box['height'] * img_height
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes in xyxy format."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def calculate_ap_per_class(predictions, ground_truths, class_id, iou_threshold=0.5):
    """
    Calculate Average Precision (AP) for a specific class at a given IoU threshold.
    Uses the COCO evaluation style.
    """
    # Filter predictions and ground truths for this class
    class_predictions = [(img, pred) for img, pred in predictions if pred['class_id'] == class_id]
    class_ground_truths = [(img, gt) for img, gt in ground_truths if gt['class_id'] == class_id]
    
    if len(class_ground_truths) == 0:
        return 0.0, 0, 0, 0
    
    # Sort predictions by confidence (descending)
    class_predictions.sort(key=lambda x: x[1]['confidence'], reverse=True)
    
    # Group ground truths by image
    gt_by_image = defaultdict(list)
    for img_name, gt in class_ground_truths:
        gt_by_image[img_name].append(gt)
    
    # Track which ground truths have been matched (by image)
    gt_matched = {img: [False] * len(gts) for img, gts in gt_by_image.items()}
    
    tp = []  # True positives
    fp = []  # False positives
    
    for img_name, pred in class_predictions:
        # Get ground truths for this image
        img_gts = gt_by_image.get(img_name, [])
        img_gt_matched = gt_matched.get(img_name, [])
        
        if len(img_gts) == 0:
            # No ground truth for this image, so it's a false positive
            tp.append(0)
            fp.append(1)
            continue
        
        best_iou = 0.0
        best_gt_idx = -1
        
        # Find best matching ground truth
        for idx, gt in enumerate(img_gts):
            if idx >= len(img_gt_matched):
                continue  # Safety check
            if img_gt_matched[idx]:
                continue  # Already matched
            
            iou = calculate_iou(pred['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        
        # Check if match is valid (IoU >= threshold)
        if best_iou >= iou_threshold and best_gt_idx >= 0 and best_gt_idx < len(img_gt_matched):
            tp.append(1)
            fp.append(0)
            img_gt_matched[best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(class_ground_truths) if len(class_ground_truths) > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum).sum() > 0 else np.zeros_like(tp_cumsum)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    num_tp = int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0
    num_fp = int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0
    num_fn = len(class_ground_truths) - num_tp
    
    return ap, num_tp, num_fp, num_fn

def run_yolo_inference(model_path, test_images_dir, conf_threshold=0.5):
    """Run YOLO model inference on test images."""
    model = YOLO(model_path)
    predictions = {}
    
    for img_path in sorted(Path(test_images_dir).glob('*.jpg')):
        img_name = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        results = model(str(img_path), verbose=False, conf=conf_threshold)
        
        pred_list = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                
                pred_list.append({
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f'Class_{class_id}',
                    'box': [x1, y1, x2, y2],
                    'confidence': confidence
                })
        
        predictions[img_name] = pred_list
    
    return predictions

def run_faster_rcnn_inference(model_path, test_images_dir, conf_threshold=0.5, num_classes=6):
    """Run Faster R-CNN model inference on test images."""
    device = torch.device('cpu')
    
    # Load model
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    predictions = {}
    
    with torch.no_grad():
        for img_path in sorted(Path(test_images_dir).glob('*.jpg')):
            img_name = img_path.stem
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = F.to_tensor(rgb).unsqueeze(0).to(device)
            
            preds = model(img_tensor)[0]
            
            h, w = img.shape[:2]
            pred_list = []
            
            for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
                if score < conf_threshold:
                    continue
                
                label_idx = int(label.item())
                if label_idx == 0:  # Skip background
                    continue
                
                # Convert to class_id (label 1 = Bus = class 0, label 2 = Car = class 1, etc.)
                class_id = label_idx - 1
                if class_id >= len(CLASS_NAMES):
                    continue
                
                x1, y1, x2, y2 = map(float, box)
                
                pred_list.append({
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id],
                    'box': [x1, y1, x2, y2],
                    'confidence': float(score.item())
                })
            
            predictions[img_name] = pred_list
    
    return predictions

def calculate_metrics_for_model(model_name, model_path, test_images_dir, test_labels_dir, 
                                 inference_func, **inference_kwargs):
    """Calculate mAP and other metrics for a model."""
    print(f"\n{'='*60}")
    print(f"Calculating metrics for {model_name}")
    print(f"{'='*60}")
    
    # Run inference
    print(f"Running inference on test images...")
    predictions_dict = inference_func(model_path, test_images_dir, **inference_kwargs)
    
    # Load ground truth
    print(f"Loading ground truth labels...")
    ground_truths = []
    predictions = []
    
    for img_path in sorted(Path(test_images_dir).glob('*.jpg')):
        img_name = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Load ground truth
        label_path = Path(test_labels_dir) / f"{img_name}.txt"
        gt_boxes = load_yolo_label(str(label_path))
        
        for gt in gt_boxes:
            gt_box_xyxy = yolo_to_xyxy(gt, w, h)
            ground_truths.append({
                'image': img_name,
                'class_id': gt['class_id'],
                'class_name': gt['class_name'],
                'box': gt_box_xyxy
            })
        
        # Add predictions
        for pred in predictions_dict.get(img_name, []):
            predictions.append({
                'image': img_name,
                'class_id': pred['class_id'],
                'class_name': pred['class_name'],
                'box': pred['box'],
                'confidence': pred['confidence']
            })
    
    # Calculate mAP50 and mAP50-95
    print(f"Calculating mAP metrics...")
    
    ap_per_class_50 = {}
    ap_per_class_50_95 = {}
    class_stats = {}
    
    for class_id in range(NUM_CLASSES):
        # mAP50
        ap_50, tp, fp, fn = calculate_ap_per_class(
            [(p['image'], p) for p in predictions],
            [(g['image'], g) for g in ground_truths],
            class_id,
            iou_threshold=0.5
        )
        ap_per_class_50[class_id] = ap_50
        
        # mAP50-95 (average over multiple IoU thresholds)
        aps = []
        for iou_thresh in IOU_THRESHOLDS:
            ap, _, _, _ = calculate_ap_per_class(
                [(p['image'], p) for p in predictions],
                [(g['image'], g) for g in ground_truths],
                class_id,
                iou_threshold=iou_thresh
            )
            aps.append(ap)
        ap_per_class_50_95[class_id] = np.mean(aps)
        
        class_stats[class_id] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'ground_truth_count': sum(1 for g in ground_truths if g['class_id'] == class_id)
        }
    
    # Calculate overall mAP
    map50 = np.mean(list(ap_per_class_50.values()))
    map50_95 = np.mean(list(ap_per_class_50_95.values()))
    
    # Calculate overall TP, FP, FN
    total_tp = sum(stats['tp'] for stats in class_stats.values())
    total_fp = sum(stats['fp'] for stats in class_stats.values())
    total_fn = sum(stats['fn'] for stats in class_stats.values())
    total_gt = sum(stats['ground_truth_count'] for stats in class_stats.values())
    
    # Calculate detection accuracy (TP / (TP + FP + FN))
    detection_accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    
    # Calculate precision and recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results = {
        'model': model_name,
        'mAP50': float(map50),
        'mAP50_95': float(map50_95),
        'detection_accuracy': float(detection_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'true_positives': int(total_tp),
        'false_positives': int(total_fp),
        'false_negatives': int(total_fn),
        'ground_truth_count': int(total_gt),
        'per_class_mAP50': {CLASS_NAMES[i]: float(ap_per_class_50[i]) for i in range(NUM_CLASSES)},
        'per_class_mAP50_95': {CLASS_NAMES[i]: float(ap_per_class_50_95[i]) for i in range(NUM_CLASSES)},
        'per_class_stats': {
            CLASS_NAMES[i]: {
                'tp': class_stats[i]['tp'],
                'fp': class_stats[i]['fp'],
                'fn': class_stats[i]['fn'],
                'ground_truth_count': class_stats[i]['ground_truth_count']
            }
            for i in range(NUM_CLASSES)
        }
    }
    
    return results

def main():
    base_dir = Path(__file__).parent
    
    # Test dataset paths
    test_images_dir = base_dir / 'drive-download-20251215T134537Z-3-001' / 'YOLOv8m' / 'data' / 'test' / 'images'
    test_labels_dir = base_dir / 'drive-download-20251215T134537Z-3-001' / 'YOLOv8m' / 'data' / 'test' / 'labels'
    
    if not test_images_dir.exists() or not test_labels_dir.exists():
        print(f"❌ Test data not found!")
        print(f"Expected test images at: {test_images_dir}")
        print(f"Expected test labels at: {test_labels_dir}")
        return
    
    print(f"✓ Test images found: {test_images_dir}")
    print(f"✓ Test labels found: {test_labels_dir}")
    
    # Model paths
    yolo8_model = base_dir / 'src' / 'yolov8' / 'Model' / 'best.pt'
    yolo9_model = base_dir / 'src' / 'yolov9' / 'Model' / 'best.pt'
    frcnn_model = base_dir / 'src' / 'coco' / 'faster R_CNN' / 'Model' / 'faster_rcnn_best.pth'
    
    all_results = {}
    
    # Calculate metrics for each model
    if yolo8_model.exists():
        all_results['YOLOv8'] = calculate_metrics_for_model(
            'YOLOv8',
            str(yolo8_model),
            str(test_images_dir),
            str(test_labels_dir),
            run_yolo_inference,
            conf_threshold=0.5
        )
    else:
        print(f"⚠ YOLOv8 model not found at {yolo8_model}")
    
    if yolo9_model.exists():
        all_results['YOLOv9'] = calculate_metrics_for_model(
            'YOLOv9',
            str(yolo9_model),
            str(test_images_dir),
            str(test_labels_dir),
            run_yolo_inference,
            conf_threshold=0.5
        )
    else:
        print(f"⚠ YOLOv9 model not found at {yolo9_model}")
    
    if frcnn_model.exists():
        all_results['Faster R-CNN'] = calculate_metrics_for_model(
            'Faster R-CNN',
            str(frcnn_model),
            str(test_images_dir),
            str(test_labels_dir),
            run_faster_rcnn_inference,
            conf_threshold=0.5,
            num_classes=6
        )
    else:
        print(f"⚠ Faster R-CNN model not found at {frcnn_model}")
    
    # Save results
    output_file = base_dir / 'unified_detection_accuracy.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("UNIFIED DETECTION ACCURACY RESULTS")
    print(f"{'='*60}\n")
    
    # Print summary table
    print("Overall Metrics:")
    print(f"{'Model':<15} {'mAP50':<10} {'mAP50-95':<12} {'Detection Acc':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 90)
    for model_name, results in all_results.items():
        print(f"{model_name:<15} {results['mAP50']:<10.4f} {results['mAP50_95']:<12.4f} "
              f"{results['detection_accuracy']:<15.4f} {results['precision']:<12.4f} "
              f"{results['recall']:<12.4f} {results['f1_score']:<12.4f}")
    
    print(f"\n{'='*60}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

