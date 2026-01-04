#!/usr/bin/env python3
"""
Calculate False Positives and False Negatives for YOLOv8, YOLOv9, and Faster R-CNN
by comparing predictions with ground truth labels on test images.
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

# Class names mapping (YOLO format: 0=Bus, 1=Car, 2=Motorcycle, 3=Pickup, 4=Truck)
CLASS_NAMES = ['Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck']
CLASS_MAPPING = {i: name for i, name in enumerate(CLASS_NAMES)}

# IoU threshold for matching predictions to ground truth
IOU_THRESHOLD = 0.5

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
                    'class_name': CLASS_MAPPING.get(class_id, f'Class_{class_id}'),
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

def match_predictions_to_ground_truth(pred_boxes, gt_boxes, img_width, img_height, iou_threshold=IOU_THRESHOLD):
    """Match predictions to ground truth boxes and return TP, FP, FN counts."""
    # Convert ground truth to xyxy
    gt_xyxy = []
    for gt in gt_boxes:
        xyxy = yolo_to_xyxy(gt, img_width, img_height)
        gt_xyxy.append({
            'box': xyxy,
            'class_id': gt['class_id'],
            'class_name': gt['class_name'],
            'matched': False
        })
    
    # Convert predictions to xyxy and match
    tp_count = {name: 0 for name in CLASS_NAMES}
    fp_count = {name: 0 for name in CLASS_NAMES}
    matched_gt = set()
    
    for pred in pred_boxes:
        pred_xyxy = pred['box']
        pred_class = pred['class_name']
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth box
        for idx, gt in enumerate(gt_xyxy):
            if gt['matched']:
                continue
            if gt['class_name'] != pred_class:
                continue
            
            iou = calculate_iou(pred_xyxy, gt['box'])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = idx
        
        if best_gt_idx >= 0:
            # True Positive
            tp_count[pred_class] += 1
            gt_xyxy[best_gt_idx]['matched'] = True
            matched_gt.add(best_gt_idx)
        else:
            # False Positive
            fp_count[pred_class] += 1
    
    # Count False Negatives (unmatched ground truth)
    fn_count = {name: 0 for name in CLASS_NAMES}
    for gt in gt_xyxy:
        if not gt['matched']:
            fn_count[gt['class_name']] += 1
    
    return tp_count, fp_count, fn_count

def run_yolo_inference(model_path, test_images_dir, conf_threshold=0.5):
    """Run YOLO model inference on test images."""
    model = YOLO(model_path)
    results = {}
    
    for img_path in Path(test_images_dir).glob('*.jpg'):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_height, img_width = img.shape[:2]
        pred_results = model(img, conf=conf_threshold, verbose=False)
        
        pred_boxes = []
        for result in pred_results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = CLASS_NAMES[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                
                pred_boxes.append({
                    'box': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                })
        
        results[img_path.name] = {
            'predictions': pred_boxes,
            'img_width': img_width,
            'img_height': img_height
        }
    
    return results

def run_faster_rcnn_inference(model_path, test_images_dir, conf_threshold=0.5, num_classes=6):
    """Run Faster R-CNN model inference on test images."""
    device = torch.device("cpu")
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    results = {}
    
    with torch.no_grad():
        for img_path in Path(test_images_dir).glob('*.jpg'):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = F.to_tensor(rgb).unsqueeze(0).to(device)
            
            preds = model(img_tensor)[0]
            
            pred_boxes = []
            for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
                if score < conf_threshold:
                    continue
                
                label_idx = int(label.item())
                # Skip background (label 0), map to class names (label 1 = Bus, etc.)
                if label_idx == 0 or label_idx > len(CLASS_NAMES):
                    continue
                
                class_name = CLASS_NAMES[label_idx - 1]
                x1, y1, x2, y2 = map(float, box)
                
                pred_boxes.append({
                    'box': [x1, y1, x2, y2],
                    'class_id': label_idx - 1,
                    'class_name': class_name,
                    'confidence': float(score)
                })
            
            results[img_path.name] = {
                'predictions': pred_boxes,
                'img_width': img_width,
                'img_height': img_height
            }
    
    return results

def calculate_metrics_for_model(model_name, model_path, test_images_dir, test_labels_dir, 
                                inference_func, **inference_kwargs):
    """Calculate TP, FP, FN for a model."""
    print(f"\n[{model_name}] Running inference on test images...")
    predictions = inference_func(model_path, test_images_dir, **inference_kwargs)
    
    print(f"[{model_name}] Comparing with ground truth...")
    total_tp = {name: 0 for name in CLASS_NAMES}
    total_fp = {name: 0 for name in CLASS_NAMES}
    total_fn = {name: 0 for name in CLASS_NAMES}
    total_gt = {name: 0 for name in CLASS_NAMES}
    
    for img_name, pred_data in predictions.items():
        # Get corresponding label file
        label_name = img_name.replace('.jpg', '.txt')
        label_path = Path(test_labels_dir) / label_name
        
        if not label_path.exists():
            continue
        
        gt_boxes = load_yolo_label(label_path)
        
        # Count ground truth by class
        for gt in gt_boxes:
            total_gt[gt['class_name']] += 1
        
        # Match predictions to ground truth
        tp, fp, fn = match_predictions_to_ground_truth(
            pred_data['predictions'],
            gt_boxes,
            pred_data['img_width'],
            pred_data['img_height']
        )
        
        for class_name in CLASS_NAMES:
            total_tp[class_name] += tp[class_name]
            total_fp[class_name] += fp[class_name]
            total_fn[class_name] += fn[class_name]
    
    # Calculate precision, recall, F1
    metrics = {}
    for class_name in CLASS_NAMES:
        tp = total_tp[class_name]
        fp = total_fp[class_name]
        fn = total_fn[class_name]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[class_name] = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'ground_truth_count': total_gt[class_name],
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4)
        }
    
    # Overall metrics
    total_tp_all = sum(total_tp.values())
    total_fp_all = sum(total_fp.values())
    total_fn_all = sum(total_fn.values())
    
    overall_precision = total_tp_all / (total_tp_all + total_fp_all) if (total_tp_all + total_fp_all) > 0 else 0.0
    overall_recall = total_tp_all / (total_tp_all + total_fn_all) if (total_tp_all + total_fn_all) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    metrics['overall'] = {
        'true_positives': total_tp_all,
        'false_positives': total_fp_all,
        'false_negatives': total_fn_all,
        'ground_truth_count': sum(total_gt.values()),
        'precision': round(overall_precision, 4),
        'recall': round(overall_recall, 4),
        'f1_score': round(overall_f1, 4)
    }
    
    return metrics

def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / 'false_results'
    output_dir.mkdir(exist_ok=True)
    
    # Test data paths
    test_images_dir = base_dir / 'drive-download-20251215T134537Z-3-001' / 'YOLOv8m' / 'data' / 'test' / 'images'
    test_labels_dir = base_dir / 'drive-download-20251215T134537Z-3-001' / 'YOLOv8m' / 'data' / 'test' / 'labels'
    
    if not test_images_dir.exists() or not test_labels_dir.exists():
        print(f"❌ Test data not found at {test_images_dir}")
        return
    
    print("=" * 60)
    print("Calculating False Positives and False Negatives")
    print("=" * 60)
    print(f"Test images: {test_images_dir}")
    print(f"Test labels: {test_labels_dir}")
    print(f"IoU Threshold: {IOU_THRESHOLD}")
    print("=" * 60)
    
    all_results = {}
    
    # YOLOv8
    yolo8_model = base_dir / 'src' / 'yolov8' / 'Model' / 'best.pt'
    if yolo8_model.exists():
        print("\n[1/3] Processing YOLOv8...")
        yolo8_metrics = calculate_metrics_for_model(
            'YOLOv8',
            str(yolo8_model),
            str(test_images_dir),
            str(test_labels_dir),
            run_yolo_inference,
            conf_threshold=0.5
        )
        all_results['YOLOv8'] = yolo8_metrics
    else:
        print(f"\n⚠ YOLOv8 model not found at {yolo8_model}")
    
    # YOLOv9
    yolo9_model = base_dir / 'src' / 'yolov9' / 'Model' / 'best.pt'
    if yolo9_model.exists():
        print("\n[2/3] Processing YOLOv9...")
        yolo9_metrics = calculate_metrics_for_model(
            'YOLOv9',
            str(yolo9_model),
            str(test_images_dir),
            str(test_labels_dir),
            run_yolo_inference,
            conf_threshold=0.5
        )
        all_results['YOLOv9'] = yolo9_metrics
    else:
        print(f"\n⚠ YOLOv9 model not found at {yolo9_model}")
    
    # Faster R-CNN
    frcnn_model = base_dir / 'src' / 'coco' / 'faster R_CNN' / 'Model' / 'faster_rcnn_best.pth'
    if frcnn_model.exists():
        print("\n[3/3] Processing Faster R-CNN...")
        frcnn_metrics = calculate_metrics_for_model(
            'Faster R-CNN',
            str(frcnn_model),
            str(test_images_dir),
            str(test_labels_dir),
            run_faster_rcnn_inference,
            conf_threshold=0.5,
            num_classes=6
        )
        all_results['Faster R-CNN'] = frcnn_metrics
    else:
        print(f"\n⚠ Faster R-CNN model not found at {frcnn_model}")
    
    # Save results to JSON
    output_json = output_dir / 'false_positives_negatives.json'
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for model_name, metrics in all_results.items():
        print(f"\n{model_name}:")
        overall = metrics['overall']
        print(f"  Overall Precision: {overall['precision']:.4f}")
        print(f"  Overall Recall: {overall['recall']:.4f}")
        print(f"  Overall F1-Score: {overall['f1_score']:.4f}")
        print(f"  True Positives: {overall['true_positives']}")
        print(f"  False Positives: {overall['false_positives']}")
        print(f"  False Negatives: {overall['false_negatives']}")
    
    print(f"\n✓ Results saved to: {output_json}")
    print("=" * 60)
    
    return all_results

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

