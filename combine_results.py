#!/usr/bin/env python3
"""
Combine all model results (YOLOv8, YOLOv9, Faster R-CNN) 
from low and high quality videos into a single JSON file.
"""

import json
import os
from pathlib import Path

def load_json(file_path):
    """Load JSON file and return its contents."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def combine_all_results():
    """Combine all results from different models and video qualities."""
    
    base_dir = Path(__file__).parent
    
    # Define paths for each model's results
    model_paths = {
        'YOLOv8': {
            'low': {
                'detections': base_dir / 'src' / 'yolov8' / 'Results_low' / 'detections.json',
                'metrics': base_dir / 'src' / 'yolov8' / 'Results_low' / 'metrics.json'
            },
            'high': {
                'detections': base_dir / 'src' / 'yolov8' / 'Results_high' / 'detections.json',
                'metrics': base_dir / 'src' / 'yolov8' / 'Results_high' / 'metrics.json'
            }
        },
        'YOLOv9': {
            'low': {
                'detections': base_dir / 'src' / 'yolov9' / 'Results_low' / 'detections.json',
                'metrics': base_dir / 'src' / 'yolov9' / 'Results_low' / 'metrics.json'
            },
            'high': {
                'detections': base_dir / 'src' / 'yolov9' / 'Results_high' / 'detections.json',
                'metrics': base_dir / 'src' / 'yolov9' / 'Results_high' / 'metrics.json'
            }
        },
        'Faster R-CNN': {
            'low': {
                'detections': base_dir / 'src' / 'coco' / 'faster R_CNN' / 'Results_low' / 'detections.json',
                'metrics': base_dir / 'src' / 'coco' / 'faster R_CNN' / 'Results_low' / 'metrics.json'
            },
            'high': {
                'detections': base_dir / 'src' / 'coco' / 'faster R_CNN' / 'Results_high' / 'detections.json',
                'metrics': base_dir / 'src' / 'coco' / 'faster R_CNN' / 'Results_high' / 'metrics.json'
            }
        }
    }
    
    # Combined results structure
    combined_results = {
        'summary': {
            'total_models': len(model_paths),
            'models': list(model_paths.keys()),
            'video_qualities': ['low', 'high']
        },
        'results': {}
    }
    
    # Load and combine all results
    for model_name, qualities in model_paths.items():
        combined_results['results'][model_name] = {}
        
        for quality in ['low', 'high']:
            detections_path = qualities[quality]['detections']
            metrics_path = qualities[quality]['metrics']
            
            detections_data = load_json(detections_path)
            metrics_data = load_json(metrics_path)
            
            if detections_data or metrics_data:
                combined_results['results'][model_name][quality] = {}
                
                if detections_data:
                    # Add percentage breakdown for each vehicle type
                    total_vehicles = detections_data.get('total_vehicles', 0)
                    total_detections = detections_data.get('total_detections', {})
                    
                    # Calculate percentages for each vehicle type
                    detections_data['total_detections_percent'] = {}
                    for vehicle_type, count in total_detections.items():
                        if total_vehicles > 0:
                            detections_data['total_detections_percent'][vehicle_type] = round((count / total_vehicles) * 100, 2)
                        else:
                            detections_data['total_detections_percent'][vehicle_type] = 0.0
                    
                    combined_results['results'][model_name][quality]['detections'] = detections_data
                
                if metrics_data:
                    combined_results['results'][model_name][quality]['metrics'] = metrics_data
    
    # Add comparison summary with percentages
    combined_results['comparison'] = {}
    
    # Calculate total across all models for percentage distribution
    total_all_models_low = 0
    total_all_models_high = 0
    
    for model_name in combined_results['results'].keys():
        if 'low' in combined_results['results'][model_name] and 'high' in combined_results['results'][model_name]:
            low_det = combined_results['results'][model_name]['low'].get('detections', {})
            high_det = combined_results['results'][model_name]['high'].get('detections', {})
            
            if low_det and high_det:
                total_low = low_det.get('total_vehicles', 0)
                total_high = high_det.get('total_vehicles', 0)
                total_all_models_low += total_low
                total_all_models_high += total_high
                
                # Calculate percentage breakdown by vehicle type
                low_detections = low_det.get('total_detections', {})
                high_detections = high_det.get('total_detections', {})
                
                vehicle_type_percentages = {}
                for vehicle_type in ['Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck']:
                    low_count = low_detections.get(vehicle_type, 0)
                    high_count = high_detections.get(vehicle_type, 0)
                    vehicle_type_percentages[vehicle_type] = {
                        'low_count': low_count,
                        'high_count': high_count,
                        'low_percent': round((low_count / total_low) * 100, 2) if total_low > 0 else 0.0,
                        'high_percent': round((high_count / total_high) * 100, 2) if total_high > 0 else 0.0,
                        'difference': high_count - low_count,
                        'difference_percent': round(((high_count - low_count) / low_count) * 100, 2) if low_count > 0 else 0.0
                    }
                
                combined_results['comparison'][model_name] = {
                    'total_vehicles_low': total_low,
                    'total_vehicles_high': total_high,
                    'difference': total_high - total_low,
                    'difference_percent': round(((total_high - total_low) / total_low) * 100, 2) if total_low > 0 else 0.0,
                    'vehicle_type_breakdown': vehicle_type_percentages
                }
    
    # Add overall model distribution percentages
    if total_all_models_low > 0 or total_all_models_high > 0:
        combined_results['model_distribution'] = {}
        for model_name in combined_results['comparison'].keys():
            model_data = combined_results['comparison'][model_name]
            combined_results['model_distribution'][model_name] = {
                'low_percent_of_total': round((model_data['total_vehicles_low'] / total_all_models_low) * 100, 2) if total_all_models_low > 0 else 0.0,
                'high_percent_of_total': round((model_data['total_vehicles_high'] / total_all_models_high) * 100, 2) if total_all_models_high > 0 else 0.0
            }
    
    # Save combined results
    output_path = base_dir / 'combined_results.json'
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print("=" * 60)
    print("Results Combined Successfully!")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print(f"\nSummary:")
    print(f"  - Models: {', '.join(combined_results['summary']['models'])}")
    print(f"  - Video Qualities: {', '.join(combined_results['summary']['video_qualities'])}")
    print(f"\nComparison Summary:")
    for model, comp in combined_results['comparison'].items():
        print(f"\n  {model}:")
        print(f"    Low Quality:  {comp['total_vehicles_low']} vehicles")
        print(f"    High Quality: {comp['total_vehicles_high']} vehicles")
        print(f"    Difference:   {comp['difference']} ({comp['difference_percent']:.2f}%)")
    print("=" * 60)
    
    return output_path

if __name__ == '__main__':
    try:
        combine_all_results()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

