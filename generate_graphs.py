#!/usr/bin/env python3
"""
Generate comparison graphs for all three models (YOLOv8, YOLOv9, Faster R-CNN)
comparing results from low and high quality videos.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_combined_results():
    """Load the combined results JSON file."""
    base_dir = Path(__file__).parent
    results_file = base_dir / 'combined_results.json'
    
    if not results_file.exists():
        print("❌ combined_results.json not found. Please run combine_results.py first.")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def create_detection_comparison_graph(data):
    """Create bar chart comparing total vehicle detections across models."""
    models = []
    low_counts = []
    high_counts = []
    
    for model_name in ['YOLOv8', 'YOLOv9', 'Faster R-CNN']:
        if model_name in data['results']:
            models.append(model_name)
            low_det = data['results'][model_name]['low'].get('detections', {})
            high_det = data['results'][model_name]['high'].get('detections', {})
            low_counts.append(low_det.get('total_vehicles', 0))
            high_counts.append(high_det.get('total_vehicles', 0))
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, low_counts, width, label='Low Quality', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, high_counts, width, label='High Quality', color='#e74c3c', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Vehicles Detected', fontsize=12, fontweight='bold')
    ax.set_title('Total Vehicle Detections: Model Comparison (Low vs High Quality)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def create_vehicle_type_comparison_graph(data):
    """Create stacked bar chart comparing vehicle types across models."""
    vehicle_types = ['Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck']
    models = ['YOLOv8', 'YOLOv9', 'Faster R-CNN']
    
    # Prepare data for low quality
    low_data = {vt: [] for vt in vehicle_types}
    high_data = {vt: [] for vt in vehicle_types}
    
    for model_name in models:
        if model_name in data['results']:
            low_det = data['results'][model_name]['low'].get('detections', {})
            high_det = data['results'][model_name]['high'].get('detections', {})
            low_detections = low_det.get('total_detections', {})
            high_detections = high_det.get('total_detections', {})
            
            for vt in vehicle_types:
                low_data[vt].append(low_detections.get(vt, 0))
                high_data[vt].append(high_detections.get(vt, 0))
    
    x = np.arange(len(models))
    width = 0.35
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Low quality
    bottom_low = np.zeros(len(models))
    for i, vt in enumerate(vehicle_types):
        ax1.bar(x, low_data[vt], width, label=vt, bottom=bottom_low, color=colors[i], alpha=0.8)
        bottom_low += low_data[vt]
    
    ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Detections', fontsize=11, fontweight='bold')
    ax1.set_title('Low Quality Video', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # High quality
    bottom_high = np.zeros(len(models))
    for i, vt in enumerate(vehicle_types):
        ax2.bar(x, high_data[vt], width, label=vt, bottom=bottom_high, color=colors[i], alpha=0.8)
        bottom_high += high_data[vt]
    
    ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Detections', fontsize=11, fontweight='bold')
    ax2.set_title('High Quality Video', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.suptitle('Vehicle Type Distribution Across Models', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def create_performance_metrics_graph(data):
    """Create graph comparing performance metrics (FPS, processing time)."""
    models = []
    fps_low = []
    fps_high = []
    time_low = []
    time_high = []
    
    for model_name in ['YOLOv8', 'YOLOv9', 'Faster R-CNN']:
        if model_name in data['results']:
            models.append(model_name)
            low_metrics = data['results'][model_name]['low'].get('metrics', {})
            high_metrics = data['results'][model_name]['high'].get('metrics', {})
            fps_low.append(low_metrics.get('fps', 0))
            fps_high.append(high_metrics.get('fps', 0))
            time_low.append(low_metrics.get('processing_time_seconds', 0))
            time_high.append(high_metrics.get('processing_time_seconds', 0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # FPS comparison
    bars1 = ax1.bar(x - width/2, fps_low, width, label='Low Quality', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, fps_high, width, label='High Quality', color='#e74c3c', alpha=0.8)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax1.set_ylabel('FPS (Frames Per Second)', fontsize=11, fontweight='bold')
    ax1.set_title('Processing Speed (FPS)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Processing time comparison
    bars3 = ax2.bar(x - width/2, time_low, width, label='Low Quality', color='#3498db', alpha=0.8)
    bars4 = ax2.bar(x + width/2, time_high, width, label='High Quality', color='#e74c3c', alpha=0.8)
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Processing Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Processing Time', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.suptitle('Performance Metrics Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def create_accuracy_metrics_graph(data):
    """Create graph comparing accuracy metrics (precision, recall, mAP)."""
    models = []
    precision = []
    recall = []
    map50 = []
    map50_95 = []
    
    for model_name in ['YOLOv8', 'YOLOv9', 'Faster R-CNN']:
        if model_name in data['results']:
            models.append(model_name)
            low_metrics = data['results'][model_name]['low'].get('metrics', {})
            
            # Try to get validation metrics (prefer metrics/precision(B) format)
            prec = low_metrics.get('metrics/precision(B)', low_metrics.get('precision', 0))
            rec = low_metrics.get('metrics/recall(B)', low_metrics.get('recall', 0))
            m50 = low_metrics.get('metrics/mAP50(B)', low_metrics.get('mAP50', 0))
            m50_95 = low_metrics.get('metrics/mAP50-95(B)', low_metrics.get('mAP50_95', 0))
            
            precision.append(prec)
            recall.append(rec)
            map50.append(m50)
            map50_95.append(m50_95)
    
    # Only plot if we have valid metrics (not all zeros)
    if max(precision + recall + map50 + map50_95) > 0:
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - 1.5*width, precision, width, label='Precision', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, map50, width, label='mAP50', color='#f39c12', alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, map50_95, width, label='mAP50-95', color='#e74c3c', alpha=0.8)
        
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy Metrics Comparison (Validation Data)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim([0, 1.1])
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    return None

def create_percentage_comparison_graph(data):
    """Create graph showing percentage distribution of vehicle types."""
    vehicle_types = ['Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck']
    models = ['YOLOv8', 'YOLOv9', 'Faster R-CNN']
    
    # Get percentages for low quality
    low_percentages = {vt: [] for vt in vehicle_types}
    
    for model_name in models:
        if model_name in data['results']:
            low_det = data['results'][model_name]['low'].get('detections', {})
            percentages = low_det.get('total_detections_percent', {})
            for vt in vehicle_types:
                low_percentages[vt].append(percentages.get(vt, 0))
    
    x = np.arange(len(models))
    width = 0.15
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bottom = np.zeros(len(models))
    for i, vt in enumerate(vehicle_types):
        ax.bar(x, low_percentages[vt], width, label=vt, bottom=bottom, color=colors[i], alpha=0.8)
        bottom += low_percentages[vt]
        
        # Add percentage labels
        for j, val in enumerate(low_percentages[vt]):
            if val > 1:  # Only show label if percentage is significant
                ax.text(j, bottom[j] - val/2, f'{val:.1f}%',
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Vehicle Type Distribution by Percentage (Low Quality Video)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([0, 100])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def generate_all_graphs():
    """Generate all comparison graphs."""
    data = load_combined_results()
    if not data:
        return
    
    base_dir = Path(__file__).parent
    output_dir = base_dir / 'graphs'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Generating Comparison Graphs...")
    print("=" * 60)
    
    # 1. Total detections comparison
    print("1. Creating total detections comparison graph...")
    fig1 = create_detection_comparison_graph(data)
    fig1.savefig(output_dir / '1_total_detections_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("   ✓ Saved: graphs/1_total_detections_comparison.png")
    
    # 2. Vehicle type distribution
    print("2. Creating vehicle type distribution graph...")
    fig2 = create_vehicle_type_comparison_graph(data)
    fig2.savefig(output_dir / '2_vehicle_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("   ✓ Saved: graphs/2_vehicle_type_distribution.png")
    
    # 3. Performance metrics
    print("3. Creating performance metrics graph...")
    fig3 = create_performance_metrics_graph(data)
    fig3.savefig(output_dir / '3_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("   ✓ Saved: graphs/3_performance_metrics.png")
    
    # 4. Accuracy metrics (if available)
    print("4. Creating accuracy metrics graph...")
    fig4 = create_accuracy_metrics_graph(data)
    if fig4:
        fig4.savefig(output_dir / '4_accuracy_metrics.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print("   ✓ Saved: graphs/4_accuracy_metrics.png")
    else:
        print("   ⚠ Skipped: No validation metrics available")
    
    # 5. Percentage distribution
    print("5. Creating percentage distribution graph...")
    fig5 = create_percentage_comparison_graph(data)
    fig5.savefig(output_dir / '5_percentage_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print("   ✓ Saved: graphs/5_percentage_distribution.png")
    
    print("=" * 60)
    print(f"✓ All graphs saved to: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    try:
        generate_all_graphs()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

