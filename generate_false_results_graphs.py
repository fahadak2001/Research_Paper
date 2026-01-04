#!/usr/bin/env python3
"""
Generate graphs for false positives and false negatives analysis.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results():
    """Load false positives/negatives results."""
    base_dir = Path(__file__).parent
    results_file = base_dir / 'false_results' / 'false_positives_negatives.json'
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def create_tp_fp_fn_comparison(data):
    """Create bar chart comparing TP, FP, FN across models."""
    models = list(data.keys())
    
    overall_tp = [data[m]['overall']['true_positives'] for m in models]
    overall_fp = [data[m]['overall']['false_positives'] for m in models]
    overall_fn = [data[m]['overall']['false_negatives'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, overall_tp, width, label='True Positives', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, overall_fp, width, label='False Positives', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, overall_fn, width, label='False Negatives', color='#f39c12', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('True Positives, False Positives, and False Negatives Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def create_precision_recall_f1_comparison(data):
    """Create bar chart comparing Precision, Recall, and F1-Score."""
    models = list(data.keys())
    
    precision = [data[m]['overall']['precision'] for m in models]
    recall = [data[m]['overall']['recall'] for m in models]
    f1 = [data[m]['overall']['f1_score'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#9b59b6', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#1abc9c', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Precision, Recall, and F1-Score Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def create_class_wise_comparison(data):
    """Create comparison of TP, FP, FN for each vehicle class."""
    models = list(data.keys())
    classes = ['Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, class_name in enumerate(classes):
        ax = axes[idx]
        
        tp = [data[m][class_name]['true_positives'] for m in models]
        fp = [data[m][class_name]['false_positives'] for m in models]
        fn = [data[m][class_name]['false_negatives'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        bars1 = ax.bar(x - width, tp, width, label='TP', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x, fp, width, label='FP', color='#e74c3c', alpha=0.8)
        bars3 = ax.bar(x + width, fn, width, label='FN', color='#f39c12', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Overall metrics in last subplot
    ax = axes[5]
    overall_tp = [data[m]['overall']['true_positives'] for m in models]
    overall_fp = [data[m]['overall']['false_positives'] for m in models]
    overall_fn = [data[m]['overall']['false_negatives'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax.bar(x - width, overall_tp, width, label='TP', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, overall_fp, width, label='FP', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, overall_fn, width, label='FN', color='#f39c12', alpha=0.8)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_title('Overall', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.suptitle('Class-wise True Positives, False Positives, and False Negatives', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig

def create_class_wise_precision_recall(data):
    """Create heatmap-style comparison of precision and recall by class."""
    models = list(data.keys())
    classes = ['Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck', 'overall']
    
    precision_matrix = []
    recall_matrix = []
    
    for model in models:
        prec_row = [data[model][cls]['precision'] for cls in classes]
        rec_row = [data[model][cls]['recall'] for cls in classes]
        precision_matrix.append(prec_row)
        recall_matrix.append(rec_row)
    
    # Capitalize 'overall' for display
    class_labels = ['Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck', 'Overall']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Precision heatmap
    im1 = ax1.imshow(precision_matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(np.arange(len(class_labels)))
    ax1.set_yticks(np.arange(len(models)))
    ax1.set_xticklabels(class_labels)
    ax1.set_yticklabels(models)
    ax1.set_title('Precision by Class and Model', fontsize=12, fontweight='bold', pad=15)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(classes)):
            text = ax1.text(j, i, f'{precision_matrix[i][j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=9)
    
    plt.colorbar(im1, ax=ax1, label='Precision')
    
    # Recall heatmap
    im2 = ax2.imshow(recall_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(np.arange(len(class_labels)))
    ax2.set_yticks(np.arange(len(models)))
    ax2.set_xticklabels(class_labels)
    ax2.set_yticklabels(models)
    ax2.set_title('Recall by Class and Model', fontsize=12, fontweight='bold', pad=15)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(classes)):
            text = ax2.text(j, i, f'{recall_matrix[i][j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=9)
    
    plt.colorbar(im2, ax=ax2, label='Recall')
    
    plt.tight_layout()
    return fig

def create_fp_fn_ratio_comparison(data):
    """Create comparison of FP and FN ratios."""
    models = list(data.keys())
    
    fp_ratios = []
    fn_ratios = []
    
    for model in models:
        overall = data[model]['overall']
        total_pred = overall['true_positives'] + overall['false_positives']
        total_gt = overall['ground_truth_count']
        
        fp_ratio = overall['false_positives'] / total_pred if total_pred > 0 else 0
        fn_ratio = overall['false_negatives'] / total_gt if total_gt > 0 else 0
        
        fp_ratios.append(fp_ratio)
        fn_ratios.append(fn_ratio)
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, fp_ratios, width, label='False Positive Ratio', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, fn_ratios, width, label='False Negative Ratio', color='#f39c12', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ratio', fontsize=12, fontweight='bold')
    ax.set_title('False Positive and False Negative Ratios', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def generate_all_graphs():
    """Generate all false positives/negatives graphs."""
    data = load_results()
    if not data:
        return
    
    base_dir = Path(__file__).parent
    output_dir = base_dir / 'false_results'
    
    print("=" * 60)
    print("Generating False Positives/Negatives Graphs...")
    print("=" * 60)
    
    # 1. TP, FP, FN comparison
    print("1. Creating TP/FP/FN comparison graph...")
    fig1 = create_tp_fp_fn_comparison(data)
    fig1.savefig(output_dir / '1_tp_fp_fn_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("   ✓ Saved: false_results/1_tp_fp_fn_comparison.png")
    
    # 2. Precision, Recall, F1 comparison
    print("2. Creating Precision/Recall/F1 comparison graph...")
    fig2 = create_precision_recall_f1_comparison(data)
    fig2.savefig(output_dir / '2_precision_recall_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("   ✓ Saved: false_results/2_precision_recall_f1_comparison.png")
    
    # 3. Class-wise comparison
    print("3. Creating class-wise TP/FP/FN comparison graph...")
    fig3 = create_class_wise_comparison(data)
    fig3.savefig(output_dir / '3_class_wise_tp_fp_fn.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("   ✓ Saved: false_results/3_class_wise_tp_fp_fn.png")
    
    # 4. Class-wise precision/recall heatmap
    print("4. Creating class-wise precision/recall heatmap...")
    fig4 = create_class_wise_precision_recall(data)
    fig4.savefig(output_dir / '4_class_wise_precision_recall_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("   ✓ Saved: false_results/4_class_wise_precision_recall_heatmap.png")
    
    # 5. FP/FN ratio comparison
    print("5. Creating FP/FN ratio comparison graph...")
    fig5 = create_fp_fn_ratio_comparison(data)
    fig5.savefig(output_dir / '5_fp_fn_ratio_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print("   ✓ Saved: false_results/5_fp_fn_ratio_comparison.png")
    
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

