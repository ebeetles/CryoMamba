#!/usr/bin/env python3
"""
Compare actual nnU-Net results with ground truth labels.
This script analyzes the performance of the real nnU-Net inference.
"""

import os
import sys
import json
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import time
import psutil
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def compare_actual_nnunet_with_labels():
    """Compare actual nnU-Net results with ground truth labels."""
    print("=" * 60)
    print("Actual nnU-Net Results vs Ground Truth Labels Comparison")
    print("=" * 60)
    
    # Test 1: Find files
    print("\n1. Finding Files...")
    test_data_path = Path("/Users/elwinli/CryoMamba/test_data")
    results_dir = test_data_path / "nnunet_results"
    labels_dir = test_data_path / "labels"
    
    # Find our actual nnU-Net segmentation mask
    seg_mask_files = list(results_dir.glob("*nnunet_actual_segmentation.nii.gz"))
    if not seg_mask_files:
        print("✗ No actual nnU-Net segmentation mask files found")
        return False
    
    seg_mask_file = seg_mask_files[0]
    print(f"✓ Found actual nnU-Net segmentation mask: {seg_mask_file.name}")
    
    # Find corresponding ground truth label
    # Extract base name (e.g., "c_elegans" from "c_elegans_em.nii_nnunet_actual_segmentation.nii.gz")
    base_name = seg_mask_file.name.replace('.nii_nnunet_actual_segmentation.nii.gz', '')
    
    # The labels are named like "c_elegans_mito.mrc" (without the "_em" part)
    if '_em' in base_name:
        label_base = base_name.split('_em')[0]
    else:
        label_base = base_name
    
    # Look for corresponding label file (labels have "_mito" suffix)
    label_file = labels_dir / f"{label_base}_mito.mrc"
    if not label_file.exists():
        # Try without _mito suffix
        label_file = labels_dir / f"{label_base}.mrc"
    
    if not label_file.exists():
        print(f"✗ No corresponding label file found for {base_name}")
        print(f"  Looked for: {labels_dir / f'{label_base}_mito.mrc'}")
        print(f"  And: {labels_dir / f'{label_base}.mrc'}")
        return False
    
    print(f"✓ Found ground truth label: {label_file.name}")
    
    # Test 2: Load files
    print("\n2. Loading Files...")
    try:
        # Load actual nnU-Net segmentation mask
        seg_img = nib.load(seg_mask_file)
        seg_data = seg_img.get_fdata()
        
        print(f"✓ Loaded actual nnU-Net segmentation mask")
        print(f"  Shape: {seg_data.shape}")
        print(f"  Data type: {seg_data.dtype}")
        print(f"  Unique values: {np.unique(seg_data)}")
        
        # Load ground truth label
        import mrcfile
        with mrcfile.open(label_file, mode='r') as mrc:
            label_data = mrc.data
            # Transpose to match segmentation orientation
            if len(label_data.shape) == 3:
                label_data = np.transpose(label_data, (2, 1, 0))
        
        print(f"✓ Loaded ground truth label")
        print(f"  Shape: {label_data.shape}")
        print(f"  Data type: {label_data.dtype}")
        print(f"  Unique values: {np.unique(label_data)}")
        
    except Exception as e:
        print(f"✗ Error loading files: {e}")
        return False
    
    # Test 3: Compare shapes and normalize
    print("\n3. Comparing Shapes and Normalizing...")
    
    # Ensure same shape
    if seg_data.shape != label_data.shape:
        print(f"⚠ Shape mismatch: seg {seg_data.shape} vs label {label_data.shape}")
        
        # Try to resize to match
        from scipy.ndimage import zoom
        if seg_data.size > label_data.size:
            # Segmentation is larger, downsample it
            zoom_factors = [label_data.shape[i] / seg_data.shape[i] for i in range(3)]
            seg_data = zoom(seg_data, zoom_factors, order=0)  # Nearest neighbor
            print(f"✓ Downsampled segmentation to match label shape")
        else:
            # Label is larger, downsample it
            zoom_factors = [seg_data.shape[i] / label_data.shape[i] for i in range(3)]
            label_data = zoom(label_data, zoom_factors, order=0)  # Nearest neighbor
            print(f"✓ Downsampled label to match segmentation shape")
    
    # Normalize label data to binary (0, 1)
    if np.max(label_data) > 1:
        label_data = (label_data > 0).astype(np.uint8)
        print(f"✓ Normalized label to binary format")
    
    # Ensure segmentation is binary
    seg_data = (seg_data > 0).astype(np.uint8)
    print(f"✓ Ensured segmentation is binary")
    
    print(f"✓ Final shapes: seg {seg_data.shape}, label {label_data.shape}")
    
    # Test 4: Calculate comparison metrics
    print("\n4. Calculating Comparison Metrics...")
    
    try:
        # Calculate basic metrics
        total_voxels = seg_data.size
        
        # True Positives, False Positives, False Negatives, True Negatives
        tp = np.sum((seg_data == 1) & (label_data == 1))
        fp = np.sum((seg_data == 1) & (label_data == 0))
        fn = np.sum((seg_data == 0) & (label_data == 1))
        tn = np.sum((seg_data == 0) & (label_data == 0))
        
        print(f"✓ Confusion Matrix:")
        print(f"  True Positives (TP):  {tp:,}")
        print(f"  False Positives (FP): {fp:,}")
        print(f"  False Negatives (FN): {fn:,}")
        print(f"  True Negatives (TN):  {tn:,}")
        
        # Calculate metrics
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        
        print(f"\n✓ Performance Metrics:")
        print(f"  Dice Coefficient: {dice:.4f}")
        print(f"  IoU (Jaccard):    {iou:.4f}")
        print(f"  Accuracy:         {accuracy:.4f}")
        print(f"  Precision:        {precision:.4f}")
        print(f"  Recall:           {recall:.4f}")
        print(f"  Specificity:      {specificity:.4f}")
        
        # Calculate overlap statistics
        seg_foreground = np.sum(seg_data)
        label_foreground = np.sum(label_data)
        overlap = tp
        
        seg_percentage = (seg_foreground / total_voxels) * 100
        label_percentage = (label_foreground / total_voxels) * 100
        overlap_percentage = (overlap / total_voxels) * 100
        
        print(f"\n✓ Overlap Statistics:")
        print(f"  nnU-Net foreground: {seg_foreground:,} ({seg_percentage:.2f}%)")
        print(f"  Label foreground:   {label_foreground:,} ({label_percentage:.2f}%)")
        print(f"  Overlap:            {overlap:,} ({overlap_percentage:.2f}%)")
        
    except Exception as e:
        print(f"✗ Error calculating metrics: {e}")
        return False
    
    # Test 5: Create comparison visualization
    print("\n5. Creating Comparison Visualization...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Find middle slice
        mid_z = seg_data.shape[2] // 2
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Actual nnU-Net vs Ground Truth Comparison - {label_base}', fontsize=16)
        
        # Original data (if available)
        try:
            original_file = test_data_path / f"{label_base}_em.mrc"
            if original_file.exists():
                with mrcfile.open(original_file, mode='r') as mrc:
                    orig_data = mrc.data
                    if len(orig_data.shape) == 3:
                        orig_data = np.transpose(orig_data, (2, 1, 0))
                
                # Resize if needed
                if orig_data.shape != seg_data.shape:
                    zoom_factors = [seg_data.shape[i] / orig_data.shape[i] for i in range(3)]
                    orig_data = zoom(orig_data, zoom_factors, order=1)
                
                axes[0, 0].imshow(orig_data[:, :, mid_z], cmap='gray')
                axes[0, 0].set_title('Original Image')
            else:
                axes[0, 0].text(0.5, 0.5, 'Original\nNot Available', ha='center', va='center')
                axes[0, 0].set_title('Original Image')
        except:
            axes[0, 0].text(0.5, 0.5, 'Original\nNot Available', ha='center', va='center')
            axes[0, 0].set_title('Original Image')
        
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(label_data[:, :, mid_z], cmap='gray')
        axes[0, 1].set_title('Ground Truth Label')
        axes[0, 1].axis('off')
        
        # Actual nnU-Net segmentation
        axes[0, 2].imshow(seg_data[:, :, mid_z], cmap='gray')
        axes[0, 2].set_title('Actual nnU-Net Segmentation')
        axes[0, 2].axis('off')
        
        # Difference (Label - Segmentation)
        diff = label_data[:, :, mid_z] - seg_data[:, :, mid_z]
        axes[1, 0].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 0].set_title('Difference (Label - nnU-Net)')
        axes[1, 0].axis('off')
        
        # Overlay
        axes[1, 1].imshow(label_data[:, :, mid_z], cmap='gray', alpha=0.7)
        axes[1, 1].imshow(seg_data[:, :, mid_z], cmap='Reds', alpha=0.3)
        axes[1, 1].set_title('Overlay (Gray: Label, Red: nnU-Net)')
        axes[1, 1].axis('off')
        
        # Metrics text
        metrics_text = f"""Performance Metrics:
Dice Coefficient: {dice:.4f}
IoU (Jaccard): {iou:.4f}
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
Specificity: {specificity:.4f}

Foreground Coverage:
nnU-Net: {seg_percentage:.2f}%
Ground Truth: {label_percentage:.2f}%
Overlap: {overlap_percentage:.2f}%"""
        
        axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Performance Metrics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save comparison
        comparison_path = results_dir / f"{label_base}_actual_nnunet_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comparison visualization: {comparison_path}")
        
    except Exception as e:
        print(f"⚠ Could not create visualization: {e}")
    
    # Test 6: Analysis and recommendations
    print("\n6. Analysis and Recommendations...")
    
    print(f"\n📊 Analysis Results:")
    
    if dice > 0.8:
        print(f"✅ Excellent match! Dice coefficient: {dice:.4f}")
    elif dice > 0.6:
        print(f"✅ Good match! Dice coefficient: {dice:.4f}")
    elif dice > 0.4:
        print(f"⚠ Moderate match. Dice coefficient: {dice:.4f}")
    else:
        print(f"❌ Poor match. Dice coefficient: {dice:.4f}")
    
    print(f"\n💡 Key Findings:")
    
    print(f"  - nnU-Net identified {seg_percentage:.2f}% of voxels as foreground")
    print(f"  - Ground truth has {label_percentage:.2f}% foreground voxels")
    print(f"  - Overlap is {overlap_percentage:.2f}% of total voxels")
    
    if dice < 0.6:
        print(f"\n🔍 Potential Issues:")
        print(f"  - Model may be trained on different data")
        print(f"  - Different annotation criteria")
        print(f"  - Model may need fine-tuning")
        print(f"  - Preprocessing differences")
    else:
        print(f"\n✅ Good Performance:")
        print(f"  - Model shows reasonable agreement with labels")
        print(f"  - Can be used for segmentation tasks")
        print(f"  - Consider fine-tuning for better results")
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETED")
    print("=" * 60)
    
    return True

def main():
    """Main function."""
    print("Comparing actual nnU-Net results with ground truth labels...")
    
    success = compare_actual_nnunet_with_labels()
    
    if success:
        print("\n✅ Comparison completed successfully!")
        print("\nKey findings:")
        print("- Actual nnU-Net segmentation vs ground truth labels analyzed")
        print("- Performance metrics calculated")
        print("- Comparison visualization created")
        print("- Analysis and recommendations provided")
    else:
        print("\n❌ Comparison failed.")
    
    return success

if __name__ == "__main__":
    main()
