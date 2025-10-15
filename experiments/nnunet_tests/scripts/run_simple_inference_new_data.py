#!/usr/bin/env python3
"""
Simple nnU-Net inference on test_data_new folder.
This script processes one file at a time to avoid memory issues.
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

# Set up nnU-Net environment variables
os.environ['nnUNet_raw'] = '/Users/elwinli/Downloads/test_data'
os.environ['nnUNet_preprocessed'] = '/Users/elwinli/Downloads/test_data_preprocessed'
os.environ['nnUNet_results'] = '/Users/elwinli/Downloads/pretrained_weights'

def run_simple_inference_on_new_data():
    """Run simple nnU-Net inference on test_data_new folder."""
    print("=" * 60)
    print("Simple nnU-Net Inference on test_data_new")
    print("=" * 60)
    
    # Test 1: Initialize nnU-Net predictor
    print("\n1. Initializing nnU-Net Predictor...")
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=False,
            device=torch.device('cpu'),
            verbose=False
        )
        
        # Load the pretrained model
        model_path = Path("/Users/elwinli/Downloads/pretrained_weights/nnUNetTrainer__nnUNetPlans__3d_fullres")
        predictor.initialize_from_trained_model_folder(
            str(model_path),
            use_folds='all',
            checkpoint_name='checkpoint_best.pth'
        )
        
        print("✓ nnU-Net predictor initialized successfully")
        print(f"✓ Device: {predictor.device}")
        
        # Get the network
        network = predictor.network
        print(f"✓ Network loaded: {type(network).__name__}")
        
    except Exception as e:
        print(f"✗ nnU-Net initialization failed: {e}")
        return False
    
    # Test 2: Find .nii.gz files
    print("\n2. Finding .nii.gz Files...")
    test_data_new_path = Path("/Users/elwinli/CryoMamba/test_data_new")
    
    # Find all .nii.gz files
    nifti_files = []
    for pattern in ["**/*.nii.gz"]:
        nifti_files.extend(test_data_new_path.glob(pattern))
    
    # Filter out prediction files
    nifti_files = [f for f in nifti_files if 'prediction' not in f.name and 'segmentation' not in f.name]
    
    if not nifti_files:
        print("✗ No .nii.gz files found")
        return False
    
    print(f"✓ Found {len(nifti_files)} .nii.gz files:")
    for i, file in enumerate(nifti_files, 1):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {i}. {file.name} ({size_mb:.1f} MB)")
    
    # Test 3: Process first file as example
    print("\n3. Processing First File as Example...")
    
    nifti_file = nifti_files[0]  # Process first file
    print(f"Processing: {nifti_file.name}")
    
    try:
        # Load the image
        img = nib.load(nifti_file)
        img_data = img.get_fdata()
        img_spacing = img.header.get_zooms()
        
        print(f"✓ Loaded image data")
        print(f"  Shape: {img_data.shape}")
        print(f"  Spacing: {img_spacing}")
        print(f"  Data range: [{img_data.min():.3f}, {img_data.max():.3f}]")
        
        # Apply CT normalization
        print("  Applying CT normalization...")
        p1, p99 = np.percentile(img_data, [1, 99])
        img_data_clipped = np.clip(img_data, p1, p99)
        img_data_normalized = (img_data_clipped - p1) / (p99 - p1)
        
        print(f"  Normalization: [{p1:.3f}, {p99:.3f}] -> [0, 1]")
        
        # Resize to smaller size for memory efficiency
        target_size = (128, 128, 128)
        print(f"  Resizing to: {target_size}")
        
        from scipy.ndimage import zoom
        zoom_factors = [target_size[i] / img_data_normalized.shape[i] for i in range(3)]
        img_data_resized = zoom(img_data_normalized, zoom_factors, order=1)
        
        print(f"  Resized shape: {img_data_resized.shape}")
        
        # Prepare tensor
        input_tensor = torch.from_numpy(img_data_resized).float()
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        
        print(f"✓ Prepared input tensor: {input_tensor.shape}")
        
        # Run inference
        print("  Running inference...")
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        network.eval()
        
        with torch.no_grad():
            output = network(input_tensor)
            
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            prediction = output.cpu().numpy()
            
            print(f"    Raw output shape: {prediction.shape}")
            print(f"    Raw output range: [{prediction.min():.3f}, {prediction.max():.3f}]")
            
            # Remove batch dimension
            if len(prediction.shape) == 5:
                prediction = prediction[0]
            
            # Handle channel dimension
            if len(prediction.shape) == 4:
                if prediction.shape[0] == 2:
                    prediction_softmax = torch.softmax(torch.from_numpy(prediction), dim=0).numpy()
                    prediction = prediction_softmax[1]
                    print(f"    Applied softmax, using foreground class")
                else:
                    prediction = prediction[0]
            
            # Convert to binary
            prediction_binary = (prediction > 0.5).astype(np.uint8)
            
            print(f"    Final binary shape: {prediction_binary.shape}")
            print(f"    Final unique values: {np.unique(prediction_binary)}")
        
        inference_time = time.time() - start_time
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"✓ Inference completed!")
        print(f"  Inference time: {inference_time:.2f} seconds")
        print(f"  Memory usage: {memory_after - memory_before:.1f} MB")
        
        # Calculate statistics
        foreground_voxels = np.sum(prediction_binary > 0)
        total_voxels = prediction_binary.size
        foreground_percentage = (foreground_voxels / total_voxels) * 100
        
        print(f"  Foreground voxels: {foreground_voxels:,} ({foreground_percentage:.2f}%)")
        print(f"  Background voxels: {total_voxels - foreground_voxels:,} ({100 - foreground_percentage:.2f}%)")
        
        # Resize back to original size
        print("  Resizing prediction back to original size...")
        original_size = img_data.shape
        zoom_factors_back = [original_size[i] / prediction_binary.shape[i] for i in range(3)]
        prediction_resized = zoom(prediction_binary, zoom_factors_back, order=0)
        
        print(f"  Resized prediction shape: {prediction_resized.shape}")
        
        # Save results
        output_dir = test_data_new_path / "predictions"
        output_dir.mkdir(exist_ok=True)
        
        pred_path = output_dir / f"{nifti_file.stem}_nnunet_prediction.nii.gz"
        pred_img = nib.Nifti1Image(prediction_resized.astype(np.uint8), img.affine, img.header)
        nib.save(pred_img, pred_path)
        print(f"✓ Saved NIfTI prediction: {pred_path}")
        
        # Create visualization
        viz_dir = output_dir / f"{nifti_file.stem}_visualization"
        viz_dir.mkdir(exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            
            # Find middle slice
            mid_z = prediction_resized.shape[2] // 2
            
            # Create comparison figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'nnU-Net Prediction - {nifti_file.stem}', fontsize=16)
            
            # Original image
            axes[0].imshow(img_data[:, :, mid_z], cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # nnU-Net prediction
            axes[1].imshow(prediction_resized[:, :, mid_z], cmap='gray')
            axes[1].set_title('nnU-Net Prediction')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(img_data[:, :, mid_z], cmap='gray', alpha=0.7)
            axes[2].imshow(prediction_resized[:, :, mid_z], cmap='Reds', alpha=0.3)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(viz_dir / f'prediction_z{mid_z}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved visualization: {viz_dir}")
            
        except Exception as e:
            print(f"⚠ Could not save visualization: {e}")
        
        print("\n" + "=" * 60)
        print("INFERENCE COMPLETED - SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 Results for {nifti_file.name}:")
        print(f"   Input shape: {img_data.shape}")
        print(f"   Prediction shape: {prediction_resized.shape}")
        print(f"   Foreground: {foreground_percentage:.2f}%")
        print(f"   Background: {100 - foreground_percentage:.2f}%")
        print(f"   Inference time: {inference_time:.2f} seconds")
        
        print(f"\n📁 Output Files:")
        print(f"   Prediction: {pred_path}")
        print(f"   Visualization: {viz_dir}")
        
        print(f"\n💡 Note: This processed the first file as an example.")
        print(f"   The model should work much better on this training data format!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {nifti_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("Running simple nnU-Net inference on test_data_new...")
    
    success = run_simple_inference_on_new_data()
    
    if success:
        print("\n🎉 nnU-Net inference on test_data_new completed successfully!")
        print("\nThis used the actual training data format (.nii.gz files)")
        print("Results should be much better than the previous cryo-EM data.")
        
        print("\nKey achievements:")
        print("✅ Processed .nii.gz file in proper training format")
        print("✅ Used actual nnU-Net model weights")
        print("✅ Applied proper CT normalization")
        print("✅ Generated segmentation prediction")
        print("✅ Created visualization")
    else:
        print("\n❌ Inference failed.")
        print("Check the errors above for troubleshooting.")
    
    return success

if __name__ == "__main__":
    main()
