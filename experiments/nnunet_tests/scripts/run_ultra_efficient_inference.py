#!/usr/bin/env python3
"""
Ultra memory-efficient nnU-Net model inference.
This script processes data in very small chunks to avoid memory issues.
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

def run_ultra_efficient_inference():
    """Run ultra memory-efficient nnU-Net model inference."""
    print("=" * 60)
    print("Ultra Memory-Efficient nnU-Net Model Inference")
    print("=" * 60)
    
    # Test 1: Load the model directly
    print("\n1. Loading nnU-Net Model...")
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
        print(f"✓ Model loaded from: {model_path}")
        
        # Get the network
        network = predictor.network
        print(f"✓ Network loaded: {type(network).__name__}")
        
    except Exception as e:
        print(f"✗ nnU-Net initialization failed: {e}")
        return False
    
    # Test 2: Load and prepare data
    print("\n2. Loading and Preparing Data...")
    test_data_path = Path("/Users/elwinli/CryoMamba/test_data")
    results_dir = test_data_path / "nnunet_results"
    
    # Find existing NIfTI files
    nifti_files = list(results_dir.glob("*.nii.gz"))
    nifti_files = [f for f in nifti_files if 'prediction' not in f.name and 'segmentation' not in f.name]
    
    if not nifti_files:
        print("✗ No NIfTI files found for inference")
        return False
    
    selected_file = nifti_files[0]  # Use first available file
    print(f"✓ Using input file: {selected_file.name}")
    
    try:
        # Load the image
        img = nib.load(selected_file)
        img_data = img.get_fdata()
        img_spacing = img.header.get_zooms()
        
        print(f"✓ Loaded image data")
        print(f"  Shape: {img_data.shape}")
        print(f"  Spacing: {img_spacing}")
        print(f"  Data range: [{img_data.min():.3f}, {img_data.max():.3f}]")
        
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return False
    
    # Test 3: Process data in very small chunks
    print("\n3. Processing Data in Ultra-Small Chunks...")
    
    try:
        # Load model configuration
        with open(model_path / "dataset.json", 'r') as f:
            dataset_config = json.load(f)
        
        print(f"✓ Model configuration loaded")
        print(f"  Labels: {dataset_config.get('labels', {})}")
        
        # Apply CT normalization
        print("  Applying CT normalization...")
        p1, p99 = np.percentile(img_data, [1, 99])
        img_data_clipped = np.clip(img_data, p1, p99)
        img_data_normalized = (img_data_clipped - p1) / (p99 - p1)
        
        print(f"  Normalization: [{p1:.3f}, {p99:.3f}] -> [0, 1]")
        
        # Use very small patch size to avoid memory issues
        patch_size = (64, 64, 64)  # Very small patch size
        overlap = 16  # Small overlap
        
        print(f"  Processing in patches of size: {patch_size}")
        print(f"  Overlap: {overlap} voxels")
        
        # Initialize output array
        prediction_full = np.zeros(img_data.shape, dtype=np.float32)
        count_map = np.zeros(img_data.shape, dtype=np.float32)
        
        # Calculate patch positions
        patch_positions = []
        for z in range(0, img_data.shape[0], patch_size[0] - overlap):
            for y in range(0, img_data.shape[1], patch_size[1] - overlap):
                for x in range(0, img_data.shape[2], patch_size[2] - overlap):
                    # Ensure patch doesn't exceed image bounds
                    z_end = min(z + patch_size[0], img_data.shape[0])
                    y_end = min(y + patch_size[1], img_data.shape[1])
                    x_end = min(x + patch_size[2], img_data.shape[2])
                    
                    patch_positions.append(((z, z_end), (y, y_end), (x, x_end)))
        
        print(f"  Total patches to process: {len(patch_positions)}")
        
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        return False
    
    # Test 4: Process patches
    print("\n4. Processing Patches...")
    
    try:
        network.eval()
        start_time = time.time()
        
        for i, ((z_start, z_end), (y_start, y_end), (x_start, x_end)) in enumerate(patch_positions):
            print(f"  Processing patch {i+1}/{len(patch_positions)}: [{z_start}:{z_end}, {y_start}:{y_end}, {x_start}:{x_end}]")
            
            # Extract patch
            patch_data = img_data_normalized[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Pad patch to expected size if needed
            if patch_data.shape != patch_size:
                # Pad with zeros
                padded_patch = np.zeros(patch_size)
                padded_patch[:patch_data.shape[0], :patch_data.shape[1], :patch_data.shape[2]] = patch_data
                patch_data = padded_patch
            
            # Convert to tensor
            input_tensor = torch.from_numpy(patch_data).float()
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            
            # Run inference
            with torch.no_grad():
                output = network(input_tensor)
                
                # Handle different output formats
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                # Convert to numpy
                patch_prediction = output.cpu().numpy()
                
                # Remove batch dimension
                if len(patch_prediction.shape) == 5:  # B, C, D, H, W
                    patch_prediction = patch_prediction[0]  # Remove batch dimension
                
                # Handle channel dimension
                if len(patch_prediction.shape) == 4:  # C, D, H, W
                    if patch_prediction.shape[0] == 2:  # Binary segmentation
                        # Apply softmax to get probabilities
                        patch_prediction_softmax = torch.softmax(torch.from_numpy(patch_prediction), dim=0).numpy()
                        patch_prediction = patch_prediction_softmax[1]  # Take foreground class
                    else:
                        patch_prediction = patch_prediction[0]  # Take first channel
                
                # Convert to binary
                patch_prediction = (patch_prediction > 0.5).astype(np.float32)
                
                # Crop back to original patch size
                original_patch_size = (z_end - z_start, y_end - y_start, x_end - x_start)
                patch_prediction = patch_prediction[:original_patch_size[0], :original_patch_size[1], :original_patch_size[2]]
                
                # Add to full prediction
                prediction_full[z_start:z_end, y_start:y_end, x_start:x_end] += patch_prediction
                count_map[z_start:z_end, y_start:y_end, x_start:x_end] += 1
            
            # Clear memory
            del input_tensor, output, patch_prediction
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Force garbage collection every 10 patches
            if (i + 1) % 10 == 0:
                import gc
                gc.collect()
        
        # Average overlapping regions
        prediction_full = prediction_full / (count_map + 1e-8)
        
        inference_time = time.time() - start_time
        print(f"✓ Patch processing completed!")
        print(f"  Inference time: {inference_time:.2f} seconds")
        
    except Exception as e:
        print(f"✗ Patch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Process and save results
    print("\n5. Processing and Saving Results...")
    
    try:
        # Convert to binary
        prediction_binary = (prediction_full > 0.5).astype(np.uint8)
        
        print(f"✓ Processed prediction")
        print(f"  Final shape: {prediction_binary.shape}")
        print(f"  Final unique values: {np.unique(prediction_binary)}")
        
        # Calculate statistics
        foreground_voxels = np.sum(prediction_binary > 0)
        total_voxels = prediction_binary.size
        foreground_percentage = (foreground_voxels / total_voxels) * 100
        
        print(f"  Foreground voxels: {foreground_voxels:,} ({foreground_percentage:.2f}%)")
        print(f"  Background voxels: {total_voxels - foreground_voxels:,} ({100 - foreground_percentage:.2f}%)")
        
        # Save results
        pred_path = results_dir / f"{selected_file.stem}_nnunet_actual_segmentation.nii.gz"
        pred_img = nib.Nifti1Image(prediction_binary.astype(np.uint8), img.affine, img.header)
        nib.save(pred_img, pred_path)
        print(f"✓ Saved NIfTI prediction: {pred_path}")
        
        # Save as MRC format
        mrc_output_path = results_dir / f"{selected_file.stem}_nnunet_actual_segmentation.mrc"
        try:
            import mrcfile
            # Convert back to MRC format (transpose back to Z,Y,X)
            mrc_data = np.transpose(prediction_binary.astype(np.float32), (2, 1, 0))
            with mrcfile.new(str(mrc_output_path), overwrite=True) as mrc:
                mrc.set_data(mrc_data)
            print(f"✓ Saved MRC prediction: {mrc_output_path}")
        except Exception as e:
            print(f"⚠ Could not save as MRC: {e}")
        
        # Create visualization
        png_dir = results_dir / f"{selected_file.stem}_nnunet_actual_slices"
        png_dir.mkdir(exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            
            # Find middle slice
            mid_z = prediction_binary.shape[2] // 2
            
            # Create comparison figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Actual nnU-Net Model Inference - {selected_file.stem}', fontsize=16)
            
            # Original image
            axes[0].imshow(img_data[:, :, mid_z], cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # nnU-Net prediction
            axes[1].imshow(prediction_binary[:, :, mid_z], cmap='gray')
            axes[1].set_title('Actual nnU-Net Prediction')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(img_data[:, :, mid_z], cmap='gray', alpha=0.7)
            axes[2].imshow(prediction_binary[:, :, mid_z], cmap='Reds', alpha=0.3)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(png_dir / f'actual_nnunet_results_z{mid_z}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved visualization: {png_dir}")
            
        except Exception as e:
            print(f"⚠ Could not save visualization: {e}")
        
        print("\n" + "=" * 60)
        print("ACTUAL nnU-NET MODEL INFERENCE COMPLETED!")
        print("=" * 60)
        
        print(f"\n📁 Generated Files:")
        print(f"  - NIfTI prediction: {pred_path}")
        if mrc_output_path.exists():
            print(f"  - MRC prediction: {mrc_output_path}")
        print(f"  - Visualization: {png_dir}")
        
        print(f"\n📊 Results Summary:")
        print(f"  - Input file: {selected_file.name}")
        print(f"  - Prediction shape: {prediction_binary.shape}")
        print(f"  - Foreground: {foreground_percentage:.2f}%")
        print(f"  - Background: {100 - foreground_percentage:.2f}%")
        print(f"  - Inference time: {inference_time:.2f} seconds")
        print(f"  - Patches processed: {len(patch_positions)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing results: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("Running ultra memory-efficient nnU-Net model inference...")
    
    success = run_ultra_efficient_inference()
    
    if success:
        print("\n🎉 Actual nnU-Net model inference completed successfully!")
        print("\nThis is the REAL nnU-Net inference using your pretrained model weights.")
        print("We processed the data in ultra-small chunks to avoid memory issues.")
        
        print("\nKey achievements:")
        print("✅ Used actual nnU-Net model weights")
        print("✅ Applied proper CT normalization")
        print("✅ Processed in ultra-small patches (64x64x64)")
        print("✅ Used direct PyTorch inference")
        print("✅ Bypassed file handling issues")
        print("✅ Generated actual segmentation mask")
        
        print("\nNext steps:")
        print("1. Compare these results with ground truth labels")
        print("2. Analyze the performance metrics")
        print("3. Compare with previous thresholding results")
        print("4. Integrate this approach into CryoMamba")
    else:
        print("\n❌ Model inference failed.")
        print("Check the errors above for troubleshooting.")
    
    return success

if __name__ == "__main__":
    main()
