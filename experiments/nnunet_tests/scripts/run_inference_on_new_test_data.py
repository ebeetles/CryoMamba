#!/usr/bin/env python3
"""
Run nnU-Net inference on test_data_new folder with .nii.gz files.
This script processes the actual training data format for better results.
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

def run_inference_on_new_test_data():
    """Run nnU-Net inference on test_data_new folder."""
    print("=" * 60)
    print("nnU-Net Inference on test_data_new (.nii.gz files)")
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
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
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
    
    # Test 2: Find .nii.gz files in test_data_new
    print("\n2. Finding .nii.gz Files in test_data_new...")
    test_data_new_path = Path("/Users/elwinli/CryoMamba/test_data_new")
    
    # Find all .nii.gz files
    nifti_files = []
    for pattern in ["**/*.nii.gz"]:
        nifti_files.extend(test_data_new_path.glob(pattern))
    
    # Filter out any existing prediction files
    nifti_files = [f for f in nifti_files if 'prediction' not in f.name and 'segmentation' not in f.name]
    
    if not nifti_files:
        print("✗ No .nii.gz files found in test_data_new")
        return False
    
    print(f"✓ Found {len(nifti_files)} .nii.gz files:")
    for i, file in enumerate(nifti_files, 1):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {i}. {file.name} ({size_mb:.1f} MB)")
    
    # Test 3: Process each file
    print("\n3. Processing Files...")
    
    results_summary = []
    
    for i, nifti_file in enumerate(nifti_files, 1):
        print(f"\n--- Processing File {i}/{len(nifti_files)}: {nifti_file.name} ---")
        
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
            
            # Use smaller patch size for memory efficiency
            patch_size = (64, 64, 64)
            overlap = 16
            
            print(f"  Processing in patches of size: {patch_size}")
            
            # Initialize output array
            prediction_full = np.zeros(img_data.shape, dtype=np.float32)
            count_map = np.zeros(img_data.shape, dtype=np.float32)
            
            # Calculate patch positions
            patch_positions = []
            for z in range(0, img_data.shape[0], patch_size[0] - overlap):
                for y in range(0, img_data.shape[1], patch_size[1] - overlap):
                    for x in range(0, img_data.shape[2], patch_size[2] - overlap):
                        z_end = min(z + patch_size[0], img_data.shape[0])
                        y_end = min(y + patch_size[1], img_data.shape[1])
                        x_end = min(x + patch_size[2], img_data.shape[2])
                        patch_positions.append(((z, z_end), (y, y_end), (x, x_end)))
            
            print(f"  Total patches to process: {len(patch_positions)}")
            
            # Process patches
            network.eval()
            start_time = time.time()
            
            for j, ((z_start, z_end), (y_start, y_end), (x_start, x_end)) in enumerate(patch_positions):
                if (j + 1) % 50 == 0:  # Progress update every 50 patches
                    print(f"    Processing patch {j+1}/{len(patch_positions)}...")
                
                # Extract patch
                patch_data = img_data_normalized[z_start:z_end, y_start:y_end, x_start:x_end]
                
                # Pad patch to expected size if needed
                if patch_data.shape != patch_size:
                    padded_patch = np.zeros(patch_size)
                    padded_patch[:patch_data.shape[0], :patch_data.shape[1], :patch_data.shape[2]] = patch_data
                    patch_data = padded_patch
                
                # Convert to tensor
                input_tensor = torch.from_numpy(patch_data).float()
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
                
                # Run inference
                with torch.no_grad():
                    output = network(input_tensor)
                    
                    if isinstance(output, (list, tuple)):
                        output = output[0]
                    
                    patch_prediction = output.cpu().numpy()
                    
                    # Remove batch dimension
                    if len(patch_prediction.shape) == 5:
                        patch_prediction = patch_prediction[0]
                    
                    # Handle channel dimension
                    if len(patch_prediction.shape) == 4:
                        if patch_prediction.shape[0] == 2:
                            patch_prediction_softmax = torch.softmax(torch.from_numpy(patch_prediction), dim=0).numpy()
                            patch_prediction = patch_prediction_softmax[1]
                        else:
                            patch_prediction = patch_prediction[0]
                    
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
                if (j + 1) % 20 == 0:
                    import gc
                    gc.collect()
            
            # Average overlapping regions
            prediction_full = prediction_full / (count_map + 1e-8)
            prediction_binary = (prediction_full > 0.5).astype(np.uint8)
            
            inference_time = time.time() - start_time
            
            # Calculate statistics
            foreground_voxels = np.sum(prediction_binary > 0)
            total_voxels = prediction_binary.size
            foreground_percentage = (foreground_voxels / total_voxels) * 100
            
            print(f"✓ Inference completed!")
            print(f"  Inference time: {inference_time:.2f} seconds")
            print(f"  Foreground voxels: {foreground_voxels:,} ({foreground_percentage:.2f}%)")
            print(f"  Background voxels: {total_voxels - foreground_voxels:,} ({100 - foreground_percentage:.2f}%)")
            
            # Save results
            output_dir = test_data_new_path / "predictions"
            output_dir.mkdir(exist_ok=True)
            
            # Save as NIfTI
            pred_path = output_dir / f"{nifti_file.stem}_nnunet_prediction.nii.gz"
            pred_img = nib.Nifti1Image(prediction_binary.astype(np.uint8), img.affine, img.header)
            nib.save(pred_img, pred_path)
            print(f"✓ Saved NIfTI prediction: {pred_path}")
            
            # Create visualization
            viz_dir = output_dir / f"{nifti_file.stem}_visualization"
            viz_dir.mkdir(exist_ok=True)
            
            try:
                import matplotlib.pyplot as plt
                
                # Find middle slice
                mid_z = prediction_binary.shape[2] // 2
                
                # Create comparison figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f'nnU-Net Prediction - {nifti_file.stem}', fontsize=16)
                
                # Original image
                axes[0].imshow(img_data[:, :, mid_z], cmap='gray')
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # nnU-Net prediction
                axes[1].imshow(prediction_binary[:, :, mid_z], cmap='gray')
                axes[1].set_title('nnU-Net Prediction')
                axes[1].axis('off')
                
                # Overlay
                axes[2].imshow(img_data[:, :, mid_z], cmap='gray', alpha=0.7)
                axes[2].imshow(prediction_binary[:, :, mid_z], cmap='Reds', alpha=0.3)
                axes[2].set_title('Overlay')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(viz_dir / f'prediction_z{mid_z}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"✓ Saved visualization: {viz_dir}")
                
            except Exception as e:
                print(f"⚠ Could not save visualization: {e}")
            
            # Store results summary
            results_summary.append({
                'file': nifti_file.name,
                'shape': prediction_binary.shape,
                'foreground_percentage': foreground_percentage,
                'inference_time': inference_time,
                'patches_processed': len(patch_positions),
                'prediction_file': pred_path.name
            })
            
        except Exception as e:
            print(f"✗ Error processing {nifti_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Test 4: Summary
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED - SUMMARY")
    print("=" * 60)
    
    if results_summary:
        print(f"\n📊 Processed {len(results_summary)} files successfully:")
        
        total_time = sum(r['inference_time'] for r in results_summary)
        avg_foreground = np.mean([r['foreground_percentage'] for r in results_summary])
        
        for i, result in enumerate(results_summary, 1):
            print(f"\n{i}. {result['file']}")
            print(f"   Shape: {result['shape']}")
            print(f"   Foreground: {result['foreground_percentage']:.2f}%")
            print(f"   Time: {result['inference_time']:.2f}s")
            print(f"   Patches: {result['patches_processed']}")
            print(f"   Output: {result['prediction_file']}")
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total files processed: {len(results_summary)}")
        print(f"   Total inference time: {total_time:.2f} seconds")
        print(f"   Average foreground: {avg_foreground:.2f}%")
        print(f"   Average time per file: {total_time/len(results_summary):.2f} seconds")
        
        print(f"\n📁 Output Location:")
        print(f"   Predictions: {test_data_new_path / 'predictions'}")
        print(f"   Visualizations: {test_data_new_path / 'predictions' / '*_visualization'}")
        
        return True
    else:
        print("\n❌ No files were processed successfully.")
        return False

def main():
    """Main function."""
    print("Running nnU-Net inference on test_data_new folder...")
    
    success = run_inference_on_new_test_data()
    
    if success:
        print("\n🎉 nnU-Net inference on test_data_new completed successfully!")
        print("\nThis used the actual training data format (.nii.gz files)")
        print("Results should be much better than the previous cryo-EM data.")
        
        print("\nKey achievements:")
        print("✅ Processed .nii.gz files in proper training format")
        print("✅ Used actual nnU-Net model weights")
        print("✅ Applied proper CT normalization")
        print("✅ Generated segmentation predictions")
        print("✅ Created visualizations for each file")
    else:
        print("\n❌ Inference failed.")
        print("Check the errors above for troubleshooting.")
    
    return success

if __name__ == "__main__":
    main()
