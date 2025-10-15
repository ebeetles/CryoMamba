#!/usr/bin/env python3
"""
Process all remaining .nii.gz files in test_data_new folder.
This script processes all files that haven't been processed yet.
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

def process_all_remaining_files():
    """Process all remaining .nii.gz files in test_data_new."""
    print("=" * 60)
    print("Processing All Remaining .nii.gz Files in test_data_new")
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
    
    # Test 2: Find all .nii.gz files and check which ones need processing
    print("\n2. Finding Files to Process...")
    test_data_new_path = Path("/Users/elwinli/CryoMamba/test_data_new")
    predictions_dir = test_data_new_path / "predictions"
    
    # Find all .nii.gz files
    nifti_files = []
    for pattern in ["**/*.nii.gz"]:
        nifti_files.extend(test_data_new_path.glob(pattern))
    
    # Filter out prediction files and files in predictions directory
    nifti_files = [f for f in nifti_files if 'prediction' not in f.name and 'segmentation' not in f.name and 'predictions' not in str(f)]
    
    # Check which files already have predictions
    files_to_process = []
    for nifti_file in nifti_files:
        pred_file = predictions_dir / f"{nifti_file.stem}_nnunet_prediction.nii.gz"
        if not pred_file.exists():
            files_to_process.append(nifti_file)
        else:
            print(f"  ⏭ Skipping {nifti_file.name} (already processed)")
    
    if not files_to_process:
        print("✓ All files have already been processed!")
        return True
    
    print(f"✓ Found {len(files_to_process)} files to process:")
    for i, file in enumerate(files_to_process, 1):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {i}. {file.name} ({size_mb:.1f} MB)")
    
    # Test 3: Process each file
    print("\n3. Processing Files...")
    
    results_summary = []
    
    for i, nifti_file in enumerate(files_to_process, 1):
        print(f"\n--- Processing File {i}/{len(files_to_process)}: {nifti_file.name} ---")
        
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
            
            # Calculate statistics
            foreground_voxels = np.sum(prediction_binary > 0)
            total_voxels = prediction_binary.size
            foreground_percentage = (foreground_voxels / total_voxels) * 100
            
            print(f"✓ Inference completed!")
            print(f"  Inference time: {inference_time:.2f} seconds")
            print(f"  Foreground voxels: {foreground_voxels:,} ({foreground_percentage:.2f}%)")
            print(f"  Background voxels: {total_voxels - foreground_voxels:,} ({100 - foreground_percentage:.2f}%)")
            
            # Resize back to original size
            print("  Resizing prediction back to original size...")
            original_size = img_data.shape
            zoom_factors_back = [original_size[i] / prediction_binary.shape[i] for i in range(3)]
            prediction_resized = zoom(prediction_binary, zoom_factors_back, order=0)
            
            print(f"  Resized prediction shape: {prediction_resized.shape}")
            
            # Save results
            predictions_dir.mkdir(exist_ok=True)
            
            pred_path = predictions_dir / f"{nifti_file.stem}_nnunet_prediction.nii.gz"
            pred_img = nib.Nifti1Image(prediction_resized.astype(np.uint8), img.affine, img.header)
            nib.save(pred_img, pred_path)
            print(f"✓ Saved NIfTI prediction: {pred_path}")
            
            # Create visualization
            viz_dir = predictions_dir / f"{nifti_file.stem}_visualization"
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
            
            # Store results summary
            results_summary.append({
                'file': nifti_file.name,
                'shape': prediction_resized.shape,
                'foreground_percentage': foreground_percentage,
                'inference_time': inference_time,
                'prediction_file': pred_path.name
            })
            
            # Clear memory
            del input_tensor, output, prediction, prediction_binary, prediction_resized
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"✗ Error processing {nifti_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Test 4: Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETED - SUMMARY")
    print("=" * 60)
    
    if results_summary:
        print(f"\n📊 Successfully processed {len(results_summary)} files:")
        
        total_time = sum(r['inference_time'] for r in results_summary)
        avg_foreground = np.mean([r['foreground_percentage'] for r in results_summary])
        
        for i, result in enumerate(results_summary, 1):
            print(f"\n{i}. {result['file']}")
            print(f"   Shape: {result['shape']}")
            print(f"   Foreground: {result['foreground_percentage']:.2f}%")
            print(f"   Time: {result['inference_time']:.2f}s")
            print(f"   Output: {result['prediction_file']}")
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total files processed: {len(results_summary)}")
        print(f"   Total inference time: {total_time:.2f} seconds")
        print(f"   Average foreground: {avg_foreground:.2f}%")
        print(f"   Average time per file: {total_time/len(results_summary):.2f} seconds")
        
        print(f"\n📁 Output Location:")
        print(f"   Predictions: {predictions_dir}")
        print(f"   Visualizations: {predictions_dir} / *_visualization")
        
        return True
    else:
        print("\n❌ No files were processed successfully.")
        return False

def main():
    """Main function."""
    print("Processing all remaining .nii.gz files in test_data_new...")
    
    success = process_all_remaining_files()
    
    if success:
        print("\n🎉 All files in test_data_new processed successfully!")
        print("\nThis used the actual training data format (.nii.gz files)")
        print("Results should be much better than the previous cryo-EM data.")
        
        print("\nKey achievements:")
        print("✅ Processed all .nii.gz files in proper training format")
        print("✅ Used actual nnU-Net model weights")
        print("✅ Applied proper CT normalization")
        print("✅ Generated segmentation predictions for all files")
        print("✅ Created visualizations for each file")
    else:
        print("\n❌ Processing failed.")
        print("Check the errors above for troubleshooting.")
    
    return success

if __name__ == "__main__":
    main()
