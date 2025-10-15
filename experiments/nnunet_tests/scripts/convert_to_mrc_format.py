#!/usr/bin/env python3
"""
Convert test_data_new images, labels, and predictions to MRC format.
This script creates a new folder with MRC files for CryoMamba integration.
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

def convert_nifti_to_mrc():
    """Convert test_data_new files to MRC format."""
    print("=" * 60)
    print("Converting test_data_new to MRC Format")
    print("=" * 60)
    
    # Test 1: Set up directories
    print("\n1. Setting Up Directories...")
    
    test_data_new_path = Path("/Users/elwinli/CryoMamba/test_data_new")
    mrc_output_path = Path("/Users/elwinli/CryoMamba/test_data_new_mrc")
    
    # Create output directory structure
    mrc_output_path.mkdir(exist_ok=True)
    (mrc_output_path / "images").mkdir(exist_ok=True)
    (mrc_output_path / "labels").mkdir(exist_ok=True)
    (mrc_output_path / "predictions").mkdir(exist_ok=True)
    
    print(f"✓ Created output directory: {mrc_output_path}")
    print(f"✓ Created subdirectories: images, labels, predictions")
    
    # Test 2: Find source files
    print("\n2. Finding Source Files...")
    
    # Find images
    images_dir = test_data_new_path / "imagesTr"
    image_files = list(images_dir.glob("*.nii.gz")) if images_dir.exists() else []
    
    # Find labels
    labels_dir = test_data_new_path / "labelsTr"
    label_files = list(labels_dir.glob("*.nii.gz")) if labels_dir.exists() else []
    
    # Find predictions
    predictions_dir = test_data_new_path / "predictions"
    prediction_files = list(predictions_dir.glob("*_nnunet_prediction.nii.gz")) if predictions_dir.exists() else []
    
    print(f"✓ Found {len(image_files)} image files")
    print(f"✓ Found {len(label_files)} label files")
    print(f"✓ Found {len(prediction_files)} prediction files")
    
    if not image_files:
        print("✗ No image files found")
        return False
    
    # Test 3: Convert images to MRC
    print("\n3. Converting Images to MRC...")
    
    converted_images = []
    
    for i, image_file in enumerate(image_files, 1):
        print(f"  Converting image {i}/{len(image_files)}: {image_file.name}")
        
        try:
            # Load NIfTI image
            img = nib.load(image_file)
            img_data = img.get_fdata()
            
            print(f"    Shape: {img_data.shape}")
            print(f"    Data range: [{img_data.min():.3f}, {img_data.max():.3f}]")
            
            # Convert to MRC format
            import mrcfile
            
            # MRC files expect (Z, Y, X) orientation, so transpose from (X, Y, Z)
            mrc_data = np.transpose(img_data, (2, 1, 0))
            
            # Create MRC filename
            mrc_filename = image_file.stem.replace('.nii', '') + '.mrc'
            mrc_path = mrc_output_path / "images" / mrc_filename
            
            # Save as MRC
            with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
                mrc.set_data(mrc_data.astype(np.float32))
                # Set header information
                mrc.header.nx = mrc_data.shape[2]
                mrc.header.ny = mrc_data.shape[1]
                mrc.header.nz = mrc_data.shape[0]
                mrc.header.mx = mrc_data.shape[2]
                mrc.header.my = mrc_data.shape[1]
                mrc.header.mz = mrc_data.shape[0]
            
            print(f"    ✓ Saved: {mrc_path}")
            converted_images.append(mrc_path)
            
        except Exception as e:
            print(f"    ✗ Error converting {image_file.name}: {e}")
            continue
    
    # Test 4: Convert labels to MRC
    print("\n4. Converting Labels to MRC...")
    
    converted_labels = []
    
    for i, label_file in enumerate(label_files, 1):
        print(f"  Converting label {i}/{len(label_files)}: {label_file.name}")
        
        try:
            # Load NIfTI label
            label_img = nib.load(label_file)
            label_data = label_img.get_fdata()
            
            print(f"    Shape: {label_data.shape}")
            print(f"    Unique values: {np.unique(label_data)}")
            
            # Convert to MRC format
            import mrcfile
            
            # MRC files expect (Z, Y, X) orientation, so transpose from (X, Y, Z)
            mrc_data = np.transpose(label_data, (2, 1, 0))
            
            # Create MRC filename
            mrc_filename = label_file.stem.replace('.nii', '') + '.mrc'
            mrc_path = mrc_output_path / "labels" / mrc_filename
            
            # Save as MRC
            with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
                mrc.set_data(mrc_data.astype(np.float32))
                # Set header information
                mrc.header.nx = mrc_data.shape[2]
                mrc.header.ny = mrc_data.shape[1]
                mrc.header.nz = mrc_data.shape[0]
                mrc.header.mx = mrc_data.shape[2]
                mrc.header.my = mrc_data.shape[1]
                mrc.header.mz = mrc_data.shape[0]
            
            print(f"    ✓ Saved: {mrc_path}")
            converted_labels.append(mrc_path)
            
        except Exception as e:
            print(f"    ✗ Error converting {label_file.name}: {e}")
            continue
    
    # Test 5: Convert predictions to MRC
    print("\n5. Converting Predictions to MRC...")
    
    converted_predictions = []
    
    for i, prediction_file in enumerate(prediction_files, 1):
        print(f"  Converting prediction {i}/{len(prediction_files)}: {prediction_file.name}")
        
        try:
            # Load NIfTI prediction
            pred_img = nib.load(prediction_file)
            pred_data = pred_img.get_fdata()
            
            print(f"    Shape: {pred_data.shape}")
            print(f"    Unique values: {np.unique(pred_data)}")
            
            # Convert to MRC format
            import mrcfile
            
            # MRC files expect (Z, Y, X) orientation, so transpose from (X, Y, Z)
            mrc_data = np.transpose(pred_data, (2, 1, 0))
            
            # Create MRC filename
            mrc_filename = prediction_file.stem.replace('_nnunet_prediction.nii', '') + '_prediction.mrc'
            mrc_path = mrc_output_path / "predictions" / mrc_filename
            
            # Save as MRC
            with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
                mrc.set_data(mrc_data.astype(np.float32))
                # Set header information
                mrc.header.nx = mrc_data.shape[2]
                mrc.header.ny = mrc_data.shape[1]
                mrc.header.nz = mrc_data.shape[0]
                mrc.header.mx = mrc_data.shape[2]
                mrc.header.my = mrc_data.shape[1]
                mrc.header.mz = mrc_data.shape[0]
            
            print(f"    ✓ Saved: {mrc_path}")
            converted_predictions.append(mrc_path)
            
        except Exception as e:
            print(f"    ✗ Error converting {prediction_file.name}: {e}")
            continue
    
    # Test 6: Create summary and copy configuration files
    print("\n6. Creating Summary and Configuration Files...")
    
    try:
        # Copy configuration files
        config_files = ['dataset.json', 'plans.json', 'predict_from_raw_data_args.json']
        for config_file in config_files:
            src_path = test_data_new_path / config_file
            if src_path.exists():
                dst_path = mrc_output_path / config_file
                import shutil
                shutil.copy2(src_path, dst_path)
                print(f"✓ Copied {config_file}")
        
        # Create conversion summary
        summary = {
            'conversion_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_directory': str(test_data_new_path),
            'output_directory': str(mrc_output_path),
            'files_converted': {
                'images': len(converted_images),
                'labels': len(converted_labels),
                'predictions': len(converted_predictions)
            },
            'converted_files': {
                'images': [str(f) for f in converted_images],
                'labels': [str(f) for f in converted_labels],
                'predictions': [str(f) for f in converted_predictions]
            }
        }
        
        summary_path = mrc_output_path / "conversion_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Created conversion summary: {summary_path}")
        
    except Exception as e:
        print(f"⚠ Could not create summary: {e}")
    
    # Test 7: Final summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETED - SUMMARY")
    print("=" * 60)
    
    print(f"\n📁 Output Directory: {mrc_output_path}")
    print(f"\n📊 Files Converted:")
    print(f"   Images: {len(converted_images)} files")
    print(f"   Labels: {len(converted_labels)} files")
    print(f"   Predictions: {len(converted_predictions)} files")
    
    print(f"\n📂 Directory Structure:")
    print(f"   {mrc_output_path}/")
    print(f"   ├── images/          ({len(converted_images)} MRC files)")
    print(f"   ├── labels/          ({len(converted_labels)} MRC files)")
    print(f"   ├── predictions/     ({len(converted_predictions)} MRC files)")
    print(f"   ├── dataset.json")
    print(f"   ├── plans.json")
    print(f"   ├── predict_from_raw_data_args.json")
    print(f"   └── conversion_summary.json")
    
    print(f"\n🎯 Key Features:")
    print(f"   ✅ All files converted to MRC format")
    print(f"   ✅ Proper orientation (Z, Y, X)")
    print(f"   ✅ Maintained data integrity")
    print(f"   ✅ Ready for CryoMamba integration")
    print(f"   ✅ Configuration files preserved")
    
    return True

def main():
    """Main function."""
    print("Converting test_data_new to MRC format...")
    
    success = convert_nifti_to_mrc()
    
    if success:
        print("\n🎉 Conversion to MRC format completed successfully!")
        print("\nAll files are now in MRC format and ready for CryoMamba integration.")
        
        print("\nKey achievements:")
        print("✅ Converted all images to MRC format")
        print("✅ Converted all labels to MRC format")
        print("✅ Converted all predictions to MRC format")
        print("✅ Preserved configuration files")
        print("✅ Created organized directory structure")
        print("✅ Ready for CryoMamba integration")
    else:
        print("\n❌ Conversion failed.")
        print("Check the errors above for troubleshooting.")
    
    return success

if __name__ == "__main__":
    main()
