#!/usr/bin/env python3
"""
Test script for nnU-Net model validation before integration into CryoMamba.
This script tests the pretrained model with the provided test data.
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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Add nnU-Net to path if available
try:
    import nnunetv2
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.file_path_utilities import get_identifier_from_cfg_name
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
    NNUNET_AVAILABLE = True
    print("✓ nnU-Net imported successfully")
except ImportError as e:
    print(f"Warning: nnU-Net not available. Error: {e}")
    NNUNET_AVAILABLE = False

class NNUModelTester:
    """Test class for nnU-Net model validation."""
    
    def __init__(self, 
                 model_path: str = "/Users/elwinli/Downloads/pretrained_weights/nnUNetTrainer__nnUNetPlans__3d_fullres",
                 test_data_path: str = "/Users/elwinli/Downloads/test_data"):
        """
        Initialize the tester.
        
        Args:
            model_path: Path to the pretrained model directory
            test_data_path: Path to the test data directory
        """
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration files
        self.dataset_json = self._load_json(self.model_path / "dataset.json")
        self.plans_json = self._load_json(self.model_path / "plans.json")
        self.debug_json = self._load_json(self.model_path / "fold_all" / "debug.json")
        
        print(f"Initialized NNUModelTester:")
        print(f"  Model path: {self.model_path}")
        print(f"  Test data path: {self.test_data_path}")
        print(f"  Device: {self.device}")
        print(f"  Dataset: {self.dataset_json.get('labels', {})}")
        
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def analyze_model_configuration(self) -> Dict[str, Any]:
        """Analyze the model configuration and return key information."""
        config_3d = self.plans_json.get('configurations', {}).get('3d_fullres', {})
        
        analysis = {
            'dataset_name': self.plans_json.get('dataset_name', 'Unknown'),
            'patch_size': config_3d.get('patch_size', [192, 192, 64]),
            'batch_size': config_3d.get('batch_size', 2),
            'median_image_size': config_3d.get('median_image_size_in_voxels', [500, 500, 200]),
            'spacing': config_3d.get('spacing', [1.0, 1.0, 1.0]),
            'normalization': config_3d.get('normalization_schemes', ['CTNormalization']),
            'architecture': config_3d.get('architecture', {}),
            'labels': self.dataset_json.get('labels', {}),
            'num_training_samples': self.dataset_json.get('numTraining', 0),
            'file_ending': self.dataset_json.get('file_ending', '.nii.gz')
        }
        
        return analysis
    
    def check_model_files(self) -> Dict[str, bool]:
        """Check if all required model files are present."""
        required_files = {
            'dataset.json': self.model_path / "dataset.json",
            'plans.json': self.model_path / "plans.json",
            'checkpoint_best.pth': self.model_path / "fold_all" / "checkpoint_best.pth",
            'checkpoint_final.pth': self.model_path / "fold_all" / "checkpoint_final.pth",
            'debug.json': self.model_path / "fold_all" / "debug.json"
        }
        
        file_status = {}
        for name, path in required_files.items():
            file_status[name] = path.exists()
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {name}: {size_mb:.1f} MB")
            else:
                print(f"  ✗ {name}: Missing")
        
        return file_status
    
    def analyze_test_data(self) -> Dict[str, Any]:
        """Analyze the test data structure and properties."""
        images_dir = self.test_data_path / "imagesTr"
        labels_dir = self.test_data_path / "labelsTr"
        
        analysis = {
            'images_found': [],
            'labels_found': [],
            'image_properties': {},
            'label_properties': {},
            'data_compatibility': True
        }
        
        # Check images
        if images_dir.exists():
            for img_file in images_dir.glob("*.nii.gz"):
                analysis['images_found'].append(img_file.name)
                try:
                    img = nib.load(img_file)
                    analysis['image_properties'][img_file.name] = {
                        'shape': img.shape,
                        'spacing': img.header.get_zooms(),
                        'dtype': img.get_fdata().dtype,
                        'size_mb': img_file.stat().st_size / (1024 * 1024)
                    }
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
        
        # Check labels
        if labels_dir.exists():
            for label_file in labels_dir.glob("*.nii.gz"):
                analysis['labels_found'].append(label_file.name)
                try:
                    label = nib.load(label_file)
                    label_data = label.get_fdata()
                    analysis['label_properties'][label_file.name] = {
                        'shape': label.shape,
                        'spacing': label.header.get_zooms(),
                        'unique_values': np.unique(label_data),
                        'size_mb': label_file.stat().st_size / (1024 * 1024)
                    }
                except Exception as e:
                    print(f"Error loading {label_file}: {e}")
        
        return analysis
    
    def test_model_loading(self) -> bool:
        """Test if the model can be loaded successfully."""
        if not NNUNET_AVAILABLE:
            print("nnU-Net not available. Cannot test model loading.")
            return False
        
        try:
            print("\n=== Testing Model Loading ===")
            
            # Initialize predictor
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=self.device,
                verbose=False
            )
            
            # Load the model
            predictor.initialize_from_trained_model_folder(
                str(self.model_path),
                use_folds='all',
                checkpoint_name='checkpoint_best.pth'
            )
            
            print("✓ Model loaded successfully")
            print(f"  Configuration: {predictor.cfg}")
            print(f"  Network: {predictor.network}")
            print(f"  Device: {predictor.device}")
            
            return True
            
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            return False
    
    def test_inference_simple(self, test_image_path: str) -> Optional[np.ndarray]:
        """Test inference on a single test image."""
        if not NNUNET_AVAILABLE:
            print("nnU-Net not available. Cannot test inference.")
            return None
        
        try:
            print(f"\n=== Testing Inference on {Path(test_image_path).name} ===")
            
            # Initialize predictor
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=self.device,
                verbose=False
            )
            
            # Load the model
            predictor.initialize_from_trained_model_folder(
                str(self.model_path),
                use_folds='all',
                checkpoint_name='checkpoint_best.pth'
            )
            
            # Load test image
            img = nib.load(test_image_path)
            img_data = img.get_fdata()
            
            print(f"  Input shape: {img_data.shape}")
            print(f"  Input dtype: {img_data.dtype}")
            print(f"  Input range: [{img_data.min():.3f}, {img_data.max():.3f}]")
            
            # Perform inference
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Create temporary file for nnU-Net
            temp_input = self.test_data_path / "temp_input.nii.gz"
            temp_output = self.test_data_path / "temp_output.nii.gz"
            
            # Save as temporary file
            nib.save(img, temp_input)
            
            # Run prediction
            predictor.predict_from_files(
                [str(temp_input)],
                [str(temp_output)],
                save_probabilities=False,
                overwrite=True
            )
            
            # Load result
            result = nib.load(temp_output)
            result_data = result.get_fdata()
            
            inference_time = time.time() - start_time
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            print(f"  Output shape: {result_data.shape}")
            print(f"  Output dtype: {result_data.dtype}")
            print(f"  Output unique values: {np.unique(result_data)}")
            print(f"  Inference time: {inference_time:.2f} seconds")
            print(f"  Memory usage: {memory_after - memory_before:.1f} MB")
            
            # Clean up temporary files
            temp_input.unlink(missing_ok=True)
            temp_output.unlink(missing_ok=True)
            
            return result_data
            
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            return None
    
    def validate_output_quality(self, prediction: np.ndarray, ground_truth_path: str) -> Dict[str, float]:
        """Validate the quality of the prediction against ground truth."""
        try:
            gt = nib.load(ground_truth_path)
            gt_data = gt.get_fdata()
            
            # Ensure same shape
            if prediction.shape != gt_data.shape:
                print(f"Shape mismatch: pred {prediction.shape} vs gt {gt_data.shape}")
                return {}
            
            # Calculate metrics
            metrics = {}
            
            # Dice coefficient
            intersection = np.sum(prediction * gt_data)
            union = np.sum(prediction) + np.sum(gt_data)
            dice = (2.0 * intersection) / (union + 1e-8)
            metrics['dice'] = dice
            
            # IoU (Jaccard)
            iou = intersection / (union - intersection + 1e-8)
            metrics['iou'] = iou
            
            # Accuracy
            accuracy = np.sum(prediction == gt_data) / gt_data.size
            metrics['accuracy'] = accuracy
            
            # Sensitivity and Specificity
            tp = np.sum((prediction == 1) & (gt_data == 1))
            fp = np.sum((prediction == 1) & (gt_data == 0))
            fn = np.sum((prediction == 0) & (gt_data == 1))
            tn = np.sum((prediction == 0) & (gt_data == 0))
            
            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            
            metrics['sensitivity'] = sensitivity
            metrics['specificity'] = specificity
            
            print(f"  Dice: {dice:.3f}")
            print(f"  IoU: {iou:.3f}")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Sensitivity: {sensitivity:.3f}")
            print(f"  Specificity: {specificity:.3f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error validating output: {e}")
            return {}
    
    def create_visualization(self, 
                           input_data: np.ndarray, 
                           prediction: np.ndarray, 
                           ground_truth: np.ndarray,
                           output_path: str = "test_results.png"):
        """Create visualization of input, prediction, and ground truth."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('nnU-Net Model Test Results', fontsize=16)
            
            # Find middle slice
            mid_slice = input_data.shape[2] // 2
            
            # Input image
            axes[0, 0].imshow(input_data[:, :, mid_slice], cmap='gray')
            axes[0, 0].set_title('Input Image')
            axes[0, 0].axis('off')
            
            # Ground truth
            axes[0, 1].imshow(ground_truth[:, :, mid_slice], cmap='gray')
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')
            
            # Prediction
            axes[0, 2].imshow(prediction[:, :, mid_slice], cmap='gray')
            axes[0, 2].set_title('Prediction')
            axes[0, 2].axis('off')
            
            # Overlay comparison
            overlay_gt = np.zeros((*ground_truth.shape[:2], 3))
            overlay_gt[:, :, 0] = ground_truth[:, :, mid_slice]  # Red for GT
            overlay_gt[:, :, 1] = prediction[:, :, mid_slice]    # Green for Pred
            
            axes[1, 0].imshow(overlay_gt)
            axes[1, 0].set_title('Overlay (Red: GT, Green: Pred)')
            axes[1, 0].axis('off')
            
            # Difference
            diff = np.abs(ground_truth[:, :, mid_slice] - prediction[:, :, mid_slice])
            axes[1, 1].imshow(diff, cmap='hot')
            axes[1, 1].set_title('Difference')
            axes[1, 1].axis('off')
            
            # Histogram comparison
            axes[1, 2].hist(ground_truth.flatten(), bins=50, alpha=0.5, label='Ground Truth', color='blue')
            axes[1, 2].hist(prediction.flatten(), bins=50, alpha=0.5, label='Prediction', color='red')
            axes[1, 2].set_title('Value Distribution')
            axes[1, 2].legend()
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved to: {output_path}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of the nnU-Net model."""
        print("=" * 60)
        print("nnU-Net Model Comprehensive Test")
        print("=" * 60)
        
        results = {
            'model_config': {},
            'file_status': {},
            'data_analysis': {},
            'model_loading': False,
            'inference_tests': [],
            'overall_success': False
        }
        
        # 1. Analyze model configuration
        print("\n1. Analyzing Model Configuration...")
        results['model_config'] = self.analyze_model_configuration()
        
        # 2. Check model files
        print("\n2. Checking Model Files...")
        results['file_status'] = self.check_model_files()
        
        # 3. Analyze test data
        print("\n3. Analyzing Test Data...")
        results['data_analysis'] = self.analyze_test_data()
        
        # 4. Test model loading
        print("\n4. Testing Model Loading...")
        results['model_loading'] = self.test_model_loading()
        
        # 5. Test inference on available test data
        if results['model_loading'] and results['data_analysis']['images_found']:
            print("\n5. Testing Inference...")
            
            images_dir = self.test_data_path / "imagesTr"
            labels_dir = self.test_data_path / "labelsTr"
            
            for img_name in results['data_analysis']['images_found']:
                img_path = images_dir / img_name
                
                # Find corresponding label
                base_name = img_name.replace('_0000.nii.gz', '.nii.gz')
                label_path = labels_dir / base_name
                
                if label_path.exists():
                    print(f"\nTesting {img_name}...")
                    
                    # Run inference
                    prediction = self.test_inference_simple(str(img_path))
                    
                    if prediction is not None:
                        # Validate against ground truth
                        metrics = self.validate_output_quality(prediction, str(label_path))
                        
                        test_result = {
                            'image': img_name,
                            'label': base_name,
                            'prediction_shape': prediction.shape,
                            'metrics': metrics,
                            'success': True
                        }
                        
                        # Create visualization
                        try:
                            input_img = nib.load(img_path).get_fdata()
                            gt_img = nib.load(label_path).get_fdata()
                            self.create_visualization(
                                input_img, prediction, gt_img,
                                f"test_result_{img_name.replace('.nii.gz', '.png')}"
                            )
                        except Exception as e:
                            print(f"Visualization failed: {e}")
                        
                        results['inference_tests'].append(test_result)
                    else:
                        results['inference_tests'].append({
                            'image': img_name,
                            'success': False,
                            'error': 'Inference failed'
                        })
        
        # 6. Overall assessment
        results['overall_success'] = (
            results['model_loading'] and 
            len(results['inference_tests']) > 0 and
            any(test.get('success', False) for test in results['inference_tests'])
        )
        
        return results
    
    def generate_integration_report(self, test_results: Dict[str, Any]) -> str:
        """Generate a report with integration recommendations."""
        report = []
        report.append("# nnU-Net Model Integration Report")
        report.append("=" * 50)
        report.append("")
        
        # Model information
        config = test_results['model_config']
        report.append("## Model Information")
        report.append(f"- **Dataset**: {config['dataset_name']}")
        report.append(f"- **Architecture**: 3D U-Net (nnU-Net)")
        report.append(f"- **Patch Size**: {config['patch_size']}")
        report.append(f"- **Expected Input Size**: {config['median_image_size']}")
        report.append(f"- **Labels**: {config['labels']}")
        report.append(f"- **Training Samples**: {config['num_training_samples']}")
        report.append("")
        
        # Test results summary
        report.append("## Test Results Summary")
        report.append(f"- **Model Loading**: {'✓ Success' if test_results['model_loading'] else '✗ Failed'}")
        report.append(f"- **Inference Tests**: {len(test_results['inference_tests'])}")
        
        successful_tests = [t for t in test_results['inference_tests'] if t.get('success', False)]
        report.append(f"- **Successful Tests**: {len(successful_tests)}")
        
        if successful_tests:
            avg_dice = np.mean([t['metrics'].get('dice', 0) for t in successful_tests])
            avg_iou = np.mean([t['metrics'].get('iou', 0) for t in successful_tests])
            report.append(f"- **Average Dice Score**: {avg_dice:.3f}")
            report.append(f"- **Average IoU**: {avg_iou:.3f}")
        
        report.append("")
        
        # Integration recommendations
        report.append("## Integration Recommendations")
        
        if test_results['overall_success']:
            report.append("### ✅ Model Ready for Integration")
            report.append("")
            report.append("The model has been successfully tested and is ready for integration into CryoMamba.")
            report.append("")
            report.append("### Key Integration Points:")
            report.append("1. **Model Loading**: Use nnU-Net's `nnUNetPredictor` class")
            report.append("2. **Input Preprocessing**: Apply CT normalization")
            report.append("3. **Output Postprocessing**: Convert to binary segmentation")
            report.append("4. **Memory Management**: Consider patch-based processing for large volumes")
            report.append("")
            report.append("### Performance Characteristics:")
            report.append(f"- **Device**: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            report.append(f"- **Patch Size**: {config['patch_size']}")
            report.append(f"- **Batch Size**: {config['batch_size']}")
            report.append("")
            report.append("### Integration Steps:")
            report.append("1. Add nnU-Net dependency to requirements.txt")
            report.append("2. Create model wrapper class in CryoMamba")
            report.append("3. Implement preprocessing pipeline")
            report.append("4. Add inference endpoint to FastAPI server")
            report.append("5. Update Napari widget for model selection")
        else:
            report.append("### ❌ Model Not Ready")
            report.append("")
            report.append("The model failed testing and requires attention before integration.")
            report.append("")
            report.append("### Issues Found:")
            if not test_results['model_loading']:
                report.append("- Model loading failed")
            if not test_results['inference_tests']:
                report.append("- No successful inference tests")
            report.append("")
            report.append("### Recommended Actions:")
            report.append("1. Verify model files are complete")
            report.append("2. Check nnU-Net installation")
            report.append("3. Validate test data format")
            report.append("4. Review error messages for specific issues")
        
        report.append("")
        report.append("## Next Steps")
        report.append("1. Review this report")
        report.append("2. Address any issues identified")
        report.append("3. Proceed with integration if model is ready")
        report.append("4. Test integration in CryoMamba environment")
        
        return "\n".join(report)

def main():
    """Main function to run the comprehensive test."""
    print("Starting nnU-Net Model Test...")
    
    # Initialize tester
    tester = NNUModelTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Generate integration report
    report = tester.generate_integration_report(results)
    
    # Save report
    with open('/Users/elwinli/CryoMamba/nnunet_integration_report.md', 'w') as f:
        f.write(report)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"Overall Success: {'✓ YES' if results['overall_success'] else '✗ NO'}")
    print(f"Report saved to: nnunet_integration_report.md")
    
    return results

if __name__ == "__main__":
    results = main()
