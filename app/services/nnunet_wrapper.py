import os
import tempfile
import time
import logging
from pathlib import Path
from typing import Optional, Callable, Any, Dict

import numpy as np

try:
    # Optional dependency: wrapper must work even if nnU-Net is not installed
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # type: ignore
    NNUNET_AVAILABLE = True
except Exception:
    nnUNetPredictor = None  # type: ignore
    NNUNET_AVAILABLE = False

logger = logging.getLogger(__name__)


class NnUNetNotAvailableError(RuntimeError):
    """Raised when nnU-Net is required but not available in the environment."""


class NnUNetWrapper:
    """Lightweight wrapper around nnU-Net v2 predictor with sliding-window defaults.

    This class hides import details and provides a stable interface for the
    orchestrator/inference pipeline. It supports lazy initialization so the
    host application can boot without nnU-Net installed.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        perform_everything_on_device: bool = True,
        checkpoint_name: str = "checkpoint_best.pth",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        self.perform_everything_on_device = perform_everything_on_device
        self.checkpoint_name = checkpoint_name

        self._predictor = None  # Lazy init

    def is_available(self) -> bool:
        return NNUNET_AVAILABLE

    def ensure_initialized(self) -> None:
        if self._predictor is not None:
            return
        if not NNUNET_AVAILABLE:
            raise NnUNetNotAvailableError(
                "nnU-Net v2 is not installed. Install nnunetv2 to enable inference."
            )
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        logger.info("Initializing nnU-Net predictor from %s", self.model_dir)
        
        # Convert string device to torch.device
        import torch
        if isinstance(self.device, str):
            torch_device = torch.device(self.device)
        else:
            torch_device = self.device
            
        predictor = nnUNetPredictor(
            tile_step_size=self.tile_step_size,
            use_gaussian=self.use_gaussian,
            use_mirroring=self.use_mirroring,
            perform_everything_on_device=self.perform_everything_on_device,
            device=torch_device,
            verbose=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(self.model_dir), use_folds="all", checkpoint_name=self.checkpoint_name
        )
        self._predictor = predictor

    def predict_numpy(
        self,
        volume: np.ndarray,
        progress_callback: Optional[Callable[[float], Any]] = None,
        preview_callback: Optional[Callable[[np.ndarray], Any]] = None,
        tmp_dir: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict segmentation mask from numpy volume using direct network inference.
        Uses patch-based processing to avoid multiprocessing issues.
        Based on the working experiment implementation.
        """
        self.ensure_initialized()
        
        # Basic input validation
        if volume.ndim not in (3, 4):
            raise ValueError("Expected a 3D or 4D volume (C,Z,Y,X or Z,Y,X)")

        # Normalize to float32 if needed
        if volume.dtype != np.float32:
            volume = volume.astype(np.float32, copy=False)
            
        # Ensure we have the right shape for processing
        # The working experiment expects (Z,Y,X) format
        if volume.ndim == 4:
            # If 4D, assume it's (C,Z,Y,X) and take the first channel
            volume = volume[0]
        
        logger.info(f"Processing volume with shape: {volume.shape}")
        logger.info(f"Volume data range: [{volume.min():.6f}, {volume.max():.6f}]")
        logger.info(f"Volume data type: {volume.dtype}")
        logger.info(f"Volume has NaN: {np.isnan(volume).any()}")
        logger.info(f"Volume has Inf: {np.isinf(volume).any()}")

        logger.info(f"Starting direct network inference on volume shape: {volume.shape}")
        
        # Apply CT normalization (from working experiment)
        p1, p99 = np.percentile(volume, [1, 99])
        
        # Handle edge cases where p1 == p99 (constant volume)
        if p1 == p99:
            logger.warning(f"Volume appears to be constant (p1=p99={p1}), using simple normalization")
            volume_normalized = np.zeros_like(volume, dtype=np.float32)
            # Set to a small positive value to avoid all zeros
            volume_normalized.fill(0.5)
        else:
            volume_clipped = np.clip(volume, p1, p99)
            volume_normalized = (volume_clipped - p1) / (p99 - p1)
        
        logger.info(f"Applied CT normalization: [{p1:.3f}, {p99:.3f}] -> [0, 1]")
        
        # Use patch-based processing (from working experiment)
        patch_size = (64, 64, 64)
        overlap = 16
        
        logger.info(f"Processing in patches of size: {patch_size}")
        
        # Initialize output array
        prediction_full = np.zeros(volume.shape, dtype=np.float32)
        count_map = np.zeros(volume.shape, dtype=np.float32)
        
        # Calculate patch positions
        patch_positions = []
        for z in range(0, volume.shape[0], patch_size[0] - overlap):
            for y in range(0, volume.shape[1], patch_size[1] - overlap):
                for x in range(0, volume.shape[2], patch_size[2] - overlap):
                    z_end = min(z + patch_size[0], volume.shape[0])
                    y_end = min(y + patch_size[1], volume.shape[1])
                    x_end = min(x + patch_size[2], volume.shape[2])
                    patch_positions.append(((z, z_end), (y, y_end), (x, x_end)))
        
        logger.info(f"Total patches to process: {len(patch_positions)}")
        
        # Process patches using direct network inference
        assert self._predictor is not None
        network = self._predictor.network
        network.eval()
        
        import torch
        
        for j, ((z_start, z_end), (y_start, y_end), (x_start, x_end)) in enumerate(patch_positions):
            # Progress callback
            if progress_callback:
                progress = 0.1 + (j / len(patch_positions)) * 0.8  # 10% to 90%
                progress_callback(progress)
            
            # Log progress every 50 patches
            if (j + 1) % 50 == 0:
                logger.info(f"Processing patch {j+1}/{len(patch_positions)} ({(j+1)/len(patch_positions)*100:.1f}%)")
            
            # Extract patch
            patch_data = volume_normalized[z_start:z_end, y_start:y_end, x_start:x_end]
            
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
            
            # Preview callback for intermediate results
            if preview_callback and (j + 1) % 10 == 0:  # Every 10 patches
                # Create intermediate preview
                intermediate_prediction = prediction_full / (count_map + 1e-8)
                intermediate_binary = (intermediate_prediction > 0.5).astype(np.float32)
                preview_callback(intermediate_binary)
            
            # Clear memory
            del input_tensor, output, patch_prediction
            if (j + 1) % 20 == 0:
                import gc
                gc.collect()
        
        # Average overlapping regions
        prediction_full = prediction_full / (count_map + 1e-8)
        prediction_binary = (prediction_full > 0.5).astype(np.float32)
        
        # Final progress callback
        if progress_callback:
            progress_callback(0.95)
        
        # Final preview callback
        if preview_callback:
            preview_callback(prediction_binary)
        
        logger.info(f"Direct network inference completed successfully")
        return prediction_binary

    def warmup(self) -> Dict[str, Any]:
        """Attempt to initialize the model and return environment info."""
        available = self.is_available()
        info: Dict[str, Any] = {
            "nnunet_available": available,
            "model_dir": str(self.model_dir),
        }
        if not available:
            info["status"] = "nnU-Net not installed"
            return info
        try:
            # Check if model directory exists and has required files
            if not self.model_dir.exists():
                info["status"] = "error: Model directory not found"
                return info
                
            required_files = ["dataset.json", "plans.json"]
            for req_file in required_files:
                if not (self.model_dir / req_file).exists():
                    info["status"] = f"error: Missing required file {req_file}"
                    return info
            
            # Try to initialize without actually running inference
            self.ensure_initialized()
            assert self._predictor is not None
            info.update(
                {
                    "device": str(self._predictor.device),
                    "model_initialized": True,
                }
            )
            info["status"] = "ready"
        except Exception as e:
            info["status"] = f"error: {e}"
        return info


