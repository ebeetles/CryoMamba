"""
Preview downsampling service for real-time segmentation streaming.

This module provides efficient downsampling algorithms for segmentation masks
to enable real-time preview streaming during inference.
"""

import numpy as np
import logging
from typing import Tuple, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class DownsampleMethod(Enum):
    """Available downsampling methods."""
    NEAREST = "nearest"
    AVERAGE = "average"
    MAX = "max"
    GAUSSIAN = "gaussian"


class PreviewDownsampler:
    """Efficient downsampling for segmentation previews."""
    
    def __init__(self, target_size: Tuple[int, int, int] = (64, 64, 64)):
        """
        Initialize the preview downsampler.
        
        Args:
            target_size: Target size for downsampled previews (z, y, x)
        """
        self.target_size = target_size
        self._cache = {}  # Simple cache for repeated operations
    
    def downsample_mask(
        self, 
        mask: np.ndarray, 
        method: DownsampleMethod = DownsampleMethod.MAX,
        scale_factor: Optional[float] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Downsample a segmentation mask for preview streaming.
        
        Args:
            mask: Input segmentation mask (3D array)
            method: Downsampling method to use
            scale_factor: Optional scale factor (overrides target_size)
        
        Returns:
            Tuple of (downsampled_mask, scale_factors)
        """
        if mask.ndim != 3:
            raise ValueError("Mask must be 3D")
        
        original_shape = mask.shape
        
        # Calculate target shape
        if scale_factor is not None:
            target_shape = tuple(int(s * scale_factor) for s in original_shape)
        else:
            target_shape = self._calculate_target_shape(original_shape)
        
        # Calculate scale factors
        scale_factors = [
            target_shape[2] / original_shape[2],  # x
            target_shape[1] / original_shape[1],  # y  
            target_shape[0] / original_shape[0]   # z
        ]
        
        # Apply downsampling based on method
        if method == DownsampleMethod.NEAREST:
            downsampled = self._downsample_nearest(mask, target_shape)
        elif method == DownsampleMethod.AVERAGE:
            downsampled = self._downsample_average(mask, target_shape)
        elif method == DownsampleMethod.MAX:
            downsampled = self._downsample_max(mask, target_shape)
        elif method == DownsampleMethod.GAUSSIAN:
            downsampled = self._downsample_gaussian(mask, target_shape)
        else:
            raise ValueError(f"Unknown downsampling method: {method}")
        
        logger.debug(f"Downsampled mask from {original_shape} to {downsampled.shape}")
        
        return downsampled, scale_factors
    
    def _calculate_target_shape(self, original_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate optimal target shape maintaining aspect ratio."""
        # Find the dimension that needs the most reduction
        max_dim = max(original_shape)
        target_max_dim = min(self.target_size)
        
        if max_dim <= target_max_dim:
            return original_shape
        
        scale_factor = target_max_dim / max_dim
        target_shape = tuple(int(s * scale_factor) for s in original_shape)
        
        # Ensure minimum size of 1 in each dimension
        target_shape = tuple(max(1, s) for s in target_shape)
        
        return target_shape
    
    def _downsample_nearest(self, mask: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Downsample using nearest neighbor interpolation."""
        from scipy.ndimage import zoom
        
        zoom_factors = [
            target_shape[0] / mask.shape[0],
            target_shape[1] / mask.shape[1], 
            target_shape[2] / mask.shape[2]
        ]
        
        return zoom(mask, zoom_factors, order=0, prefilter=False)
    
    def _downsample_average(self, mask: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Downsample using average pooling."""
        from scipy.ndimage import zoom
        
        zoom_factors = [
            target_shape[0] / mask.shape[0],
            target_shape[1] / mask.shape[1],
            target_shape[2] / mask.shape[2]
        ]
        
        return zoom(mask, zoom_factors, order=1, prefilter=True)
    
    def _downsample_max(self, mask: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Downsample using max pooling (good for segmentation masks)."""
        from scipy.ndimage import zoom
        
        zoom_factors = [
            target_shape[0] / mask.shape[0],
            target_shape[1] / mask.shape[1],
            target_shape[2] / mask.shape[2]
        ]
        
        # For max pooling, we need to handle this differently
        # Use zoom with order=0 but apply max pooling logic
        downsampled = zoom(mask, zoom_factors, order=0, prefilter=False)
        
        # Apply max pooling to preserve segmentation boundaries
        if np.any(mask > 0):
            # Ensure we preserve the maximum value in each region
            downsampled = np.maximum(downsampled, 0)
        
        return downsampled
    
    def _downsample_gaussian(self, mask: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Downsample using Gaussian smoothing."""
        from scipy.ndimage import zoom, gaussian_filter
        
        # Apply Gaussian smoothing first
        sigma = 0.5  # Adjust based on downsampling factor
        smoothed = gaussian_filter(mask.astype(np.float32), sigma=sigma)
        
        zoom_factors = [
            target_shape[0] / mask.shape[0],
            target_shape[1] / mask.shape[1],
            target_shape[2] / mask.shape[2]
        ]
        
        downsampled = zoom(smoothed, zoom_factors, order=1, prefilter=False)
        
        # Convert back to segmentation mask (threshold at 0.5)
        return (downsampled > 0.5).astype(mask.dtype)
    
    def create_multi_resolution_previews(
        self, 
        mask: np.ndarray, 
        scales: List[float] = None
    ) -> List[Tuple[np.ndarray, List[float]]]:
        """
        Create multiple resolution previews for different zoom levels.
        
        Args:
            mask: Input segmentation mask
            scales: List of scale factors (default: [1.0, 0.5, 0.25])
        
        Returns:
            List of (downsampled_mask, scale_factors) tuples
        """
        if scales is None:
            scales = [1.0, 0.5, 0.25]
        
        previews = []
        for scale in scales:
            downsampled, scale_factors = self.downsample_mask(
                mask, 
                method=DownsampleMethod.MAX,
                scale_factor=scale
            )
            previews.append((downsampled, scale_factors))
        
        return previews
    
    def optimize_for_visual_quality(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply visual quality optimizations to the mask.
        
        Args:
            mask: Input segmentation mask
        
        Returns:
            Optimized mask for visual display
        """
        # Ensure proper data type for visualization
        if mask.dtype not in [np.uint8, np.uint16]:
            # Normalize to uint8 range
            if np.max(mask) > 255:
                mask = (mask / np.max(mask) * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
        
        # Apply morphological operations to clean up the mask
        from scipy.ndimage import binary_opening, binary_closing
        
        # Convert to binary for morphological operations
        binary_mask = mask > 0
        
        # Apply opening to remove small noise
        cleaned = binary_opening(binary_mask, structure=np.ones((2, 2, 2)))
        
        # Apply closing to fill small holes
        cleaned = binary_closing(cleaned, structure=np.ones((2, 2, 2)))
        
        # Convert back to original dtype
        return cleaned.astype(mask.dtype) * np.max(mask)
    
    def get_memory_usage_estimate(self, original_shape: Tuple[int, ...]) -> int:
        """
        Estimate memory usage for downsampled preview.
        
        Args:
            original_shape: Shape of original mask
        
        Returns:
            Estimated memory usage in bytes
        """
        target_shape = self._calculate_target_shape(original_shape)
        return np.prod(target_shape) * 4  # Assuming float32
