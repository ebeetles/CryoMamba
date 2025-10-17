"""
Export service for CryoMamba - handles mask and visualization exports.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import mrcfile
from datetime import datetime
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ExportService:
    """Service for exporting masks and visualizations in various formats."""
    
    def __init__(self):
        self.supported_formats = {
            'mrc': self._export_mrc,
            'nifti': self._export_nifti,
            'nrrd': self._export_nrrd,
            'png': self._export_png
        }
        
        # Export queue management
        self.export_queue = queue.Queue()
        self.export_thread = None
        self.export_running = False
        self.max_concurrent_exports = 2
        
        # Performance monitoring
        self.export_stats = {
            'total_exports': 0,
            'successful_exports': 0,
            'failed_exports': 0,
            'average_export_time': 0.0,
            'last_export_time': None
        }
        
        # Persistent statistics storage
        self.stats_file = Path.home() / '.cryomamba' / 'export_stats.json'
        self.stats_file.parent.mkdir(exist_ok=True)
        self._load_persistent_stats()
        
        # Start export worker thread
        self._start_export_worker()
    
    def export_mask(self, mask_data: np.ndarray, output_path: str, 
                   format_type: str = 'mrc', metadata: Optional[Dict[str, Any]] = None,
                   async_export: bool = False) -> bool:
        """
        Export mask data to specified format.
        
        Args:
            mask_data: 3D numpy array containing mask data
            output_path: Path where to save the exported file
            format_type: Export format ('mrc', 'nifti', 'nrrd')
            metadata: Optional metadata including voxel size, scale info
            async_export: If True, queue export for background processing
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            if format_type not in self.supported_formats:
                logger.error(f"Unsupported export format: {format_type}")
                return False
            
            # Validate mask data
            if not self._validate_mask_data(mask_data):
                return False
            
            # Create output directory if it doesn't exist
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if async_export:
                # Queue for background processing
                export_task = {
                    'mask_data': mask_data,
                    'output_path': str(output_path),
                    'format_type': format_type,
                    'metadata': metadata,
                    'timestamp': time.time()
                }
                self.export_queue.put(export_task)
                logger.info(f"Queued export task for {output_path}")
                return True
            else:
                # Immediate export
                start_time = time.time()
                success = self.supported_formats[format_type](mask_data, str(output_path), metadata)
                export_time = time.time() - start_time
                
                # Update statistics
                self._update_export_stats(success, export_time)
                
                if success:
                    logger.info(f"Successfully exported mask to {output_path} in {export_time:.2f}s")
                    return True
                else:
                    logger.error(f"Failed to export mask to {output_path}")
                    return False
                
        except Exception as e:
            logger.error(f"Error exporting mask: {str(e)}")
            return False
    
    def _validate_mask_data(self, mask_data: np.ndarray) -> bool:
        """Validate mask data before export."""
        try:
            if not isinstance(mask_data, np.ndarray):
                logger.error("Mask data must be a numpy array")
                return False
            
            if mask_data.ndim != 3:
                logger.error(f"Mask data must be 3D, got {mask_data.ndim}D")
                return False
            
            if mask_data.size == 0:
                logger.error("Mask data is empty")
                return False
            
            # Check for reasonable data types
            if mask_data.dtype not in [np.uint8, np.uint16, np.int16, np.int32, np.float32]:
                logger.warning(f"Unusual data type for mask: {mask_data.dtype}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating mask data: {str(e)}")
            return False
    
    def _export_mrc(self, mask_data: np.ndarray, output_path: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Export mask as MRC file with proper voxel alignment."""
        try:
            # Ensure data is in appropriate format for MRC
            if mask_data.dtype not in [np.int8, np.int16, np.int32, np.float32]:
                mask_data = mask_data.astype(np.float32)
            
            with mrcfile.new(output_path, overwrite=True) as mrc:
                mrc.set_data(mask_data)
                
                # Set voxel size if provided in metadata
                if metadata and 'voxel_size' in metadata:
                    voxel_size = metadata['voxel_size']
                    if len(voxel_size) == 3:
                        mrc.voxel_size = voxel_size
                
                # Set additional metadata
                if metadata:
                    if 'origin' in metadata:
                        mrc.header.origin = metadata['origin']
                    if 'min' in metadata:
                        mrc.header.dmin = metadata['min']
                    if 'max' in metadata:
                        mrc.header.dmax = metadata['max']
                    if 'mean' in metadata:
                        mrc.header.dmean = metadata['mean']
                
                # Add export metadata to header
                mrc.header.extra = f"Exported by CryoMamba on {datetime.now().isoformat()}".encode()
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting MRC file: {str(e)}")
            return False
    
    def _export_nifti(self, mask_data: np.ndarray, output_path: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Export mask as NIfTI file using nibabel."""
        try:
            import nibabel as nib
            
            # Create NIfTI image
            nii_img = nib.Nifti1Image(mask_data, affine=np.eye(4))
            
            # Set voxel size if provided
            if metadata and 'voxel_size' in metadata:
                voxel_size = metadata['voxel_size']
                if len(voxel_size) == 3:
                    # Create affine matrix with voxel size
                    affine = np.eye(4)
                    affine[0, 0] = voxel_size[0]
                    affine[1, 1] = voxel_size[1]
                    affine[2, 2] = voxel_size[2]
                    nii_img = nib.Nifti1Image(mask_data, affine=affine)
            
            # Save NIfTI file
            nib.save(nii_img, output_path)
            
            return True
            
        except ImportError:
            logger.error("nibabel not available for NIfTI export")
            return False
        except Exception as e:
            logger.error(f"Error exporting NIfTI file: {str(e)}")
            return False
    
    def _export_nrrd(self, mask_data: np.ndarray, output_path: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Export mask as NRRD file using pynrrd."""
        try:
            import nrrd
            
            # Prepare NRRD header
            header = {
                'type': str(mask_data.dtype),
                'dimension': 3,
                'sizes': list(mask_data.shape)
            }
            
            # Add voxel size if provided
            if metadata and 'voxel_size' in metadata:
                voxel_size = metadata['voxel_size']
                if len(voxel_size) == 3:
                    header['space directions'] = [
                        [voxel_size[0], 0, 0],
                        [0, voxel_size[1], 0],
                        [0, 0, voxel_size[2]]
                    ]
            
            # Write NRRD file
            nrrd.write(output_path, mask_data, header)
            
            return True
            
        except ImportError:
            logger.error("pynrrd not available for NRRD export")
            return False
        except Exception as e:
            logger.error(f"Error exporting NRRD file: {str(e)}")
            return False
    
    def _export_png(self, mask_data: np.ndarray, output_path: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Export mask as PNG screenshot (2D slice)."""
        try:
            from PIL import Image
            
            # For PNG export, we'll take the middle slice
            if mask_data.ndim == 3:
                middle_slice = mask_data[mask_data.shape[0] // 2, :, :]
            else:
                middle_slice = mask_data
            
            # Normalize to 0-255 range
            if middle_slice.max() > 255 or middle_slice.min() < 0:
                middle_slice = ((middle_slice - middle_slice.min()) / 
                              (middle_slice.max() - middle_slice.min()) * 255).astype(np.uint8)
            
            # Convert to PIL Image and save
            img = Image.fromarray(middle_slice)
            img.save(output_path)
            
            return True
            
        except ImportError:
            logger.error("PIL not available for PNG export")
            return False
        except Exception as e:
            logger.error(f"Error exporting PNG file: {str(e)}")
            return False
    
    def get_export_info(self, mask_data: np.ndarray) -> Dict[str, Any]:
        """Get information about the mask for export metadata."""
        return {
            'shape': mask_data.shape,
            'dtype': str(mask_data.dtype),
            'min': float(np.min(mask_data)),
            'max': float(np.max(mask_data)),
            'mean': float(np.mean(mask_data)),
            'non_zero_voxels': int(np.count_nonzero(mask_data)),
            'total_voxels': int(mask_data.size),
            'export_timestamp': datetime.now().isoformat()
        }
    
    def _start_export_worker(self):
        """Start the background export worker thread."""
        if self.export_thread is None or not self.export_thread.is_alive():
            self.export_running = True
            self.export_thread = threading.Thread(target=self._export_worker_loop, daemon=True)
            self.export_thread.start()
            logger.info("Export worker thread started")
    
    def _export_worker_loop(self):
        """Background worker loop for processing export queue."""
        with ThreadPoolExecutor(max_workers=self.max_concurrent_exports) as executor:
            while self.export_running:
                try:
                    # Get next export task (with timeout)
                    export_task = self.export_queue.get(timeout=1.0)
                    
                    # Process export task
                    start_time = time.time()
                    success = self.supported_formats[export_task['format_type']](
                        export_task['mask_data'],
                        export_task['output_path'],
                        export_task['metadata']
                    )
                    export_time = time.time() - start_time
                    
                    # Update statistics
                    self._update_export_stats(success, export_time)
                    
                    if success:
                        logger.info(f"Background export completed: {export_task['output_path']} in {export_time:.2f}s")
                    else:
                        logger.error(f"Background export failed: {export_task['output_path']}")
                    
                    self.export_queue.task_done()
                    
                except queue.Empty:
                    # No tasks in queue, continue
                    continue
                except Exception as e:
                    logger.error(f"Error in export worker: {str(e)}")
    
    def _update_export_stats(self, success: bool, export_time: float):
        """Update export performance statistics."""
        self.export_stats['total_exports'] += 1
        if success:
            self.export_stats['successful_exports'] += 1
        else:
            self.export_stats['failed_exports'] += 1
        
        # Update average export time
        total_exports = self.export_stats['total_exports']
        current_avg = self.export_stats['average_export_time']
        self.export_stats['average_export_time'] = (
            (current_avg * (total_exports - 1) + export_time) / total_exports
        )
        
        self.export_stats['last_export_time'] = export_time
        
        # Save to persistent storage
        self._save_persistent_stats()
    
    def get_export_stats(self) -> Dict[str, Any]:
        """Get current export performance statistics."""
        return self.export_stats.copy()
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current export queue status."""
        return {
            'queue_size': self.export_queue.qsize(),
            'worker_running': self.export_running and self.export_thread and self.export_thread.is_alive(),
            'max_concurrent': self.max_concurrent_exports
        }
    
    def stop_export_worker(self):
        """Stop the background export worker."""
        self.export_running = False
        if self.export_thread and self.export_thread.is_alive():
            self.export_thread.join(timeout=5.0)
        logger.info("Export worker thread stopped")
    
    def _load_persistent_stats(self):
        """Load persistent statistics from file."""
        try:
            if self.stats_file.exists():
                import json
                with open(self.stats_file, 'r') as f:
                    saved_stats = json.load(f)
                    # Merge with current stats
                    self.export_stats.update(saved_stats)
                logger.info(f"Loaded persistent export statistics from {self.stats_file}")
        except Exception as e:
            logger.warning(f"Could not load persistent stats: {e}")
    
    def _save_persistent_stats(self):
        """Save current statistics to file."""
        try:
            import json
            with open(self.stats_file, 'w') as f:
                json.dump(self.export_stats, f, indent=2)
            logger.debug(f"Saved export statistics to {self.stats_file}")
        except Exception as e:
            logger.warning(f"Could not save persistent stats: {e}")

# Global export service instance
export_service = ExportService()
