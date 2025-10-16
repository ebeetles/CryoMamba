"""
Preview streaming service for real-time segmentation updates.

This module handles periodic preview generation and WebSocket streaming
during nnU-Net inference.
"""

import asyncio
import logging
import numpy as np
import time
from typing import Optional, Callable, Any, Dict, List
from datetime import datetime
import json

from app.services.preview_downsampler import PreviewDownsampler, DownsampleMethod
from app.models.preview import PreviewMessage
from app.routes.websocket import broadcast_job_update
from app.services.preview_performance import (
    performance_monitor, 
    preview_cache, 
    adaptive_controller,
    PerformanceMetrics
)

logger = logging.getLogger(__name__)


class PreviewStreamer:
    """Handles real-time preview streaming during inference."""
    
    def __init__(
        self,
        job_id: str,
        frequency: float = 1.0,  # Updates per second
        target_size: tuple = (64, 64, 64),
        method: DownsampleMethod = DownsampleMethod.MAX
    ):
        """
        Initialize the preview streamer.
        
        Args:
            job_id: Job ID for WebSocket broadcasting
            frequency: Preview update frequency in Hz
            target_size: Target size for downsampled previews
            method: Downsampling method to use
        """
        self.job_id = job_id
        self.frequency = frequency
        self.downsampler = PreviewDownsampler(target_size)
        self.method = method
        
        self._streaming_task: Optional[asyncio.Task] = None
        self._current_mask: Optional[np.ndarray] = None
        self._stop_event = asyncio.Event()
        self._preview_count = 0
        self._last_preview_hash: Optional[str] = None
        
    async def start_streaming(self, initial_mask: Optional[np.ndarray] = None):
        """
        Start preview streaming.
        
        Args:
            initial_mask: Optional initial mask to start with
        """
        if self._streaming_task is not None:
            logger.warning(f"Preview streaming already active for job {self.job_id}")
            return
        
        if initial_mask is not None:
            self._current_mask = initial_mask
        
        self._stop_event.clear()
        self._streaming_task = asyncio.create_task(self._streaming_loop())
        
        logger.info(f"Started preview streaming for job {self.job_id} at {self.frequency}Hz")
    
    async def stop_streaming(self):
        """Stop preview streaming."""
        if self._streaming_task is None:
            return
        
        self._stop_event.set()
        self._streaming_task.cancel()
        
        try:
            await self._streaming_task
        except asyncio.CancelledError:
            pass
        
        self._streaming_task = None
        logger.info(f"Stopped preview streaming for job {self.job_id}")
    
    def update_mask(self, new_mask: np.ndarray):
        """
        Update the current mask for streaming.
        
        Args:
            new_mask: New segmentation mask
        """
        self._current_mask = new_mask.copy()
        logger.debug(f"Updated mask for job {self.job_id}, shape: {new_mask.shape}")
    
    async def _streaming_loop(self):
        """Main streaming loop."""
        try:
            while not self._stop_event.is_set():
                if self._current_mask is not None:
                    await self._send_preview()
                
                # Use adaptive frequency
                current_frequency = adaptive_controller.get_current_frequency()
                await asyncio.sleep(1.0 / current_frequency)
                
        except asyncio.CancelledError:
            logger.info(f"Preview streaming cancelled for job {self.job_id}")
        except Exception as e:
            logger.error(f"Error in preview streaming for job {self.job_id}: {e}")
            await broadcast_job_update(self.job_id, "error", {
                "message": f"Preview streaming error: {str(e)}"
            })
    
    async def _send_preview(self):
        """Generate and send a preview update."""
        start_time = time.time()
        
        try:
            if self._current_mask is None:
                return
            
            # Check if mask has changed significantly
            current_hash = hash(self._current_mask.tobytes())
            if current_hash == self._last_preview_hash:
                return  # Skip if no significant change
            
            self._last_preview_hash = current_hash
            
            # Check cache first
            cache_key = f"{self.job_id}_{current_hash}"
            cached_preview = preview_cache.get(cache_key)
            
            if cached_preview is not None:
                logger.debug(f"Using cached preview for job {self.job_id}")
                optimized_mask = cached_preview
                scale_factors = [1.0, 1.0, 1.0]  # Default scale for cached data
            else:
                # Generate downsampled preview
                timing_key = performance_monitor.start_timing(self.job_id, "downsampling")
                downsampled_mask, scale_factors = self.downsampler.downsample_mask(
                    self._current_mask, 
                    method=self.method
                )
                downsampling_time = performance_monitor.stop_timing(timing_key)
                
                # Optimize for visual quality
                timing_key = performance_monitor.start_timing(self.job_id, "optimization")
                optimized_mask = self.downsampler.optimize_for_visual_quality(downsampled_mask)
                optimization_time = performance_monitor.stop_timing(timing_key)
                
                # Cache the result
                preview_cache.put(cache_key, optimized_mask)
            
            # Create preview message
            timing_key = performance_monitor.start_timing(self.job_id, "serialization")
            preview_message = PreviewMessage.from_numpy_array(
                optimized_mask, 
                scale=scale_factors
            )
            serialization_time = performance_monitor.stop_timing(timing_key)
            
            # Add metadata
            preview_data = preview_message.model_dump()
            preview_data.update({
                "preview_id": self._preview_count,
                "timestamp": datetime.now().isoformat(),
                "original_shape": list(self._current_mask.shape),
                "downsampled_shape": list(optimized_mask.shape),
                "method": self.method.value
            })
            
            # Broadcast via WebSocket
            timing_key = performance_monitor.start_timing(self.job_id, "websocket_send")
            await broadcast_job_update(self.job_id, "preview", preview_data)
            websocket_send_time = performance_monitor.stop_timing(timing_key)
            
            # Record performance metrics
            total_latency = (time.time() - start_time) * 1000
            metrics = PerformanceMetrics(
                job_id=self.job_id,
                timestamp=datetime.now(),
                preview_generation_time_ms=total_latency,
                downsampling_time_ms=getattr(locals(), 'downsampling_time', 0),
                serialization_time_ms=serialization_time,
                websocket_send_time_ms=websocket_send_time,
                preview_size_bytes=optimized_mask.nbytes,
                compression_ratio=1.0,  # No compression for now
                total_latency_ms=total_latency
            )
            performance_monitor.record_metrics(metrics)
            
            # Update adaptive controller
            adaptive_controller.update_performance(total_latency)
            
            self._preview_count += 1
            logger.debug(f"Sent preview {self._preview_count} for job {self.job_id} (latency: {total_latency:.1f}ms)")
            
        except Exception as e:
            logger.error(f"Error sending preview for job {self.job_id}: {e}")
            await broadcast_job_update(self.job_id, "error", {
                "message": f"Preview generation error: {str(e)}"
            })
    
    async def send_final_preview(self, final_mask: np.ndarray):
        """
        Send a final high-quality preview when inference completes.
        
        Args:
            final_mask: Final segmentation mask
        """
        try:
            # Use higher quality downsampling for final preview
            downsampled_mask, scale_factors = self.downsampler.downsample_mask(
                final_mask,
                method=DownsampleMethod.GAUSSIAN,
                scale_factor=0.5  # Higher quality than streaming
            )
            
            # Create final preview message
            preview_message = PreviewMessage.from_numpy_array(
                downsampled_mask,
                scale=scale_factors
            )
            
            preview_data = preview_message.model_dump()
            preview_data.update({
                "preview_id": "final",
                "timestamp": datetime.now().isoformat(),
                "original_shape": list(final_mask.shape),
                "downsampled_shape": list(downsampled_mask.shape),
                "method": "final",
                "is_final": True
            })
            
            await broadcast_job_update(self.job_id, "preview", preview_data)
            logger.info(f"Sent final preview for job {self.job_id}")
            
        except Exception as e:
            logger.error(f"Error sending final preview for job {self.job_id}: {e}")


class PreviewManager:
    """Manages multiple preview streamers for different jobs."""
    
    def __init__(self):
        self._streamers: Dict[str, PreviewStreamer] = {}
    
    async def create_streamer(
        self,
        job_id: str,
        frequency: float = 1.0,
        target_size: tuple = (64, 64, 64),
        method: DownsampleMethod = DownsampleMethod.MAX
    ) -> PreviewStreamer:
        """
        Create a new preview streamer for a job.
        
        Args:
            job_id: Job ID
            frequency: Update frequency in Hz
            target_size: Target preview size
            method: Downsampling method
        
        Returns:
            PreviewStreamer instance
        """
        if job_id in self._streamers:
            await self._streamers[job_id].stop_streaming()
        
        streamer = PreviewStreamer(
            job_id=job_id,
            frequency=frequency,
            target_size=target_size,
            method=method
        )
        
        self._streamers[job_id] = streamer
        return streamer
    
    async def start_streaming(self, job_id: str, initial_mask: Optional[np.ndarray] = None):
        """Start streaming for a job."""
        if job_id in self._streamers:
            await self._streamers[job_id].start_streaming(initial_mask)
    
    async def stop_streaming(self, job_id: str):
        """Stop streaming for a job."""
        if job_id in self._streamers:
            await self._streamers[job_id].stop_streaming()
            del self._streamers[job_id]
    
    def update_mask(self, job_id: str, mask: np.ndarray):
        """Update mask for a job."""
        if job_id in self._streamers:
            self._streamers[job_id].update_mask(mask)
    
    async def send_final_preview(self, job_id: str, final_mask: np.ndarray):
        """Send final preview for a job."""
        if job_id in self._streamers:
            await self._streamers[job_id].send_final_preview(final_mask)
    
    async def cleanup(self):
        """Cleanup all streamers."""
        for streamer in self._streamers.values():
            await streamer.stop_streaming()
        self._streamers.clear()


# Global preview manager instance
preview_manager = PreviewManager()
