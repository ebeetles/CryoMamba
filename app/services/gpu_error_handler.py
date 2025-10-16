import asyncio
import logging
from typing import Dict, Optional, List, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from app.services.gpu_monitor import get_gpu_monitor, GPUInfo
from app.services.gpu_scheduler import get_gpu_scheduler
from app.services.gpu_memory_manager import get_gpu_memory_manager

logger = logging.getLogger(__name__)


class GPUErrorType(Enum):
    """Types of GPU errors"""
    CUDA_OOM = "cuda_out_of_memory"
    CUDA_ERROR = "cuda_error"
    DEVICE_UNAVAILABLE = "device_unavailable"
    MEMORY_FRAGMENTATION = "memory_fragmentation"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class GPUError:
    """Represents a GPU error with context"""
    def __init__(self, error_type: GPUErrorType, device_id: int, 
                 message: str, job_id: Optional[str] = None,
                 timestamp: Optional[datetime] = None):
        self.error_type = error_type
        self.device_id = device_id
        self.message = message
        self.job_id = job_id
        self.timestamp = timestamp or datetime.now()
        self.retry_count = 0
        self.max_retries = 3
        
    def can_retry(self) -> bool:
        """Check if this error can be retried"""
        return self.retry_count < self.max_retries and self.error_type in {
            GPUErrorType.CUDA_OOM, GPUErrorType.MEMORY_FRAGMENTATION, GPUErrorType.TIMEOUT
        }


class GPUErrorHandler:
    """Handles GPU errors and recovery mechanisms"""
    
    def __init__(self):
        self.error_history: List[GPUError] = []
        self.error_callbacks: Dict[GPUErrorType, List[Callable]] = {}
        self._lock = threading.Lock()
        self._recovery_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
    async def start(self):
        """Start error handling"""
        logger.info("Starting GPU error handler")
        self._stop_event.clear()
        self._recovery_task = asyncio.create_task(self._recovery_loop())
        
    async def stop(self):
        """Stop error handling"""
        logger.info("Stopping GPU error handler")
        self._stop_event.set()
        if self._recovery_task:
            await self._recovery_task
        self._recovery_task = None
        
    def register_error_callback(self, error_type: GPUErrorType, callback: Callable):
        """Register a callback for specific error types"""
        if error_type not in self.error_callbacks:
            self.error_callbacks[error_type] = []
        self.error_callbacks[error_type].append(callback)
        
    def handle_error(self, error: GPUError) -> bool:
        """Handle a GPU error and attempt recovery"""
        with self._lock:
            self.error_history.append(error)
            
        logger.error("GPU error: %s on device %d - %s", 
                    error.error_type.value, error.device_id, error.message)
        
        # Call registered callbacks
        callbacks = self.error_callbacks.get(error.error_type, [])
        for callback in callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error("Error in callback for %s: %s", error.error_type.value, e)
                
        # Attempt recovery based on error type
        return self._attempt_recovery(error)
        
    def _attempt_recovery(self, error: GPUError) -> bool:
        """Attempt to recover from a GPU error"""
        if not error.can_retry():
            logger.warning("Error %s cannot be retried (attempts: %d/%d)",
                         error.error_type.value, error.retry_count, error.max_retries)
            return False
            
        error.retry_count += 1
        
        if error.error_type == GPUErrorType.CUDA_OOM:
            return self._recover_from_oom(error)
        elif error.error_type == GPUErrorType.MEMORY_FRAGMENTATION:
            return self._recover_from_fragmentation(error)
        elif error.error_type == GPUErrorType.DEVICE_UNAVAILABLE:
            return self._recover_from_device_unavailable(error)
        elif error.error_type == GPUErrorType.CUDA_ERROR:
            return self._recover_from_cuda_error(error)
        else:
            logger.warning("No recovery strategy for error type: %s", error.error_type.value)
            return False
            
    def _recover_from_oom(self, error: GPUError) -> bool:
        """Recover from CUDA out of memory error"""
        logger.info("Attempting OOM recovery for device %d", error.device_id)
        
        # Clear GPU cache
        if TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache for device %d", error.device_id)
            except Exception as e:
                logger.error("Failed to clear CUDA cache: %s", e)
                return False
                
        # Defragment memory
        memory_manager = get_gpu_memory_manager()
        if memory_manager:
            if memory_manager.defragment_memory(error.device_id):
                logger.info("Memory defragmentation successful for device %d", error.device_id)
                return True
            else:
                logger.warning("Memory defragmentation failed for device %d", error.device_id)
                
        return False
        
    def _recover_from_fragmentation(self, error: GPUError) -> bool:
        """Recover from memory fragmentation"""
        logger.info("Attempting fragmentation recovery for device %d", error.device_id)
        
        memory_manager = get_gpu_memory_manager()
        if memory_manager:
            return memory_manager.defragment_memory(error.device_id)
            
        return False
        
    def _recover_from_device_unavailable(self, error: GPUError) -> bool:
        """Recover from device unavailable error"""
        logger.info("Attempting device recovery for device %d", error.device_id)
        
        # Check if device is actually available
        gpu_monitor = get_gpu_monitor()
        if gpu_monitor:
            gpu_info = gpu_monitor.get_gpu_info(error.device_id)
            if gpu_info and gpu_info.is_available:
                logger.info("Device %d is now available", error.device_id)
                return True
                
        return False
        
    def _recover_from_cuda_error(self, error: GPUError) -> bool:
        """Recover from general CUDA error"""
        logger.info("Attempting CUDA error recovery for device %d", error.device_id)
        
        # Reset CUDA context
        if TORCH_AVAILABLE:
            try:
                torch.cuda.set_device(error.device_id)
                torch.cuda.empty_cache()
                logger.info("Reset CUDA context for device %d", error.device_id)
                return True
            except Exception as e:
                logger.error("Failed to reset CUDA context: %s", e)
                
        return False
        
    async def _recovery_loop(self):
        """Periodic recovery loop"""
        while not self._stop_event.is_set():
            try:
                await self._check_device_health()
                await self._cleanup_old_errors()
                await asyncio.sleep(60.0)  # Check every minute
            except Exception as e:
                logger.error("Error in recovery loop: %s", e)
                await asyncio.sleep(60.0)
                
    async def _check_device_health(self):
        """Check health of all GPU devices"""
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            return
            
        available_gpus = gpu_monitor.get_available_gpus()
        for gpu_info in available_gpus:
            # Check for high memory usage
            usage_percent = gpu_info.used_memory_mb / gpu_info.total_memory_mb * 100
            if usage_percent > 95:
                error = GPUError(
                    GPUErrorType.CUDA_OOM,
                    gpu_info.device_id,
                    f"High memory usage: {usage_percent:.1f}%"
                )
                self.handle_error(error)
                
    async def _cleanup_old_errors(self):
        """Clean up old error history"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        with self._lock:
            self.error_history = [
                error for error in self.error_history 
                if error.timestamp > cutoff_time
            ]
            
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        with self._lock:
            recent_errors = [
                error for error in self.error_history
                if error.timestamp > datetime.now() - timedelta(hours=1)
            ]
            
            error_counts = {}
            for error in recent_errors:
                error_type = error.error_type.value
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
            return {
                "total_errors_last_hour": len(recent_errors),
                "error_counts": error_counts,
                "recent_errors": [
                    {
                        "type": error.error_type.value,
                        "device_id": error.device_id,
                        "message": error.message,
                        "timestamp": error.timestamp.isoformat(),
                        "retry_count": error.retry_count
                    }
                    for error in recent_errors[-10:]  # Last 10 errors
                ]
            }
            
    def is_device_healthy(self, device_id: int) -> bool:
        """Check if a device is healthy based on recent errors"""
        with self._lock:
            recent_errors = [
                error for error in self.error_history
                if (error.device_id == device_id and 
                    error.timestamp > datetime.now() - timedelta(minutes=5))
            ]
            
            # Consider device unhealthy if it has had multiple recent errors
            return len(recent_errors) < 3


class GPUAvailabilityManager:
    """Manages GPU availability and graceful degradation"""
    
    def __init__(self):
        self.unavailable_devices: set = set()
        self.fallback_to_cpu = True
        self._lock = threading.Lock()
        
    def mark_device_unavailable(self, device_id: int, reason: str):
        """Mark a device as unavailable"""
        with self._lock:
            self.unavailable_devices.add(device_id)
            logger.warning("Marked GPU %d as unavailable: %s", device_id, reason)
            
    def mark_device_available(self, device_id: int):
        """Mark a device as available"""
        with self._lock:
            self.unavailable_devices.discard(device_id)
            logger.info("Marked GPU %d as available", device_id)
            
    def is_device_available(self, device_id: int) -> bool:
        """Check if a device is available"""
        with self._lock:
            return device_id not in self.unavailable_devices
            
    def get_available_devices(self) -> List[int]:
        """Get list of available device IDs"""
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            return []
            
        available_gpus = gpu_monitor.get_available_gpus()
        with self._lock:
            return [
                gpu.device_id for gpu in available_gpus
                if gpu.device_id not in self.unavailable_devices
            ]
            
    def should_fallback_to_cpu(self) -> bool:
        """Check if we should fallback to CPU"""
        available_devices = self.get_available_devices()
        return self.fallback_to_cpu and len(available_devices) == 0


# Global instances
gpu_error_handler: Optional[GPUErrorHandler] = None
gpu_availability_manager: Optional[GPUAvailabilityManager] = None


async def init_gpu_error_handler():
    """Initialize global GPU error handler"""
    global gpu_error_handler, gpu_availability_manager
    if gpu_error_handler is None:
        gpu_error_handler = GPUErrorHandler()
        gpu_availability_manager = GPUAvailabilityManager()
        await gpu_error_handler.start()


async def shutdown_gpu_error_handler():
    """Shutdown global GPU error handler"""
    global gpu_error_handler
    if gpu_error_handler is not None:
        await gpu_error_handler.stop()
        gpu_error_handler = None


def get_gpu_error_handler() -> Optional[GPUErrorHandler]:
    """Get global GPU error handler instance"""
    return gpu_error_handler


def get_gpu_availability_manager() -> Optional[GPUAvailabilityManager]:
    """Get global GPU availability manager instance"""
    return gpu_availability_manager
