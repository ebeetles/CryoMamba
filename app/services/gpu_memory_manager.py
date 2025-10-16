import asyncio
import logging
import gc
from typing import Dict, Optional, List, Any
from datetime import datetime
import threading
import weakref

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from app.services.gpu_monitor import get_gpu_monitor

logger = logging.getLogger(__name__)


class MemoryAllocation:
    """Tracks a memory allocation"""
    def __init__(self, job_id: str, device_id: int, size_bytes: int, 
                 allocation_time: datetime, allocation_type: str = "tensor"):
        self.job_id = job_id
        self.device_id = device_id
        self.size_bytes = size_bytes
        self.allocation_time = allocation_time
        self.allocation_type = allocation_type
        self.tensor_ref: Optional[weakref.ref] = None
        
    def set_tensor_ref(self, tensor):
        """Set weak reference to the tensor"""
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            self.tensor_ref = weakref.ref(tensor)
            
    def is_valid(self) -> bool:
        """Check if the tensor is still valid"""
        if self.tensor_ref is None:
            return True
        return self.tensor_ref() is not None


class GPUMemoryManager:
    """Manages GPU memory allocation, monitoring, and cleanup"""
    
    def __init__(self, max_memory_usage_percent: float = 90.0, 
                 fragmentation_threshold_percent: float = 20.0):
        self.max_memory_usage_percent = max_memory_usage_percent
        self.fragmentation_threshold_percent = fragmentation_threshold_percent
        self.allocations: Dict[str, List[MemoryAllocation]] = {}  # job_id -> allocations
        self._lock = threading.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
    async def start(self):
        """Start memory management"""
        logger.info("Starting GPU memory manager")
        self._stop_event.clear()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop memory management"""
        logger.info("Stopping GPU memory manager")
        self._stop_event.set()
        if self._cleanup_task:
            await self._cleanup_task
        self._cleanup_task = None
        
    def allocate_memory(self, job_id: str, device_id: int, size_bytes: int, 
                      tensor: Optional[Any] = None) -> bool:
        """Allocate memory for a job"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available for memory allocation")
            return False
            
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            logger.error("GPU monitor not available")
            return False
            
        gpu_info = gpu_monitor.get_gpu_info(device_id)
        if not gpu_info or not gpu_info.is_available:
            logger.error("GPU %d not available", device_id)
            return False
            
        # Check memory limits
        size_mb = size_bytes // (1024 * 1024)
        if gpu_info.free_memory_mb < size_mb:
            logger.warning("Insufficient GPU memory: needed %d MB, available %d MB",
                         size_mb, gpu_info.free_memory_mb)
            return False
            
        # Check total memory usage
        total_memory_mb = gpu_info.total_memory_mb
        used_memory_mb = gpu_info.used_memory_mb
        if (used_memory_mb + size_mb) / total_memory_mb * 100 > self.max_memory_usage_percent:
            logger.warning("Memory usage would exceed limit: %d%% > %d%%",
                         (used_memory_mb + size_mb) / total_memory_mb * 100,
                         self.max_memory_usage_percent)
            return False
            
        # Create allocation record
        allocation = MemoryAllocation(
            job_id=job_id,
            device_id=device_id,
            size_bytes=size_bytes,
            allocation_time=datetime.now(),
            allocation_type="tensor"
        )
        
        if tensor is not None:
            allocation.set_tensor_ref(tensor)
            
        with self._lock:
            if job_id not in self.allocations:
                self.allocations[job_id] = []
            self.allocations[job_id].append(allocation)
            
        logger.debug("Allocated %d bytes for job %s on GPU %d", 
                    size_bytes, job_id, device_id)
        return True
        
    def deallocate_job_memory(self, job_id: str) -> bool:
        """Deallocate all memory for a job"""
        with self._lock:
            if job_id not in self.allocations:
                return False
                
            allocations = self.allocations.pop(job_id)
            
        # Force garbage collection to free memory
        if TORCH_AVAILABLE:
            for allocation in allocations:
                if allocation.tensor_ref and allocation.tensor_ref():
                    # Tensor still exists, try to delete it
                    try:
                        tensor = allocation.tensor_ref()
                        if tensor is not None:
                            del tensor
                    except Exception as e:
                        logger.debug("Error deleting tensor reference: %s", e)
                        
        # Force garbage collection
        gc.collect()
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
            
        logger.info("Deallocated %d memory allocations for job %s", 
                   len(allocations), job_id)
        return True
        
    def get_memory_usage(self, device_id: int) -> Dict[str, Any]:
        """Get memory usage information for a device"""
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            return {}
            
        gpu_info = gpu_monitor.get_gpu_info(device_id)
        if not gpu_info:
            return {}
            
        with self._lock:
            job_allocations = {}
            total_allocated_bytes = 0
            
            for job_id, allocations in self.allocations.items():
                job_bytes = sum(alloc.size_bytes for alloc in allocations 
                              if alloc.device_id == device_id)
                if job_bytes > 0:
                    job_allocations[job_id] = job_bytes
                    total_allocated_bytes += job_bytes
                    
        return {
            "device_id": device_id,
            "total_memory_mb": gpu_info.total_memory_mb,
            "used_memory_mb": gpu_info.used_memory_mb,
            "free_memory_mb": gpu_info.free_memory_mb,
            "allocated_by_jobs_mb": total_allocated_bytes // (1024 * 1024),
            "job_allocations": job_allocations,
            "utilization_percent": gpu_info.utilization_percent
        }
        
    def check_memory_fragmentation(self, device_id: int) -> bool:
        """Check if memory is fragmented"""
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            return False
            
        gpu_info = gpu_monitor.get_gpu_info(device_id)
        if not gpu_info:
            return False
            
        # Simple fragmentation check based on free memory vs total allocated
        total_memory_mb = gpu_info.total_memory_mb
        used_memory_mb = gpu_info.used_memory_mb
        free_memory_mb = gpu_info.free_memory_mb
        
        if total_memory_mb == 0:
            return False
            
        # If we have a lot of used memory but small free chunks, consider fragmented
        fragmentation_ratio = free_memory_mb / total_memory_mb * 100
        return fragmentation_ratio < self.fragmentation_threshold_percent
        
    def defragment_memory(self, device_id: int) -> bool:
        """Attempt to defragment memory"""
        if not TORCH_AVAILABLE:
            return False
            
        logger.info("Attempting memory defragmentation on GPU %d", device_id)
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Check if fragmentation improved
        fragmentation_before = self.check_memory_fragmentation(device_id)
        fragmentation_after = self.check_memory_fragmentation(device_id)
        
        if fragmentation_before and not fragmentation_after:
            logger.info("Memory defragmentation successful on GPU %d", device_id)
            return True
        else:
            logger.debug("Memory defragmentation had no effect on GPU %d", device_id)
            return False
            
    async def _cleanup_loop(self):
        """Periodic cleanup loop"""
        while not self._stop_event.is_set():
            try:
                await self._cleanup_invalid_allocations()
                await self._check_memory_pressure()
                await asyncio.sleep(30.0)  # Check every 30 seconds
            except Exception as e:
                logger.error("Error in memory cleanup loop: %s", e)
                await asyncio.sleep(30.0)
                
    async def _cleanup_invalid_allocations(self):
        """Clean up invalid memory allocations"""
        with self._lock:
            jobs_to_cleanup = []
            for job_id, allocations in self.allocations.items():
                valid_allocations = [alloc for alloc in allocations if alloc.is_valid()]
                if len(valid_allocations) != len(allocations):
                    self.allocations[job_id] = valid_allocations
                    if not valid_allocations:
                        jobs_to_cleanup.append(job_id)
                        
            # Remove jobs with no valid allocations
            for job_id in jobs_to_cleanup:
                del self.allocations[job_id]
                
    async def _check_memory_pressure(self):
        """Check for memory pressure and take action"""
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            return
            
        available_gpus = gpu_monitor.get_available_gpus()
        for gpu_info in available_gpus:
            # Check if memory usage is high
            usage_percent = gpu_info.used_memory_mb / gpu_info.total_memory_mb * 100
            if usage_percent > self.max_memory_usage_percent:
                logger.warning("High memory usage on GPU %d: %d%%", 
                             gpu_info.device_id, usage_percent)
                
                # Check for fragmentation
                if self.check_memory_fragmentation(gpu_info.device_id):
                    logger.info("Memory fragmentation detected on GPU %d, attempting defragmentation",
                              gpu_info.device_id)
                    self.defragment_memory(gpu_info.device_id)
                    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of all memory allocations"""
        with self._lock:
            summary = {
                "total_jobs": len(self.allocations),
                "total_allocations": sum(len(allocations) for allocations in self.allocations.values()),
                "jobs": {}
            }
            
            for job_id, allocations in self.allocations.items():
                total_bytes = sum(alloc.size_bytes for alloc in allocations)
                summary["jobs"][job_id] = {
                    "allocations": len(allocations),
                    "total_bytes": total_bytes,
                    "total_mb": total_bytes // (1024 * 1024)
                }
                
            return summary


# Global GPU memory manager instance
gpu_memory_manager: Optional[GPUMemoryManager] = None


async def init_gpu_memory_manager():
    """Initialize global GPU memory manager"""
    global gpu_memory_manager
    if gpu_memory_manager is None:
        gpu_memory_manager = GPUMemoryManager()
        await gpu_memory_manager.start()


async def shutdown_gpu_memory_manager():
    """Shutdown global GPU memory manager"""
    global gpu_memory_manager
    if gpu_memory_manager is not None:
        await gpu_memory_manager.stop()
        gpu_memory_manager = None


def get_gpu_memory_manager() -> Optional[GPUMemoryManager]:
    """Get global GPU memory manager instance"""
    return gpu_memory_manager
