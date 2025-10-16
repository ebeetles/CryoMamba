import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU information and current state"""
    device_id: int
    name: str
    total_memory_mb: int
    free_memory_mb: int
    used_memory_mb: int
    utilization_percent: float
    temperature_c: Optional[float] = None
    power_usage_w: Optional[float] = None
    is_available: bool = True
    last_updated: datetime = None


@dataclass
class GPUResourceAllocation:
    """Tracks GPU resource allocation for a job"""
    job_id: str
    device_id: int
    allocated_memory_mb: int
    allocated_at: datetime
    estimated_duration_seconds: Optional[int] = None


class GPUMonitor:
    """Monitors GPU resources and provides availability information"""
    
    def __init__(self, update_interval_seconds: float = 5.0):
        self.update_interval = update_interval_seconds
        self.gpus: Dict[int, GPUInfo] = {}
        self.resource_allocations: Dict[str, GPUResourceAllocation] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._lock = threading.Lock()
        
    async def start(self):
        """Start GPU monitoring"""
        logger.info("Starting GPU monitoring")
        self._stop_event.clear()
        await self._detect_gpus()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
    async def stop(self):
        """Stop GPU monitoring"""
        logger.info("Stopping GPU monitoring")
        self._stop_event.set()
        if self._monitoring_task:
            await self._monitoring_task
        self._monitoring_task = None
        
    async def _detect_gpus(self):
        """Detect available GPUs"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info("Found %d CUDA devices", device_count)
                
                for device_id in range(device_count):
                    device_name = torch.cuda.get_device_name(device_id)
                    total_memory = torch.cuda.get_device_properties(device_id).total_memory
                    total_memory_mb = total_memory // (1024 * 1024)
                    
                    self.gpus[device_id] = GPUInfo(
                        device_id=device_id,
                        name=device_name,
                        total_memory_mb=total_memory_mb,
                        free_memory_mb=0,  # Will be updated in monitoring loop
                        used_memory_mb=0,
                        utilization_percent=0.0,
                        is_available=True,
                        last_updated=datetime.now()
                    )
            else:
                logger.warning("CUDA not available - no GPUs detected")
        except ImportError:
            logger.warning("PyTorch not available - GPU monitoring disabled")
        except Exception as e:
            logger.error("Error detecting GPUs: %s", e)
            
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                await self._update_gpu_stats()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error("Error in GPU monitoring loop: %s", e)
                await asyncio.sleep(self.update_interval)
                
    async def _update_gpu_stats(self):
        """Update GPU statistics"""
        try:
            import torch
            if not torch.cuda.is_available():
                return
                
            for device_id, gpu_info in self.gpus.items():
                try:
                    # Get memory info
                    torch.cuda.set_device(device_id)
                    total_memory = torch.cuda.get_device_properties(device_id).total_memory
                    allocated_memory = torch.cuda.memory_allocated(device_id)
                    cached_memory = torch.cuda.memory_reserved(device_id)
                    
                    total_memory_mb = total_memory // (1024 * 1024)
                    allocated_memory_mb = allocated_memory // (1024 * 1024)
                    cached_memory_mb = cached_memory // (1024 * 1024)
                    
                    # Calculate utilization (simplified - based on memory usage)
                    memory_utilization = (allocated_memory_mb / total_memory_mb) * 100.0
                    
                    # Update GPU info
                    with self._lock:
                        gpu_info.total_memory_mb = total_memory_mb
                        gpu_info.used_memory_mb = allocated_memory_mb
                        gpu_info.free_memory_mb = total_memory_mb - cached_memory_mb
                        gpu_info.utilization_percent = memory_utilization
                        gpu_info.last_updated = datetime.now()
                        gpu_info.is_available = True
                        
                except Exception as e:
                    logger.error("Error updating GPU %d stats: %s", device_id, e)
                    with self._lock:
                        gpu_info.is_available = False
                        
        except ImportError:
            pass  # PyTorch not available
        except Exception as e:
            logger.error("Error updating GPU stats: %s", e)
            
    def get_available_gpus(self) -> List[GPUInfo]:
        """Get list of available GPUs"""
        with self._lock:
            return [gpu for gpu in self.gpus.values() if gpu.is_available]
            
    def get_gpu_info(self, device_id: int) -> Optional[GPUInfo]:
        """Get info for specific GPU"""
        with self._lock:
            return self.gpus.get(device_id)
            
    def allocate_gpu_resources(self, job_id: str, device_id: int, 
                             memory_mb: int, estimated_duration_seconds: Optional[int] = None) -> bool:
        """Allocate GPU resources for a job"""
        with self._lock:
            gpu_info = self.gpus.get(device_id)
            if not gpu_info or not gpu_info.is_available:
                return False
                
            # Check if enough memory is available
            if gpu_info.free_memory_mb < memory_mb:
                return False
                
            # Check if device is already allocated to another job
            for allocation in self.resource_allocations.values():
                if allocation.device_id == device_id:
                    return False
                    
            # Allocate resources
            allocation = GPUResourceAllocation(
                job_id=job_id,
                device_id=device_id,
                allocated_memory_mb=memory_mb,
                allocated_at=datetime.now(),
                estimated_duration_seconds=estimated_duration_seconds
            )
            self.resource_allocations[job_id] = allocation
            return True
            
    def deallocate_gpu_resources(self, job_id: str) -> bool:
        """Deallocate GPU resources for a job"""
        with self._lock:
            if job_id in self.resource_allocations:
                del self.resource_allocations[job_id]
                return True
            return False
            
    def get_resource_allocation(self, job_id: str) -> Optional[GPUResourceAllocation]:
        """Get resource allocation for a job"""
        with self._lock:
            return self.resource_allocations.get(job_id)
            
    def get_gpu_utilization_summary(self) -> Dict[str, Any]:
        """Get summary of GPU utilization"""
        with self._lock:
            summary = {
                "total_gpus": len(self.gpus),
                "available_gpus": len([gpu for gpu in self.gpus.values() if gpu.is_available]),
                "allocated_jobs": len(self.resource_allocations),
                "gpus": []
            }
            
            for gpu_info in self.gpus.values():
                gpu_summary = {
                    "device_id": gpu_info.device_id,
                    "name": gpu_info.name,
                    "total_memory_mb": gpu_info.total_memory_mb,
                    "used_memory_mb": gpu_info.used_memory_mb,
                    "free_memory_mb": gpu_info.free_memory_mb,
                    "utilization_percent": gpu_info.utilization_percent,
                    "is_available": gpu_info.is_available,
                    "last_updated": gpu_info.last_updated.isoformat() if gpu_info.last_updated else None
                }
                summary["gpus"].append(gpu_summary)
                
            return summary


# Global GPU monitor instance
gpu_monitor: Optional[GPUMonitor] = None


async def init_gpu_monitor():
    """Initialize global GPU monitor"""
    global gpu_monitor
    if gpu_monitor is None:
        gpu_monitor = GPUMonitor()
        await gpu_monitor.start()


async def shutdown_gpu_monitor():
    """Shutdown global GPU monitor"""
    global gpu_monitor
    if gpu_monitor is not None:
        await gpu_monitor.stop()
        gpu_monitor = None


def get_gpu_monitor() -> Optional[GPUMonitor]:
    """Get global GPU monitor instance"""
    return gpu_monitor
