import asyncio
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime
from enum import Enum
import heapq
import threading

from app.services.gpu_monitor import GPUInfo, GPUResourceAllocation, get_gpu_monitor
from app.models import JobRecord, JobState

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class GPUJobScheduler:
    """GPU job scheduler with resource management"""
    
    def __init__(self, max_concurrent_jobs: int = 1):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue: List[tuple] = []  # Priority queue: (priority, timestamp, job_id)
        self.running_jobs: Dict[str, JobRecord] = {}
        self.gpu_allocations: Dict[str, GPUResourceAllocation] = {}
        self._lock = threading.Lock()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
    async def start(self):
        """Start the GPU scheduler"""
        logger.info("Starting GPU job scheduler")
        self._stop_event.clear()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
    async def stop(self):
        """Stop the GPU scheduler"""
        logger.info("Stopping GPU job scheduler")
        self._stop_event.set()
        if self._scheduler_task:
            await self._scheduler_task
        self._scheduler_task = None
        
    async def submit_job(self, job: JobRecord, priority: JobPriority = JobPriority.NORMAL,
                        estimated_memory_mb: int = 1024, estimated_duration_seconds: Optional[int] = None) -> bool:
        """Submit a job to the GPU scheduler"""
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            logger.error("GPU monitor not available")
            return False
            
        with self._lock:
            # Check if we can accept more jobs
            if len(self.running_jobs) >= self.max_concurrent_jobs:
                # Add to queue
                heapq.heappush(self.job_queue, (
                    -priority.value,  # Negative for max-heap behavior
                    datetime.now().timestamp(),
                    job.job_id
                ))
                logger.info("Job %s queued (priority: %s)", job.job_id, priority.name)
                return True
            else:
                # Try to allocate GPU immediately
                return await self._try_allocate_gpu(job, estimated_memory_mb, estimated_duration_seconds)
                
    async def _try_allocate_gpu(self, job: JobRecord, estimated_memory_mb: int, 
                              estimated_duration_seconds: Optional[int] = None) -> bool:
        """Try to allocate GPU resources for a job"""
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            return False
            
        available_gpus = gpu_monitor.get_available_gpus()
        if not available_gpus:
            logger.warning("No available GPUs for job %s", job.job_id)
            return False
            
        # Find best GPU (highest free memory)
        best_gpu = max(available_gpus, key=lambda gpu: gpu.free_memory_mb)
        
        # Check if GPU has enough memory
        if best_gpu.free_memory_mb < estimated_memory_mb:
            logger.warning("Insufficient GPU memory for job %s (needed: %d MB, available: %d MB)",
                         job.job_id, estimated_memory_mb, best_gpu.free_memory_mb)
            return False
            
        # Allocate resources
        if gpu_monitor.allocate_gpu_resources(job.job_id, best_gpu.device_id, 
                                            estimated_memory_mb, estimated_duration_seconds):
            self.running_jobs[job.job_id] = job
            allocation = gpu_monitor.get_resource_allocation(job.job_id)
            if allocation:
                self.gpu_allocations[job.job_id] = allocation
            logger.info("Job %s allocated GPU %d (%d MB)", 
                       job.job_id, best_gpu.device_id, estimated_memory_mb)
            return True
            
        return False
        
    async def complete_job(self, job_id: str) -> bool:
        """Mark a job as completed and free resources"""
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            return False
            
        with self._lock:
            if job_id in self.running_jobs:
                # Free GPU resources
                gpu_monitor.deallocate_gpu_resources(job_id)
                del self.running_jobs[job_id]
                self.gpu_allocations.pop(job_id, None)
                
                # Try to schedule next job from queue
                await self._schedule_next_job()
                logger.info("Job %s completed, resources freed", job_id)
                return True
                
        return False
        
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job and free resources"""
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            return False
            
        with self._lock:
            # Remove from queue if present
            original_queue_length = len(self.job_queue)
            self.job_queue = [(p, t, jid) for p, t, jid in self.job_queue if jid != job_id]
            heapq.heapify(self.job_queue)  # Re-heapify after removal
            
            # Free resources if running
            if job_id in self.running_jobs:
                gpu_monitor.deallocate_gpu_resources(job_id)
                del self.running_jobs[job_id]
                self.gpu_allocations.pop(job_id, None)
                
                # Try to schedule next job
                await self._schedule_next_job()
                logger.info("Job %s cancelled, resources freed", job_id)
                return True
            
            # Check if job was in queue
            if len(self.job_queue) < original_queue_length:
                logger.info("Job %s cancelled from queue", job_id)
                return True
                
        return False
        
    async def _schedule_next_job(self):
        """Schedule the next job from the queue"""
        if not self.job_queue or len(self.running_jobs) >= self.max_concurrent_jobs:
            return
            
        # Get highest priority job
        priority, timestamp, job_id = heapq.heappop(self.job_queue)
        
        # Try to allocate GPU (this would need job record - simplified for now)
        logger.info("Attempting to schedule queued job %s", job_id)
        
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while not self._stop_event.is_set():
            try:
                await self._monitor_running_jobs()
                await asyncio.sleep(1.0)  # Check every second
            except Exception as e:
                logger.error("Error in scheduler loop: %s", e)
                await asyncio.sleep(1.0)
                
    async def _monitor_running_jobs(self):
        """Monitor running jobs for completion or errors"""
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            return
            
        with self._lock:
            completed_jobs = []
            for job_id, job in self.running_jobs.items():
                # Check if job is in terminal state
                if job.state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
                    completed_jobs.append(job_id)
                    
            # Clean up completed jobs
            for job_id in completed_jobs:
                await self.complete_job(job_id)
                
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        with self._lock:
            return {
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "running_jobs": len(self.running_jobs),
                "queued_jobs": len(self.job_queue),
                "running_job_ids": list(self.running_jobs.keys()),
                "queued_job_ids": [job_id for _, _, job_id in self.job_queue]
            }
            
    def get_job_allocation(self, job_id: str) -> Optional[GPUResourceAllocation]:
        """Get GPU allocation for a job"""
        return self.gpu_allocations.get(job_id)
        
    def is_job_running(self, job_id: str) -> bool:
        """Check if a job is currently running"""
        with self._lock:
            return job_id in self.running_jobs
            
    def is_job_queued(self, job_id: str) -> bool:
        """Check if a job is queued"""
        with self._lock:
            return any(job_id == jid for _, _, jid in self.job_queue)


# Global GPU scheduler instance
gpu_scheduler: Optional[GPUJobScheduler] = None


async def init_gpu_scheduler():
    """Initialize global GPU scheduler"""
    global gpu_scheduler
    if gpu_scheduler is None:
        gpu_scheduler = GPUJobScheduler(max_concurrent_jobs=1)
        await gpu_scheduler.start()


async def shutdown_gpu_scheduler():
    """Shutdown global GPU scheduler"""
    global gpu_scheduler
    if gpu_scheduler is not None:
        await gpu_scheduler.stop()
        gpu_scheduler = None


def get_gpu_scheduler() -> Optional[GPUJobScheduler]:
    """Get global GPU scheduler instance"""
    return gpu_scheduler
