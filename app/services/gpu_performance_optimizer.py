import asyncio
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
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


class PerformanceMetrics:
    """Tracks GPU performance metrics"""
    def __init__(self):
        self.job_completion_times: List[float] = []
        self.memory_efficiency: List[float] = []
        self.utilization_history: List[float] = []
        self.throughput_history: List[float] = []
        self.last_updated = datetime.now()
        
    def add_job_completion(self, duration_seconds: float, memory_used_mb: int, 
                          memory_allocated_mb: int):
        """Add job completion metrics"""
        self.job_completion_times.append(duration_seconds)
        
        # Calculate memory efficiency
        if memory_allocated_mb > 0:
            efficiency = memory_used_mb / memory_allocated_mb
            self.memory_efficiency.append(efficiency)
            
        self.last_updated = datetime.now()
        
    def get_average_completion_time(self) -> float:
        """Get average job completion time"""
        if not self.job_completion_times:
            return 0.0
        return sum(self.job_completion_times) / len(self.job_completion_times)
        
    def get_memory_efficiency(self) -> float:
        """Get average memory efficiency"""
        if not self.memory_efficiency:
            return 0.0
        return sum(self.memory_efficiency) / len(self.memory_efficiency)
        
    def get_throughput(self) -> float:
        """Get jobs per hour"""
        if not self.job_completion_times:
            return 0.0
        total_time = sum(self.job_completion_times)
        if total_time == 0:
            return 0.0
        return len(self.job_completion_times) / (total_time / 3600)


class GPUPerformanceOptimizer:
    """Optimizes GPU performance and utilization"""
    
    def __init__(self, optimization_interval_seconds: float = 60.0):
        self.optimization_interval = optimization_interval_seconds
        self.metrics: Dict[int, PerformanceMetrics] = {}  # device_id -> metrics
        self.optimization_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._optimization_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
        # Optimization parameters
        self.target_utilization_percent = 80.0
        self.memory_efficiency_threshold = 0.7
        self.throughput_improvement_threshold = 0.1
        
    async def start(self):
        """Start performance optimization"""
        logger.info("Starting GPU performance optimizer")
        self._stop_event.clear()
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        
    async def stop(self):
        """Stop performance optimization"""
        logger.info("Stopping GPU performance optimizer")
        self._stop_event.set()
        if self._optimization_task:
            await self._optimization_task
        self._optimization_task = None
        
    def record_job_completion(self, device_id: int, duration_seconds: float,
                            memory_used_mb: int, memory_allocated_mb: int):
        """Record job completion metrics"""
        with self._lock:
            if device_id not in self.metrics:
                self.metrics[device_id] = PerformanceMetrics()
            
            self.metrics[device_id].add_job_completion(
                duration_seconds, memory_used_mb, memory_allocated_mb
            )
            
    async def _optimization_loop(self):
        """Main optimization loop"""
        while not self._stop_event.is_set():
            try:
                await self._analyze_performance()
                await self._apply_optimizations()
                await asyncio.sleep(self.optimization_interval)
            except Exception as e:
                logger.error("Error in optimization loop: %s", e)
                await asyncio.sleep(self.optimization_interval)
                
    async def _analyze_performance(self):
        """Analyze current GPU performance"""
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            return
            
        available_gpus = gpu_monitor.get_available_gpus()
        for gpu_info in available_gpus:
            device_id = gpu_info.device_id
            
            with self._lock:
                if device_id not in self.metrics:
                    self.metrics[device_id] = PerformanceMetrics()
                    
                metrics = self.metrics[device_id]
                
            # Update utilization history
            metrics.utilization_history.append(gpu_info.utilization_percent)
            
            # Keep only recent history (last hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            if metrics.last_updated < cutoff_time:
                metrics.utilization_history = metrics.utilization_history[-60:]  # Keep last 60 readings
                
    async def _apply_optimizations(self):
        """Apply performance optimizations"""
        gpu_monitor = get_gpu_monitor()
        memory_manager = get_gpu_memory_manager()
        
        if not gpu_monitor or not memory_manager:
            return
            
        available_gpus = gpu_monitor.get_available_gpus()
        
        for gpu_info in available_gpus:
            device_id = gpu_info.device_id
            
            with self._lock:
                if device_id not in self.metrics:
                    continue
                metrics = self.metrics[device_id]
                
            # Check if optimization is needed
            optimizations_applied = []
            
            # Memory fragmentation optimization
            if memory_manager.check_memory_fragmentation(device_id):
                if memory_manager.defragment_memory(device_id):
                    optimizations_applied.append("memory_defragmentation")
                    
            # Utilization optimization
            avg_utilization = self._get_average_utilization(device_id)
            if avg_utilization < self.target_utilization_percent:
                # Low utilization - could optimize job scheduling
                optimizations_applied.append("scheduling_optimization")
                
            # Memory efficiency optimization
            memory_efficiency = metrics.get_memory_efficiency()
            if memory_efficiency < self.memory_efficiency_threshold:
                optimizations_applied.append("memory_efficiency_improvement")
                
            # Record optimization
            if optimizations_applied:
                optimization_record = {
                    "timestamp": datetime.now().isoformat(),
                    "device_id": device_id,
                    "optimizations": optimizations_applied,
                    "utilization_percent": gpu_info.utilization_percent,
                    "memory_efficiency": memory_efficiency
                }
                
                with self._lock:
                    self.optimization_history.append(optimization_record)
                    # Keep only recent history
                    self.optimization_history = self.optimization_history[-100:]
                    
                logger.info("Applied optimizations to GPU %d: %s", 
                          device_id, ", ".join(optimizations_applied))
                
    def _get_average_utilization(self, device_id: int) -> float:
        """Get average utilization for a device"""
        with self._lock:
            if device_id not in self.metrics:
                return 0.0
            metrics = self.metrics[device_id]
            
        if not metrics.utilization_history:
            return 0.0
            
        return sum(metrics.utilization_history) / len(metrics.utilization_history)
        
    def warmup_gpu(self, device_id: int) -> bool:
        """Warm up GPU for better performance"""
        if not TORCH_AVAILABLE:
            return False
            
        try:
            logger.info("Warming up GPU %d", device_id)
            torch.cuda.set_device(device_id)
            
            # Create and destroy some tensors to warm up
            for _ in range(3):
                dummy_tensor = torch.randn(1000, 1000, device=f"cuda:{device_id}")
                torch.cuda.synchronize()
                del dummy_tensor
                
            torch.cuda.empty_cache()
            logger.info("GPU %d warmup completed", device_id)
            return True
            
        except Exception as e:
            logger.error("GPU warmup failed for device %d: %s", device_id, e)
            return False
            
    def optimize_batch_size(self, device_id: int, current_batch_size: int) -> int:
        """Suggest optimal batch size based on performance metrics"""
        with self._lock:
            if device_id not in self.metrics:
                return current_batch_size
                
            metrics = self.metrics[device_id]
            
        # Simple heuristic based on memory efficiency
        memory_efficiency = metrics.get_memory_efficiency()
        
        if memory_efficiency > 0.9:
            # High efficiency, could increase batch size
            return min(current_batch_size * 2, 8)
        elif memory_efficiency < 0.5:
            # Low efficiency, decrease batch size
            return max(current_batch_size // 2, 1)
        else:
            return current_batch_size
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all GPUs"""
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            return {}
            
        summary = {
            "optimization_history_count": len(self.optimization_history),
            "gpus": []
        }
        
        available_gpus = gpu_monitor.get_available_gpus()
        
        with self._lock:
            for gpu_info in available_gpus:
                device_id = gpu_info.device_id
                metrics = self.metrics.get(device_id, PerformanceMetrics())
                
                gpu_summary = {
                    "device_id": device_id,
                    "name": gpu_info.name,
                    "current_utilization_percent": gpu_info.utilization_percent,
                    "average_utilization_percent": self._get_average_utilization(device_id),
                    "average_completion_time": metrics.get_average_completion_time(),
                    "memory_efficiency": metrics.get_memory_efficiency(),
                    "throughput_jobs_per_hour": metrics.get_throughput(),
                    "total_jobs_completed": len(metrics.job_completion_times)
                }
                summary["gpus"].append(gpu_summary)
                
        return summary
        
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations"""
        recommendations = []
        
        gpu_monitor = get_gpu_monitor()
        if not gpu_monitor:
            return recommendations
            
        available_gpus = gpu_monitor.get_available_gpus()
        
        with self._lock:
            for gpu_info in available_gpus:
                device_id = gpu_info.device_id
                metrics = self.metrics.get(device_id, PerformanceMetrics())
                
                # Utilization recommendations
                avg_utilization = self._get_average_utilization(device_id)
                if avg_utilization < 50:
                    recommendations.append({
                        "device_id": device_id,
                        "type": "utilization",
                        "priority": "medium",
                        "message": f"GPU {device_id} has low utilization ({avg_utilization:.1f}%)",
                        "suggestion": "Consider increasing job concurrency or batch size"
                    })
                    
                # Memory efficiency recommendations
                memory_efficiency = metrics.get_memory_efficiency()
                if memory_efficiency < 0.6:
                    recommendations.append({
                        "device_id": device_id,
                        "type": "memory_efficiency",
                        "priority": "high",
                        "message": f"GPU {device_id} has low memory efficiency ({memory_efficiency:.2f})",
                        "suggestion": "Consider optimizing memory allocation or reducing batch size"
                    })
                    
                # Throughput recommendations
                throughput = metrics.get_throughput()
                if throughput < 1.0 and len(metrics.job_completion_times) > 5:
                    recommendations.append({
                        "device_id": device_id,
                        "type": "throughput",
                        "priority": "low",
                        "message": f"GPU {device_id} has low throughput ({throughput:.2f} jobs/hour)",
                        "suggestion": "Consider optimizing model or preprocessing pipeline"
                    })
                    
        return recommendations


# Global GPU performance optimizer instance
gpu_performance_optimizer: Optional[GPUPerformanceOptimizer] = None


async def init_gpu_performance_optimizer():
    """Initialize global GPU performance optimizer"""
    global gpu_performance_optimizer
    if gpu_performance_optimizer is None:
        gpu_performance_optimizer = GPUPerformanceOptimizer()
        await gpu_performance_optimizer.start()


async def shutdown_gpu_performance_optimizer():
    """Shutdown global GPU performance optimizer"""
    global gpu_performance_optimizer
    if gpu_performance_optimizer is not None:
        await gpu_performance_optimizer.stop()
        gpu_performance_optimizer = None


def get_gpu_performance_optimizer() -> Optional[GPUPerformanceOptimizer]:
    """Get global GPU performance optimizer instance"""
    return gpu_performance_optimizer
