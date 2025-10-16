"""
Performance monitoring and optimization for preview streaming.

This module provides performance monitoring, caching, and compression
optimizations for real-time preview streaming.
"""

import time
import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import zlib
import pickle

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for preview streaming."""
    job_id: str
    timestamp: datetime
    preview_generation_time_ms: float
    downsampling_time_ms: float
    serialization_time_ms: float
    websocket_send_time_ms: float
    preview_size_bytes: int
    compression_ratio: float
    total_latency_ms: float


class PerformanceMonitor:
    """Monitors performance metrics for preview streaming."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.job_metrics: Dict[str, List[PerformanceMetrics]] = {}
        self.start_times: Dict[str, float] = {}
    
    def start_timing(self, job_id: str, operation: str) -> str:
        """
        Start timing an operation.
        
        Args:
            job_id: Job ID
            operation: Operation name
        
        Returns:
            Timing key for stopping timing
        """
        timing_key = f"{job_id}_{operation}_{time.time()}"
        self.start_times[timing_key] = time.time()
        return timing_key
    
    def stop_timing(self, timing_key: str) -> float:
        """
        Stop timing an operation and return duration.
        
        Args:
            timing_key: Timing key from start_timing
        
        Returns:
            Duration in milliseconds
        """
        if timing_key not in self.start_times:
            logger.warning(f"Timing key not found: {timing_key}")
            return 0.0
        
        duration_ms = (time.time() - self.start_times[timing_key]) * 1000
        del self.start_times[timing_key]
        return duration_ms
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.metrics_history.append(metrics)
        
        if metrics.job_id not in self.job_metrics:
            self.job_metrics[metrics.job_id] = []
        
        self.job_metrics[metrics.job_id].append(metrics)
        
        # Keep only recent metrics per job
        if len(self.job_metrics[metrics.job_id]) > 100:
            self.job_metrics[metrics.job_id] = self.job_metrics[metrics.job_id][-100:]
    
    def get_job_performance(self, job_id: str) -> Dict[str, Any]:
        """
        Get performance statistics for a job.
        
        Args:
            job_id: Job ID
        
        Returns:
            Performance statistics dictionary
        """
        if job_id not in self.job_metrics:
            return {}
        
        metrics = self.job_metrics[job_id]
        if not metrics:
            return {}
        
        return {
            "total_previews": len(metrics),
            "avg_generation_time_ms": np.mean([m.preview_generation_time_ms for m in metrics]),
            "avg_downsampling_time_ms": np.mean([m.downsampling_time_ms for m in metrics]),
            "avg_serialization_time_ms": np.mean([m.serialization_time_ms for m in metrics]),
            "avg_websocket_send_time_ms": np.mean([m.websocket_send_time_ms for m in metrics]),
            "avg_total_latency_ms": np.mean([m.total_latency_ms for m in metrics]),
            "avg_preview_size_bytes": np.mean([m.preview_size_bytes for m in metrics]),
            "avg_compression_ratio": np.mean([m.compression_ratio for m in metrics]),
            "min_latency_ms": min([m.total_latency_ms for m in metrics]),
            "max_latency_ms": max([m.total_latency_ms for m in metrics])
        }
    
    def get_overall_performance(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        if not self.metrics_history:
            return {}
        
        return {
            "total_previews": len(self.metrics_history),
            "avg_generation_time_ms": np.mean([m.preview_generation_time_ms for m in self.metrics_history]),
            "avg_downsampling_time_ms": np.mean([m.downsampling_time_ms for m in self.metrics_history]),
            "avg_serialization_time_ms": np.mean([m.serialization_time_ms for m in self.metrics_history]),
            "avg_websocket_send_time_ms": np.mean([m.websocket_send_time_ms for m in self.metrics_history]),
            "avg_total_latency_ms": np.mean([m.total_latency_ms for m in self.metrics_history]),
            "avg_preview_size_bytes": np.mean([m.preview_size_bytes for m in self.metrics_history]),
            "avg_compression_ratio": np.mean([m.compression_ratio for m in self.metrics_history])
        }


class PreviewCache:
    """Cache for preview data to reduce computation."""
    
    def __init__(self, max_size: int = 100):
        """
        Initialize preview cache.
        
        Args:
            max_size: Maximum number of cached previews
        """
        self.max_size = max_size
        self.cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.access_times: Dict[str, float] = {}
    
    def _evict_oldest(self):
        """Evict oldest cached item."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get cached preview data.
        
        Args:
            key: Cache key
        
        Returns:
            Cached preview data or None if not found
        """
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key][0]
        return None
    
    def put(self, key: str, data: np.ndarray, metadata: Dict[str, Any] = None):
        """
        Cache preview data.
        
        Args:
            key: Cache key
            data: Preview data to cache
            metadata: Optional metadata
        """
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = (data.copy(), time.time())
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_items": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": 0.0  # Would need to track hits/misses for accurate rate
        }


class PreviewCompressor:
    """Compression utilities for preview data."""
    
    @staticmethod
    def compress_data(data: np.ndarray, method: str = "zlib") -> Tuple[bytes, float]:
        """
        Compress preview data.
        
        Args:
            data: Data to compress
            method: Compression method ("zlib", "pickle")
        
        Returns:
            Tuple of (compressed_data, compression_ratio)
        """
        original_size = data.nbytes
        
        if method == "zlib":
            compressed = zlib.compress(data.tobytes())
        elif method == "pickle":
            compressed = pickle.dumps(data)
        else:
            raise ValueError(f"Unknown compression method: {method}")
        
        compression_ratio = len(compressed) / original_size
        return compressed, compression_ratio
    
    @staticmethod
    def decompress_data(compressed_data: bytes, shape: Tuple[int, ...], dtype: str, method: str = "zlib") -> np.ndarray:
        """
        Decompress preview data.
        
        Args:
            compressed_data: Compressed data
            shape: Original data shape
            dtype: Original data dtype
            method: Compression method
        
        Returns:
            Decompressed numpy array
        """
        if method == "zlib":
            decompressed_bytes = zlib.decompress(compressed_data)
        elif method == "pickle":
            return pickle.loads(compressed_data)
        else:
            raise ValueError(f"Unknown compression method: {method}")
        
        data = np.frombuffer(decompressed_bytes, dtype=dtype)
        return data.reshape(shape)


class AdaptiveStreamingController:
    """Adaptive controller for streaming performance."""
    
    def __init__(self, target_latency_ms: float = 1000.0):
        """
        Initialize adaptive streaming controller.
        
        Args:
            target_latency_ms: Target latency in milliseconds
        """
        self.target_latency_ms = target_latency_ms
        self.current_frequency = 1.0  # Hz
        self.min_frequency = 0.1  # Hz
        self.max_frequency = 5.0  # Hz
        self.adaptation_factor = 0.1
        self.performance_history: deque = deque(maxlen=10)
    
    def update_performance(self, latency_ms: float):
        """
        Update performance metrics and adjust streaming parameters.
        
        Args:
            latency_ms: Current latency in milliseconds
        """
        self.performance_history.append(latency_ms)
        
        if len(self.performance_history) < 3:
            return
        
        avg_latency = np.mean(list(self.performance_history))
        
        # Adjust frequency based on latency
        if avg_latency > self.target_latency_ms * 1.2:
            # Latency too high, reduce frequency
            self.current_frequency = max(
                self.min_frequency,
                self.current_frequency * (1 - self.adaptation_factor)
            )
        elif avg_latency < self.target_latency_ms * 0.8:
            # Latency low, can increase frequency
            self.current_frequency = min(
                self.max_frequency,
                self.current_frequency * (1 + self.adaptation_factor)
            )
        
        logger.debug(f"Adapted streaming frequency to {self.current_frequency:.2f} Hz (latency: {avg_latency:.1f}ms)")
    
    def get_current_frequency(self) -> float:
        """Get current streaming frequency."""
        return self.current_frequency
    
    def reset(self):
        """Reset controller state."""
        self.current_frequency = 1.0
        self.performance_history.clear()


# Global instances
performance_monitor = PerformanceMonitor()
preview_cache = PreviewCache()
adaptive_controller = AdaptiveStreamingController()
