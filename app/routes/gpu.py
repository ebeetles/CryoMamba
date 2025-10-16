from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
import logging

from app.services.gpu_monitor import get_gpu_monitor
from app.services.gpu_scheduler import get_gpu_scheduler
from app.services.gpu_memory_manager import get_gpu_memory_manager
from app.services.gpu_error_handler import get_gpu_error_handler, get_gpu_availability_manager
from app.services.gpu_performance_optimizer import get_gpu_performance_optimizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gpu", tags=["gpu"])


@router.get("/status")
async def get_gpu_status():
    """Get overall GPU status and availability"""
    gpu_monitor = get_gpu_monitor()
    if not gpu_monitor:
        raise HTTPException(status_code=503, detail="GPU monitoring not available")
        
    return gpu_monitor.get_gpu_utilization_summary()


@router.get("/devices")
async def get_gpu_devices():
    """Get list of available GPU devices"""
    gpu_monitor = get_gpu_monitor()
    if not gpu_monitor:
        raise HTTPException(status_code=503, detail="GPU monitoring not available")
        
    available_gpus = gpu_monitor.get_available_gpus()
    return {
        "devices": [
            {
                "device_id": gpu.device_id,
                "name": gpu.name,
                "total_memory_mb": gpu.total_memory_mb,
                "free_memory_mb": gpu.free_memory_mb,
                "used_memory_mb": gpu.used_memory_mb,
                "utilization_percent": gpu.utilization_percent,
                "is_available": gpu.is_available,
                "last_updated": gpu.last_updated.isoformat() if gpu.last_updated else None
            }
            for gpu in available_gpus
        ]
    }


@router.get("/devices/{device_id}")
async def get_gpu_device(device_id: int):
    """Get detailed information for a specific GPU device"""
    gpu_monitor = get_gpu_monitor()
    if not gpu_monitor:
        raise HTTPException(status_code=503, detail="GPU monitoring not available")
        
    gpu_info = gpu_monitor.get_gpu_info(device_id)
    if not gpu_info:
        raise HTTPException(status_code=404, detail=f"GPU device {device_id} not found")
        
    return {
        "device_id": gpu_info.device_id,
        "name": gpu_info.name,
        "total_memory_mb": gpu_info.total_memory_mb,
        "free_memory_mb": gpu_info.free_memory_mb,
        "used_memory_mb": gpu_info.used_memory_mb,
        "utilization_percent": gpu_info.utilization_percent,
        "temperature_c": gpu_info.temperature_c,
        "power_usage_w": gpu_info.power_usage_w,
        "is_available": gpu_info.is_available,
        "last_updated": gpu_info.last_updated.isoformat() if gpu_info.last_updated else None
    }


@router.get("/scheduler/status")
async def get_scheduler_status():
    """Get GPU scheduler status"""
    gpu_scheduler = get_gpu_scheduler()
    if not gpu_scheduler:
        raise HTTPException(status_code=503, detail="GPU scheduler not available")
        
    return gpu_scheduler.get_queue_status()


@router.get("/memory/usage")
async def get_memory_usage():
    """Get GPU memory usage information"""
    memory_manager = get_gpu_memory_manager()
    if not memory_manager:
        raise HTTPException(status_code=503, detail="GPU memory manager not available")
        
    gpu_monitor = get_gpu_monitor()
    if not gpu_monitor:
        raise HTTPException(status_code=503, detail="GPU monitoring not available")
        
    available_gpus = gpu_monitor.get_available_gpus()
    memory_info = {}
    
    for gpu_info in available_gpus:
        device_id = gpu_info.device_id
        memory_info[f"gpu_{device_id}"] = memory_manager.get_memory_usage(device_id)
        
    return {
        "memory_usage": memory_info,
        "allocation_summary": memory_manager.get_allocation_summary()
    }


@router.get("/memory/usage/{device_id}")
async def get_device_memory_usage(device_id: int):
    """Get memory usage for a specific device"""
    memory_manager = get_gpu_memory_manager()
    if not memory_manager:
        raise HTTPException(status_code=503, detail="GPU memory manager not available")
        
    memory_info = memory_manager.get_memory_usage(device_id)
    if not memory_info:
        raise HTTPException(status_code=404, detail=f"Memory info for GPU {device_id} not found")
        
    return memory_info


@router.post("/memory/defragment/{device_id}")
async def defragment_memory(device_id: int):
    """Defragment memory on a specific GPU device"""
    memory_manager = get_gpu_memory_manager()
    if not memory_manager:
        raise HTTPException(status_code=503, detail="GPU memory manager not available")
        
    success = memory_manager.defragment_memory(device_id)
    return {
        "device_id": device_id,
        "success": success,
        "message": "Memory defragmentation completed" if success else "Memory defragmentation failed"
    }


@router.get("/errors/summary")
async def get_error_summary():
    """Get GPU error summary"""
    error_handler = get_gpu_error_handler()
    if not error_handler:
        raise HTTPException(status_code=503, detail="GPU error handler not available")
        
    return error_handler.get_error_summary()


@router.get("/performance/summary")
async def get_performance_summary():
    """Get GPU performance summary"""
    optimizer = get_gpu_performance_optimizer()
    if not optimizer:
        raise HTTPException(status_code=503, detail="GPU performance optimizer not available")
        
    return optimizer.get_performance_summary()


@router.get("/performance/recommendations")
async def get_performance_recommendations():
    """Get performance optimization recommendations"""
    optimizer = get_gpu_performance_optimizer()
    if not optimizer:
        raise HTTPException(status_code=503, detail="GPU performance optimizer not available")
        
    return {
        "recommendations": optimizer.get_optimization_recommendations()
    }


@router.post("/warmup/{device_id}")
async def warmup_gpu(device_id: int):
    """Warm up a specific GPU device"""
    optimizer = get_gpu_performance_optimizer()
    if not optimizer:
        raise HTTPException(status_code=503, detail="GPU performance optimizer not available")
        
    success = optimizer.warmup_gpu(device_id)
    return {
        "device_id": device_id,
        "success": success,
        "message": "GPU warmup completed" if success else "GPU warmup failed"
    }


@router.post("/availability/mark-unavailable/{device_id}")
async def mark_device_unavailable(device_id: int, reason: str = "Manual marking"):
    """Mark a GPU device as unavailable"""
    availability_manager = get_gpu_availability_manager()
    if not availability_manager:
        raise HTTPException(status_code=503, detail="GPU availability manager not available")
        
    availability_manager.mark_device_unavailable(device_id, reason)
    return {
        "device_id": device_id,
        "status": "unavailable",
        "reason": reason
    }


@router.post("/availability/mark-available/{device_id}")
async def mark_device_available(device_id: int):
    """Mark a GPU device as available"""
    availability_manager = get_gpu_availability_manager()
    if not availability_manager:
        raise HTTPException(status_code=503, detail="GPU availability manager not available")
        
    availability_manager.mark_device_available(device_id)
    return {
        "device_id": device_id,
        "status": "available"
    }


@router.get("/health")
async def get_gpu_health():
    """Get overall GPU health status"""
    gpu_monitor = get_gpu_monitor()
    error_handler = get_gpu_error_handler()
    availability_manager = get_gpu_availability_manager()
    
    if not gpu_monitor:
        return {
            "status": "unhealthy",
            "reason": "GPU monitoring not available"
        }
        
    available_gpus = gpu_monitor.get_available_gpus()
    if not available_gpus:
        return {
            "status": "unhealthy",
            "reason": "No GPUs available"
        }
        
    # Check for recent errors
    recent_errors = 0
    if error_handler:
        error_summary = error_handler.get_error_summary()
        recent_errors = error_summary.get("total_errors_last_hour", 0)
        
    # Check device health
    unhealthy_devices = []
    if error_handler and availability_manager:
        for gpu_info in available_gpus:
            if not error_handler.is_device_healthy(gpu_info.device_id):
                unhealthy_devices.append(gpu_info.device_id)
                
    if unhealthy_devices:
        return {
            "status": "degraded",
            "reason": f"Unhealthy devices: {unhealthy_devices}",
            "unhealthy_devices": unhealthy_devices,
            "recent_errors": recent_errors
        }
        
    return {
        "status": "healthy",
        "available_gpus": len(available_gpus),
        "recent_errors": recent_errors
    }
