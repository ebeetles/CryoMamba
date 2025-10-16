import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime

from app.models import JobRecord, JobState
from app.routes.websocket import broadcast_job_update
from app.config import settings
from app.services.nnunet_wrapper import NnUNetWrapper, NnUNetNotAvailableError
from app.services.preview_streamer import preview_manager
from app.services.gpu_monitor import get_gpu_monitor
from app.services.gpu_scheduler import get_gpu_scheduler, JobPriority
from app.services.gpu_memory_manager import get_gpu_memory_manager
from app.services.gpu_error_handler import get_gpu_error_handler, get_gpu_availability_manager, GPUError, GPUErrorType
import numpy as np
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class JobStateMachine:
    """Validates and applies job state transitions with audit history."""

    VALID_TRANSITIONS = {
        JobState.QUEUED: {JobState.RUNNING, JobState.CANCELLED},
        JobState.RUNNING: {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED},
        JobState.COMPLETED: set(),
        JobState.FAILED: set(),
        JobState.CANCELLED: set(),
    }

    @staticmethod
    def transition(job: JobRecord, new_state: JobState, reason: Optional[str] = None):
        if new_state not in JobStateMachine.VALID_TRANSITIONS.get(job.state, set()):
            raise ValueError(f"Invalid transition: {job.state} -> {new_state}")

        now = datetime.now()
        if job.history is None:
            job.history = []
        job.history.append({
            "from": job.state,
            "to": new_state,
            "timestamp": now.isoformat(),
            "reason": reason or ""
        })

        job.state = new_state
        if new_state == JobState.RUNNING:
            job.started_at = now
        if new_state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
            job.completed_at = now
            if job.started_at:
                job.duration_ms = int((now - job.started_at).total_seconds() * 1000)
        job.updated_at = now


class InMemoryOrchestrator:
    """GPU-aware orchestrator with resource management and error handling."""

    def __init__(self, concurrency: int = 1, queue_size: int = 100):
        self.jobs: Dict[str, JobRecord] = {}
        self.queue: asyncio.Queue[Optional[JobRecord]] = asyncio.Queue(maxsize=queue_size)
        self.concurrency = concurrency
        self.workers: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        # Per-job cancellation events to support cooperative cancellation mid-run
        self._cancel_events: Dict[str, asyncio.Event] = {}
        # Simple job timeout to prevent stuck jobs (seconds)
        self.max_job_duration_seconds: int = 1800  # 30 minutes for real inference
        # GPU resource management
        self.estimated_memory_mb: int = 2048  # Default 2GB per job

    async def start(self):
        logger.info("Starting orchestrator with %s workers", self.concurrency)
        self._stop_event.clear()
        for i in range(self.concurrency):
            task = asyncio.create_task(self._worker_loop(i))
            self.workers.append(task)

    async def stop(self):
        logger.info("Stopping orchestrator")
        self._stop_event.set()
        for _ in self.workers:
            await self.queue.put(None)
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

    async def submit(self, job: JobRecord):
        self.jobs[job.job_id] = job
        # Ensure a fresh cancel event exists for the job
        self._cancel_events[job.job_id] = asyncio.Event()
        
        # Try to submit to GPU scheduler first
        gpu_scheduler = get_gpu_scheduler()
        if gpu_scheduler:
            # Estimate memory requirements based on job parameters
            params = job.params or {}
            volume_shape = params.get("volume_shape", (64, 64, 64))
            estimated_memory_mb = self._estimate_memory_requirements(volume_shape)
            
            # Submit to GPU scheduler
            success = await gpu_scheduler.submit_job(
                job=job,
                priority=JobPriority.NORMAL,
                estimated_memory_mb=estimated_memory_mb,
                estimated_duration_seconds=300  # 5 minutes default
            )
            
            if success:
                await broadcast_job_update(job.job_id, "job_enqueued", {"state": job.state})
                return
        
        # Fallback to original queue if GPU scheduler unavailable
        await broadcast_job_update(job.job_id, "job_enqueued", {"state": job.state})
        await self.queue.put(job)

    def _estimate_memory_requirements(self, volume_shape: tuple) -> int:
        """Estimate GPU memory requirements for a volume"""
        # Rough estimation: 4 bytes per voxel for input + output + intermediate tensors
        # Add some overhead for model weights and activations
        voxels = volume_shape[0] * volume_shape[1] * volume_shape[2]
        memory_bytes = voxels * 4 * 3  # Input, output, intermediate
        memory_mb = memory_bytes // (1024 * 1024)
        
        # Add model overhead (rough estimate)
        model_overhead_mb = 1024  # 1GB for model weights
        
        return memory_mb + model_overhead_mb

    async def _resolve_device(self, device_override: Optional[str], job: JobRecord) -> str:
        """Resolve the best available device for a job"""
        availability_manager = get_gpu_availability_manager()
        
        if device_override:
            if device_override == "cpu":
                return "cpu"
            elif device_override.startswith("cuda:"):
                device_id = int(device_override.split(":")[1])
                if availability_manager and availability_manager.is_device_available(device_id):
                    return device_override
                else:
                    logger.warning("Requested GPU %d not available, falling back to CPU", device_id)
                    return "cpu"
        
        # Check for available GPUs
        if availability_manager:
            available_devices = availability_manager.get_available_devices()
            if available_devices:
                # Use first available GPU
                return f"cuda:{available_devices[0]}"
            elif availability_manager.should_fallback_to_cpu():
                logger.info("No GPUs available, falling back to CPU")
                return "cpu"
        
        # Default fallback
        return settings.nnunet_device or "cpu"

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self.jobs.get(job_id)

    def list(self):
        return list(self.jobs.values())

    async def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
            
        # Try GPU scheduler first
        gpu_scheduler = get_gpu_scheduler()
        if gpu_scheduler:
            success = await gpu_scheduler.cancel_job(job_id)
            if success:
                # Clean up GPU resources
                gpu_memory_manager = get_gpu_memory_manager()
                if gpu_memory_manager:
                    gpu_memory_manager.deallocate_job_memory(job_id)
                return True
        
        # Signal cancellation for worker loop regardless of current state
        if job_id not in self._cancel_events:
            self._cancel_events[job_id] = asyncio.Event()
        self._cancel_events[job_id].set()

        # If job already terminal, consider it cancelled/satisfied
        if job.state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
            return True

        # If still queued, move directly to cancelled and broadcast
        if job.state == JobState.QUEUED:
            JobStateMachine.transition(job, JobState.CANCELLED, reason="user_cancel")
            await broadcast_job_update(job_id, "job_cancelled", {"state": job.state})
            return True

        # If running, worker loop will observe the cancel event and finalize state
        return True

    async def _worker_loop(self, worker_idx: int):
        while not self._stop_event.is_set():
            job = await self.queue.get()
            if job is None:
                break
            try:
                # If job was cancelled while queued, skip processing
                cancel_event = self._cancel_events.get(job.job_id)
                if cancel_event is not None and cancel_event.is_set():
                    if job.state not in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
                        JobStateMachine.transition(job, JobState.CANCELLED, reason="user_cancel_before_start")
                        await broadcast_job_update(job.job_id, "job_cancelled", {"state": job.state})
                    continue

                JobStateMachine.transition(job, JobState.RUNNING, reason=f"worker_{worker_idx}")
                await broadcast_job_update(job.job_id, "job_started", {"state": job.state})

                # Cooperative cancellation check before heavy work
                cancel_event = self._cancel_events.get(job.job_id)
                if cancel_event is not None and cancel_event.is_set():
                    JobStateMachine.transition(job, JobState.CANCELLED, reason="user_cancel_before_infer")
                    await broadcast_job_update(job.job_id, "job_cancelled", {"state": job.state})
                    continue

                # Prepare input
                params = job.params or {}
                input_nifti_path = params.get("input_nifti_path")
                device_override = params.get("device")
                dummy_inference = bool(params.get("dummy_inference", False))
                simulate_oom = bool(params.get("oom_simulate", False))

                # Resolve device preference with GPU availability check
                chosen_device = await self._resolve_device(device_override, job)

                volume: np.ndarray
                if input_nifti_path and os.path.exists(input_nifti_path):
                    import nibabel as nib  # lazy import
                    img = nib.load(input_nifti_path)
                    data = img.get_fdata()
                    # Ensure float32 and (Z,Y,X)
                    volume = np.asarray(data, dtype=np.float32)
                else:
                    # Try to load uploaded MRC file
                    upload_id = params.get("upload_id")
                    if upload_id:
                        # Load the uploaded MRC file - it's saved with the original filename
                        upload_dir = Path(settings.upload_base_dir) / upload_id
                        mrc_files = list(upload_dir.glob("*.mrc"))
                        if mrc_files:
                            mrc_path = mrc_files[0]  # Take the first .mrc file found
                            import mrcfile
                            with mrcfile.open(str(mrc_path)) as mrc:
                                data = mrc.data.copy()
                            volume = np.asarray(data, dtype=np.float32)
                            logger.info(f"Loaded uploaded MRC file: {mrc_path}")
                        else:
                            logger.warning(f"No MRC file found in upload directory: {upload_dir}")
                            # Fallback to small test volume
                            volume_shape = tuple(params.get("volume_shape", (16, 16, 16)))
                            volume = np.zeros(volume_shape, dtype=np.float32)
                    else:
                        # Fallback small volume to keep CPU tests fast
                        volume_shape = tuple(params.get("volume_shape", (16, 16, 16)))
                        volume = np.zeros(volume_shape, dtype=np.float32)

                wrapper = NnUNetWrapper(
                    model_dir=settings.nnunet_model_dir or "",
                    device=chosen_device,
                )

                def _on_progress(p: float):
                    job.progress = min(max(p, 0.0), 0.99)
                    # Fire-and-forget; no await inside external callback
                    asyncio.create_task(broadcast_job_update(job.job_id, "progress", {"progress": job.progress}))
                
                def _on_preview(mask: np.ndarray):
                    # Fire-and-forget preview update
                    preview_manager.update_mask(job.job_id, mask)

                try:
                    # Start preview streaming
                    streamer = await preview_manager.create_streamer(
                        job_id=job.job_id,
                        frequency=1.0,  # 1 Hz updates
                        target_size=(64, 64, 64)
                    )
                    await preview_manager.start_streaming(job.job_id, volume)
                    
                    # Optionally simulate OOM for tests
                    if simulate_oom:
                        raise RuntimeError("CUDA out of memory. simulated")

                    # Dummy path: fast CPU deterministic mask
                    if dummy_inference:
                        job.progress = 0.5
                        await broadcast_job_update(job.job_id, "progress", {"progress": job.progress})
                        pred = (volume > float(np.mean(volume))).astype(np.float32)
                        # Update preview with dummy result
                        preview_manager.update_mask(job.job_id, pred)
                    else:
                        # Use real nnU-Net inference with direct network access
                        logger.info("Starting real nnU-Net inference using direct network access")
                        
                        # Check if nnU-Net is ready
                        warm = wrapper.warmup()
                        status_str = str(warm.get("status", ""))
                        
                        if status_str != "ready":
                            raise NnUNetNotAvailableError(f"nnU-Net not ready: {status_str}")
                        
                        # Run real inference using direct network access
                        pred = wrapper.predict_numpy(
                            volume, 
                            progress_callback=_on_progress,
                            preview_callback=_on_preview
                        )

                    # Save artifact
                    artifacts_dir = Path(settings.upload_base_dir) / job.job_id
                    artifacts_dir.mkdir(parents=True, exist_ok=True)
                    out_path = artifacts_dir / "output_mask.nii.gz"
                    try:
                        import nibabel as nib
                        affine = np.eye(4, dtype=np.float32)
                        nib.save(nib.Nifti1Image(pred.astype(np.float32), affine), str(out_path))
                    except Exception:
                        # If nibabel is unavailable at runtime, skip artifact save
                        out_path = None  # type: ignore

                    # Update job artifacts
                    if out_path is not None:
                        if job.artifacts is None:
                            job.artifacts = {}
                        job.artifacts["mask_nifti"] = str(out_path)

                    # Send final preview and stop streaming
                    await preview_manager.send_final_preview(job.job_id, pred)
                    await preview_manager.stop_streaming(job.job_id)
                    
                    # Finalize success
                    job.progress = 1.0
                    await broadcast_job_update(job.job_id, "progress", {"progress": job.progress})
                    JobStateMachine.transition(job, JobState.COMPLETED, reason="inference_done")
                    await broadcast_job_update(job.job_id, "job_completed", {"state": job.state})

                except NnUNetNotAvailableError as e:
                    # Stop preview streaming on error
                    await preview_manager.stop_streaming(job.job_id)
                    JobStateMachine.transition(job, JobState.FAILED, reason="nnunet_unavailable")
                    job.errors = {"code": "NNUNET_NOT_AVAILABLE", "message": str(e), "retryable": False}
                    await broadcast_job_update(job.job_id, "job_failed", {"error": job.errors})
                except RuntimeError as e:
                    # Stop preview streaming on error
                    await preview_manager.stop_streaming(job.job_id)
                    
                    # Handle GPU errors with error handler
                    error_handler = get_gpu_error_handler()
                    if error_handler:
                        msg = str(e)
                        if "out of memory" in msg.lower():
                            # Extract device ID from chosen_device
                            device_id = 0  # Default
                            if chosen_device.startswith("cuda:"):
                                device_id = int(chosen_device.split(":")[1])
                            
                            gpu_error = GPUError(
                                GPUErrorType.CUDA_OOM,
                                device_id,
                                msg,
                                job.job_id
                            )
                            
                            # Attempt recovery
                            if error_handler.handle_error(gpu_error):
                                logger.info("GPU error recovery successful for job %s", job.job_id)
                                # Retry the job or continue with CPU
                                JobStateMachine.transition(job, JobState.FAILED, reason="cuda_oom_recovered")
                                job.errors = {
                                    "code": "CUDA_OOM_RECOVERED",
                                    "message": "CUDA out of memory, attempted recovery",
                                    "hint": "Job failed but GPU resources were recovered",
                                    "retryable": True,
                                }
                            else:
                                JobStateMachine.transition(job, JobState.FAILED, reason="cuda_oom")
                                job.errors = {
                                    "code": "CUDA_OOM",
                                    "message": "CUDA out of memory during inference",
                                    "hint": "Reduce volume size or adjust sliding window parameters",
                                    "retryable": True,
                                }
                            await broadcast_job_update(job.job_id, "job_failed", {"error": job.errors})
                        else:
                            # Other CUDA errors
                            device_id = 0
                            if chosen_device.startswith("cuda:"):
                                device_id = int(chosen_device.split(":")[1])
                            
                            gpu_error = GPUError(
                                GPUErrorType.CUDA_ERROR,
                                device_id,
                                msg,
                                job.job_id
                            )
                            error_handler.handle_error(gpu_error)
                            raise
                    else:
                        # Fallback to original error handling
                        msg = str(e)
                        if "out of memory" in msg.lower():
                            JobStateMachine.transition(job, JobState.FAILED, reason="cuda_oom")
                            job.errors = {
                                "code": "CUDA_OOM",
                                "message": "CUDA out of memory during inference",
                                "hint": "Reduce volume size or adjust sliding window parameters",
                                "retryable": True,
                            }
                            await broadcast_job_update(job.job_id, "job_failed", {"error": job.errors})
                        else:
                            raise
            except Exception as e:
                logger.exception("Job %s failed: %s", job.job_id, e)
                # Stop preview streaming on error
                await preview_manager.stop_streaming(job.job_id)
                # Categorize as processing error with envelope-like structure
                job.errors = {
                    "code": "PROCESSING_ERROR",
                    "message": str(e),
                    "hint": "Check server logs for details",
                    "retryable": False
                }
                try:
                    JobStateMachine.transition(job, JobState.FAILED, reason="exception")
                except Exception:
                    pass
                await broadcast_job_update(job.job_id, "job_failed", {"error": job.errors})
            finally:
                # Cleanup GPU resources for completed jobs
                if job.state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
                    # Clean up GPU scheduler resources
                    gpu_scheduler = get_gpu_scheduler()
                    if gpu_scheduler:
                        await gpu_scheduler.complete_job(job.job_id)
                    
                    # Clean up GPU memory
                    gpu_memory_manager = get_gpu_memory_manager()
                    if gpu_memory_manager:
                        gpu_memory_manager.deallocate_job_memory(job.job_id)
                    
                    # Cleanup cancel event for completed terminal jobs to avoid leaks
                    self._cancel_events.pop(job.job_id, None)
                self.queue.task_done()


# Global orchestrator instance for app
orchestrator: Optional[InMemoryOrchestrator] = None

async def init_orchestrator():
    global orchestrator
    if orchestrator is None:
        # Initialize GPU services first
        from app.services.gpu_monitor import init_gpu_monitor
        from app.services.gpu_scheduler import init_gpu_scheduler
        from app.services.gpu_memory_manager import init_gpu_memory_manager
        from app.services.gpu_error_handler import init_gpu_error_handler
        from app.services.gpu_performance_optimizer import init_gpu_performance_optimizer
        
        await init_gpu_monitor()
        await init_gpu_scheduler()
        await init_gpu_memory_manager()
        await init_gpu_error_handler()
        await init_gpu_performance_optimizer()
        
        orchestrator = InMemoryOrchestrator(concurrency=1)
        await orchestrator.start()

async def shutdown_orchestrator():
    global orchestrator
    if orchestrator is not None:
        await orchestrator.stop()
        orchestrator = None
        
        # Shutdown GPU services
        from app.services.gpu_monitor import shutdown_gpu_monitor
        from app.services.gpu_scheduler import shutdown_gpu_scheduler
        from app.services.gpu_memory_manager import shutdown_gpu_memory_manager
        from app.services.gpu_error_handler import shutdown_gpu_error_handler
        from app.services.gpu_performance_optimizer import shutdown_gpu_performance_optimizer
        
        await shutdown_gpu_monitor()
        await shutdown_gpu_scheduler()
        await shutdown_gpu_memory_manager()
        await shutdown_gpu_error_handler()
        await shutdown_gpu_performance_optimizer()


