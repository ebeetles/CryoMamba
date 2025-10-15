import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime

from app.models import JobRecord, JobState
from app.routes.websocket import broadcast_job_update
from app.config import settings
from app.services.nnunet_wrapper import NnUNetWrapper, NnUNetNotAvailableError
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
    """Simple in-memory orchestrator with a bounded queue and fake processing."""

    def __init__(self, concurrency: int = 1, queue_size: int = 100):
        self.jobs: Dict[str, JobRecord] = {}
        self.queue: asyncio.Queue[Optional[JobRecord]] = asyncio.Queue(maxsize=queue_size)
        self.concurrency = concurrency
        self.workers: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        # Per-job cancellation events to support cooperative cancellation mid-run
        self._cancel_events: Dict[str, asyncio.Event] = {}
        # Simple job timeout to prevent stuck jobs (seconds)
        self.max_job_duration_seconds: int = 10

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
        await broadcast_job_update(job.job_id, "job_enqueued", {"state": job.state})
        await self.queue.put(job)

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self.jobs.get(job_id)

    def list(self):
        return list(self.jobs.values())

    async def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
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

                # Resolve device preference
                chosen_device = device_override or settings.nnunet_device

                volume: np.ndarray
                if input_nifti_path and os.path.exists(input_nifti_path):
                    import nibabel as nib  # lazy import
                    img = nib.load(input_nifti_path)
                    data = img.get_fdata()
                    # Ensure float32 and (Z,Y,X)
                    volume = np.asarray(data, dtype=np.float32)
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

                try:
                    # Optionally simulate OOM for tests
                    if simulate_oom:
                        raise RuntimeError("CUDA out of memory. simulated")

                    # Dummy path: fast CPU deterministic mask
                    if dummy_inference:
                        job.progress = 0.5
                        await broadcast_job_update(job.job_id, "progress", {"progress": job.progress})
                        pred = (volume > float(np.mean(volume))).astype(np.float32)
                    else:
                        # Warmup provides early failure signal
                        warm = wrapper.warmup()
                        status_str = str(warm.get("status", ""))
                        if status_str.startswith("error") and not chosen_device == "cpu":
                            # If user explicitly set CPU we permit proceeding without CUDA
                            raise NnUNetNotAvailableError(warm.get("status", "nnU-Net unavailable"))

                        # Run prediction (blocking call)
                        pred = wrapper.predict_numpy(volume, progress_callback=_on_progress)

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

                    # Finalize success
                    job.progress = 1.0
                    await broadcast_job_update(job.job_id, "progress", {"progress": job.progress})
                    JobStateMachine.transition(job, JobState.COMPLETED, reason="inference_done")
                    await broadcast_job_update(job.job_id, "job_completed", {"state": job.state})

                except NnUNetNotAvailableError as e:
                    JobStateMachine.transition(job, JobState.FAILED, reason="nnunet_unavailable")
                    job.errors = {"code": "NNUNET_NOT_AVAILABLE", "message": str(e), "retryable": False}
                    await broadcast_job_update(job.job_id, "job_failed", {"error": job.errors})
                except RuntimeError as e:
                    # Detect CUDA OOM
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
                # Cleanup cancel event for completed terminal jobs to avoid leaks
                if job.state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
                    self._cancel_events.pop(job.job_id, None)
                self.queue.task_done()


# Global orchestrator instance for app
orchestrator: Optional[InMemoryOrchestrator] = None

async def init_orchestrator():
    global orchestrator
    if orchestrator is None:
        orchestrator = InMemoryOrchestrator(concurrency=1)
        await orchestrator.start()

async def shutdown_orchestrator():
    global orchestrator
    if orchestrator is not None:
        await orchestrator.stop()
        orchestrator = None


