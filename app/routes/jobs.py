from fastapi import APIRouter, HTTPException
from app.models.job import JobRecord, JobState
from datetime import datetime
import logging
import uuid
import asyncio
from typing import Optional, Dict, Any
from pydantic import BaseModel
from app.routes.websocket import broadcast_job_update, start_fake_preview_streaming
import app.services.orchestrator as orchestrator_service

class JobCreateRequest(BaseModel):
    """Request model for creating a job."""
    volume_shape: Optional[list] = None
    volume_dtype: Optional[str] = None
    has_volume: Optional[bool] = False
    params: Optional[Dict[str, Any]] = None

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory job storage for dummy implementation
jobs_db = {}
# Track active preview streaming tasks
preview_tasks = {}
# Background task manager
background_tasks = set()

@router.post("/jobs")
async def create_job(job_request: JobCreateRequest = None):
    """
    Create a new job
    Returns mock job_id for dummy implementation
    """
    job_id = str(uuid.uuid4())
    now = datetime.now()
    
    # Extract volume parameters if provided
    volume_shape = (64, 64, 64)  # Default test volume shape
    if job_request and job_request.has_volume and job_request.volume_shape:
        volume_shape = tuple(job_request.volume_shape)
        logger.info(f"Using provided volume shape: {volume_shape}")
    else:
        logger.info(f"Using default volume shape: {volume_shape}")
    
    # Prepare job parameters
    job_params = {"dummy": True}
    if job_request:
        if job_request.params:
            job_params.update(job_request.params)
        if job_request.has_volume:
            job_params.update({
                "volume_shape": job_request.volume_shape,
                "volume_dtype": job_request.volume_dtype,
                "has_volume": True
            })
    
    job_record = JobRecord(
        job_id=job_id,
        state=JobState.QUEUED,
        params=job_params,
        created_at=now,
        updated_at=now
    )
    
    jobs_db[job_id] = job_record
    # Submit to orchestrator queue
    if orchestrator_service.orchestrator is not None:
        await orchestrator_service.orchestrator.submit(job_record)
    
    # Broadcast job creation
    await broadcast_job_update(job_id, "job_created", {
        "job_id": job_id,
        "state": job_record.state,
        "message": "Job created successfully"
    })
    
    # Preview streaming now handled by orchestrator during processing as needed
    logger.info(f"Created job: {job_id} with volume shape: {volume_shape}")
    
    return {
        "job_id": job_id,
        "status": "created",
        "message": f"Dummy job created successfully with fake preview streaming for volume {volume_shape}"
    }

@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """
    Get job status and details
    Returns mock job information
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    logger.info(f"Retrieved job: {job_id}")
    
    # Map internal 'queued' state to externally expected 'pending'
    external_state = "pending" if str(job.state) == "queued" else job.state
    return {
        "job_id": job.job_id,
        "state": external_state,
        "params": job.params,
        "artifacts": job.artifacts,
        "errors": job.errors,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat()
    }

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a job
    For future implementation - currently returns mock response
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    if orchestrator_service.orchestrator is not None:
        await orchestrator_service.orchestrator.cancel(job_id)
    else:
        job.state = JobState.CANCELLED
        job.updated_at = datetime.now()
    
    # Stop preview streaming if active
    if job_id in preview_tasks:
        preview_task = preview_tasks[job_id]
        preview_task.cancel()
        try:
            await preview_task
        except asyncio.CancelledError:
            pass
        del preview_tasks[job_id]
    
    # Broadcast job cancellation
    await broadcast_job_update(job_id, "job_cancelled", {
        "job_id": job_id,
        "state": job.state,
        "message": "Job cancelled and preview streaming stopped"
    })
    
    logger.info(f"Cancelled job: {job_id}")
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job cancelled"
    }


@router.get("/jobs")
async def list_jobs():
    """List jobs with basic filtering in the future."""
    res = []
    if orchestrator_service.orchestrator is not None:
        for job in orchestrator_service.orchestrator.list():
            res.append({
                "job_id": job.job_id,
                "state": ("pending" if str(job.state) == "queued" else job.state),
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
            })
    else:
        for job in jobs_db.values():
            res.append({
                "job_id": job.job_id,
                "state": ("pending" if str(job.state) == "queued" else job.state),
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
            })
    return {"jobs": res}
