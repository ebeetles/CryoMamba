from fastapi import APIRouter, HTTPException
from app.models.job import JobRecord, JobState
from datetime import datetime
import logging
import uuid
import asyncio
from app.routes.websocket import broadcast_job_update, start_fake_preview_streaming

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory job storage for dummy implementation
jobs_db = {}
# Track active preview streaming tasks
preview_tasks = {}

@router.post("/jobs")
async def create_job():
    """
    Create a new job
    Returns mock job_id for dummy implementation
    """
    job_id = str(uuid.uuid4())
    now = datetime.now()
    
    job_record = JobRecord(
        job_id=job_id,
        state=JobState.PENDING,
        params={"dummy": True},
        created_at=now,
        updated_at=now
    )
    
    jobs_db[job_id] = job_record
    
    # Broadcast job creation
    await broadcast_job_update(job_id, "job_created", {
        "job_id": job_id,
        "state": job_record.state,
        "message": "Job created successfully"
    })
    
    # Start fake preview streaming for testing
    # Using a default volume shape for testing
    volume_shape = (64, 64, 64)  # Default test volume shape
    preview_task = asyncio.create_task(
        start_fake_preview_streaming(job_id, volume_shape, frequency=1.0)
    )
    preview_tasks[job_id] = preview_task
    
    # Update job state to running
    job_record.state = JobState.RUNNING
    job_record.updated_at = datetime.now()
    
    await broadcast_job_update(job_id, "job_started", {
        "job_id": job_id,
        "state": job_record.state,
        "message": "Job started with fake preview streaming"
    })
    
    logger.info(f"Created dummy job: {job_id}")
    
    return {
        "job_id": job_id,
        "status": "created",
        "message": "Dummy job created successfully with fake preview streaming"
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
    
    return {
        "job_id": job.job_id,
        "state": job.state,
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
        "message": "Job cancellation not yet implemented"
    }
