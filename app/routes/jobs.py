from fastapi import APIRouter, HTTPException
from app.models.job import JobRecord, JobState
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory job storage for dummy implementation
jobs_db = {}

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
    logger.info(f"Created dummy job: {job_id}")
    
    return {
        "job_id": job_id,
        "status": "created",
        "message": "Dummy job created successfully"
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
    
    logger.info(f"Cancelled job: {job_id}")
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job cancellation not yet implemented"
    }
