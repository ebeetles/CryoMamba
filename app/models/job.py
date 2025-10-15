from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

class JobState(str, Enum):
    """Job state enumeration"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobRecord(BaseModel):
    """Job record schema"""
    model_config = ConfigDict(use_enum_values=True)
    
    job_id: str
    state: JobState
    params: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, str]] = None
    errors: Optional[Dict[str, str]] = None
    progress: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    history: Optional[List[Dict[str, Any]]]=None
