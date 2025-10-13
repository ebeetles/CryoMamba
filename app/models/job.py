from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from enum import Enum
from datetime import datetime

class JobState(str, Enum):
    """Job state enumeration"""
    PENDING = "pending"
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
    created_at: datetime
    updated_at: datetime
