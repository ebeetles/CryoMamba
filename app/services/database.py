"""
Database service for CryoMamba job persistence.
Uses SQLite for simplicity and reliability.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import os

from app.models.job import JobRecord, JobState

logger = logging.getLogger(__name__)

class DatabaseService:
    """SQLite-based database service for job persistence."""
    
    def __init__(self, db_path: str = "cryomamba.db"):
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize the database with required tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    progress REAL DEFAULT 0.0,
                    params TEXT,  -- JSON string
                    artifacts TEXT,  -- JSON string
                    errors TEXT,  -- JSON string
                    history TEXT,  -- JSON string
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    duration_ms INTEGER
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def create_job(self, job: JobRecord) -> JobRecord:
        """Create a new job in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO jobs (
                    job_id, state, progress, params, artifacts, errors, history,
                    created_at, updated_at, started_at, completed_at, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id,
                job.state.value if hasattr(job.state, 'value') else str(job.state),
                job.progress,
                json.dumps(job.params) if job.params else None,
                json.dumps(job.artifacts) if job.artifacts else None,
                json.dumps(job.errors) if job.errors else None,
                json.dumps(job.history) if job.history else None,
                job.created_at.isoformat(),
                job.updated_at.isoformat(),
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.duration_ms
            ))
            
            conn.commit()
            logger.info(f"Created job {job.job_id} in database")
            return job
    
    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Get a job by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT job_id, state, progress, params, artifacts, errors, history,
                       created_at, updated_at, started_at, completed_at, duration_ms
                FROM jobs WHERE job_id = ?
            """, (job_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return self._row_to_job_record(row)
    
    def update_job(self, job: JobRecord) -> JobRecord:
        """Update an existing job."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE jobs SET
                    state = ?, progress = ?, params = ?, artifacts = ?, errors = ?,
                    history = ?, updated_at = ?, started_at = ?, completed_at = ?,
                    duration_ms = ?
                WHERE job_id = ?
            """, (
                job.state.value if hasattr(job.state, 'value') else str(job.state),
                job.progress,
                json.dumps(job.params) if job.params else None,
                json.dumps(job.artifacts) if job.artifacts else None,
                json.dumps(job.errors) if job.errors else None,
                json.dumps(job.history) if job.history else None,
                job.updated_at.isoformat(),
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.duration_ms,
                job.job_id
            ))
            
            conn.commit()
            logger.info(f"Updated job {job.job_id} in database")
            return job
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            deleted = cursor.rowcount > 0
            
            conn.commit()
            if deleted:
                logger.info(f"Deleted job {job_id} from database")
            return deleted
    
    def list_jobs(self, limit: Optional[int] = None, offset: int = 0) -> List[JobRecord]:
        """List jobs with optional pagination."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT job_id, state, progress, params, artifacts, errors, history,
                       created_at, updated_at, started_at, completed_at, duration_ms
                FROM jobs
                ORDER BY created_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit} OFFSET {offset}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            return [self._row_to_job_record(row) for row in rows]
    
    def list_jobs_by_state(self, state: JobState) -> List[JobRecord]:
        """List jobs by state."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT job_id, state, progress, params, artifacts, errors, history,
                       created_at, updated_at, started_at, completed_at, duration_ms
                FROM jobs WHERE state = ?
                ORDER BY created_at DESC
            """, (state.value,))
            
            rows = cursor.fetchall()
            return [self._row_to_job_record(row) for row in rows]
    
    def _row_to_job_record(self, row) -> JobRecord:
        """Convert database row to JobRecord object."""
        return JobRecord(
            job_id=row[0],
            state=JobState(row[1]),
            progress=row[2] or 0.0,
            params=json.loads(row[3]) if row[3] else None,
            artifacts=json.loads(row[4]) if row[4] else None,
            errors=json.loads(row[5]) if row[5] else None,
            history=json.loads(row[6]) if row[6] else None,
            created_at=datetime.fromisoformat(row[7]),
            updated_at=datetime.fromisoformat(row[8]),
            started_at=datetime.fromisoformat(row[9]) if row[9] else None,
            completed_at=datetime.fromisoformat(row[10]) if row[10] else None,
            duration_ms=row[11]
        )
    
    def cleanup_old_jobs(self, days_old: int = 30) -> int:
        """Clean up jobs older than specified days."""
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_old)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM jobs 
                WHERE created_at < ? AND state IN ('completed', 'failed', 'cancelled')
            """, (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old jobs")
            
            return deleted_count

# Global database service instance
db_service: Optional[DatabaseService] = None

def get_db_service() -> DatabaseService:
    """Get the global database service instance."""
    global db_service
    if db_service is None:
        db_service = DatabaseService()
    return db_service

def init_database(db_path: str = "cryomamba.db"):
    """Initialize the database service."""
    global db_service
    db_service = DatabaseService(db_path)
    logger.info("Database service initialized")

def cleanup_database():
    """Cleanup database resources."""
    global db_service
    db_service = None
