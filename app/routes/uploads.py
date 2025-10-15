import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field

from app.config import settings
from app.models import UploadSession

logger = logging.getLogger(__name__)
router = APIRouter()


# In-memory stores for sessions and files for this story implementation
UPLOAD_SESSIONS = {}


class InitUploadRequest(BaseModel):
    filename: str
    total_size_bytes: int
    chunk_size_bytes: int
    checksum_alg: Optional[str] = None
    expected_checksum: Optional[str] = None
    upload_id: Optional[str] = None  # to resume an existing session


class InitUploadResponse(BaseModel):
    upload_id: str
    total_parts: int
    chunk_size_bytes: int
    resume_parts: List[int] = Field(default_factory=list)
    received_parts: List[int] = Field(default_factory=list)
    completed: bool = False


def _persist_session(session: UploadSession, base_dir: str) -> None:
    out_dir = os.path.join(base_dir, session.upload_id)
    os.makedirs(out_dir, exist_ok=True)
    session_path = os.path.join(out_dir, "session.json")
    with open(session_path, "w") as f:
        json.dump(session.model_dump(), f, default=str)


def _load_session(upload_id: str, base_dir: str) -> Optional[UploadSession]:
    session_path = os.path.join(base_dir, upload_id, "session.json")
    if not os.path.exists(session_path):
        return None
    with open(session_path, "r") as f:
        data = json.load(f)
    # Convert datetime fields back
    data["created_at"] = datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
    data["updated_at"] = datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"]
    return UploadSession(**data)


def _scan_received_parts(upload_id: str, base_dir: str) -> List[int]:
    out_dir = os.path.join(base_dir, upload_id)
    if not os.path.isdir(out_dir):
        return []
    parts = []
    for name in os.listdir(out_dir):
        if name.startswith("part_"):
            try:
                idx = int(name.split("_")[1])
                parts.append(idx)
            except Exception:
                continue
    return sorted(parts)


@router.post("/uploads/init", response_model=InitUploadResponse)
async def init_upload(req: InitUploadRequest):
    if req.chunk_size_bytes <= 0:
        raise HTTPException(status_code=400, detail="chunk_size_bytes must be > 0")
    if req.total_size_bytes <= 0:
        raise HTTPException(status_code=400, detail="total_size_bytes must be > 0")
    if req.total_size_bytes > settings.max_upload_size_gb * 1024 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    if req.chunk_size_bytes > settings.max_chunk_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Chunk size exceeds configured maximum")

    total_parts = (req.total_size_bytes + req.chunk_size_bytes - 1) // req.chunk_size_bytes
    base_dir = settings.upload_base_dir

    # Resume existing session if provided and exists
    if req.upload_id:
        existing = _load_session(req.upload_id, base_dir)
        if existing:
            received = _scan_received_parts(req.upload_id, base_dir)
            return InitUploadResponse(
                upload_id=req.upload_id,
                total_parts=existing.total_parts,
                chunk_size_bytes=existing.chunk_size_bytes,
                resume_parts=[i for i in range(existing.total_parts) if i not in received],
                received_parts=received,
                completed=existing.completed,
            )

    upload_id = str(uuid.uuid4())

    now = datetime.now()
    session = UploadSession(
        upload_id=upload_id,
        filename=req.filename,
        total_size_bytes=req.total_size_bytes,
        chunk_size_bytes=req.chunk_size_bytes,
        total_parts=total_parts,
        created_at=now,
        updated_at=now,
        checksum_alg=req.checksum_alg,
        expected_checksum=req.expected_checksum,
    )

    # prepare storage dir and persist session
    os.makedirs(os.path.join(base_dir, upload_id), exist_ok=True)
    _persist_session(session, base_dir)

    UPLOAD_SESSIONS[upload_id] = session
    logger.info(f"Created upload session {upload_id} for {req.filename}")

    return InitUploadResponse(
        upload_id=upload_id,
        total_parts=total_parts,
        chunk_size_bytes=req.chunk_size_bytes,
        resume_parts=list(range(total_parts)),
        received_parts=[],
        completed=False,
    )


@router.put("/uploads/{upload_id}/part/{index}")
async def upload_part(upload_id: str, index: int, content: UploadFile = File(...)):
    session = UPLOAD_SESSIONS.get(upload_id)
    base_dir = settings.upload_base_dir
    if not session:
        # try load from disk (server restart)
        session = _load_session(upload_id, base_dir)
        if session:
            UPLOAD_SESSIONS[upload_id] = session
    if not session:
        raise HTTPException(status_code=404, detail="Upload session not found")
    if index < 0 or index >= session.total_parts:
        raise HTTPException(status_code=400, detail="Invalid part index")

    # write to disk
    part_path = os.path.join(base_dir, upload_id, f"part_{index:06d}")

    try:
        with open(part_path, "wb") as f:
            while True:
                chunk = await content.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        await content.close()

    session.mark_part_received(index)
    _persist_session(session, base_dir)
    logger.info(f"Received part {index} for {upload_id}")
    return {"status": "ok", "received_part": index}


class CompleteUploadRequest(BaseModel):
    checksum: Optional[str] = None


@router.post("/uploads/{upload_id}/complete")
async def complete_upload(upload_id: str, req: CompleteUploadRequest):
    base_dir = settings.upload_base_dir
    session = UPLOAD_SESSIONS.get(upload_id)
    if not session:
        session = _load_session(upload_id, base_dir)
        if session:
            UPLOAD_SESSIONS[upload_id] = session
    if not session:
        raise HTTPException(status_code=404, detail="Upload session not found")

    if not session.is_complete():
        raise HTTPException(status_code=400, detail="Not all parts uploaded")

    out_dir = os.path.join(base_dir, upload_id)
    assembled_path = os.path.join(out_dir, session.filename)

    # assemble
    with open(assembled_path, "wb") as out_f:
        for i in range(session.total_parts):
            part_path = os.path.join(out_dir, f"part_{i:06d}")
            if not os.path.exists(part_path):
                raise HTTPException(status_code=500, detail=f"Missing part {i}")
            with open(part_path, "rb") as pf:
                while True:
                    chunk = pf.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)

    # checksum validation
    if session.checksum_alg:
        alg = session.checksum_alg.lower()
        if alg not in ("sha256", "md5"):
            raise HTTPException(status_code=400, detail="Unsupported checksum algorithm")
        h = hashlib.sha256() if alg == "sha256" else hashlib.md5()
        with open(assembled_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        actual = h.hexdigest()
        expected = req.checksum or session.expected_checksum
        if expected and actual != expected:
            raise HTTPException(status_code=400, detail="Checksum mismatch")

    # mark complete
    session.completed = True
    session.updated_at = datetime.now()
    _persist_session(session, base_dir)

    # cleanup parts
    for i in range(session.total_parts):
        part_path = os.path.join(out_dir, f"part_{i:06d}")
        try:
            os.remove(part_path)
        except FileNotFoundError:
            pass

    return {
        "status": "completed",
        "upload_id": upload_id,
        "filename": session.filename,
        "path": assembled_path,
    }


@router.get("/uploads/{upload_id}/status")
async def upload_status(upload_id: str):
    base_dir = settings.upload_base_dir
    session = UPLOAD_SESSIONS.get(upload_id) or _load_session(upload_id, base_dir)
    if not session:
        raise HTTPException(status_code=404, detail="Upload session not found")
    received = _scan_received_parts(upload_id, base_dir)
    progress = len(received) / session.total_parts if session.total_parts else 0.0
    return {
        "upload_id": upload_id,
        "filename": session.filename,
        "received_parts": received,
        "total_parts": session.total_parts,
        "progress": progress,
        "completed": session.completed,
    }


@router.delete("/uploads/{upload_id}")
async def cancel_upload(upload_id: str):
    base_dir = settings.upload_base_dir
    # remove directory
    out_dir = os.path.join(base_dir, upload_id)
    if not os.path.isdir(out_dir):
        raise HTTPException(status_code=404, detail="Upload session not found")
    # best-effort cleanup
    for name in os.listdir(out_dir):
        try:
            os.remove(os.path.join(out_dir, name))
        except IsADirectoryError:
            pass
        except FileNotFoundError:
            pass
    try:
        os.rmdir(out_dir)
    except Exception:
        pass
    # drop from memory
    UPLOAD_SESSIONS.pop(upload_id, None)
    return {"status": "cancelled", "upload_id": upload_id}


