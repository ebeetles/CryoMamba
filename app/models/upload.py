from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime


class UploadPartStatus(str, Enum):
    PENDING = "pending"
    RECEIVED = "received"
    VERIFIED = "verified"


class UploadSession(BaseModel):
    """Represents a chunked upload session and its progression state."""

    model_config = ConfigDict(use_enum_values=True)

    upload_id: str
    filename: str
    total_size_bytes: int
    chunk_size_bytes: int
    total_parts: int
    created_at: datetime
    updated_at: datetime
    completed: bool = False
    checksum_alg: Optional[str] = None  # e.g., sha256
    expected_checksum: Optional[str] = None

    # part index (0-based) -> status
    parts: Dict[int, UploadPartStatus] = Field(default_factory=dict)

    def mark_part_received(self, index: int) -> None:
        self.parts[index] = UploadPartStatus.RECEIVED
        self.updated_at = datetime.now()

    def is_complete(self) -> bool:
        if self.total_parts == 0:
            return False
        return len([i for i, s in self.parts.items() if s in (UploadPartStatus.RECEIVED, UploadPartStatus.VERIFIED)]) == self.total_parts


class FileRecord(BaseModel):
    """Represents a finalized file assembled from chunks."""

    file_id: str
    upload_id: str
    filename: str
    size_bytes: int
    checksum_alg: Optional[str] = None
    checksum: Optional[str] = None
    path: str
    created_at: datetime


