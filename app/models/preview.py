"""
Preview message models for WebSocket streaming.
"""

from pydantic import BaseModel
from typing import List, Union
import base64
import numpy as np

class PreviewMessage(BaseModel):
    """Preview message schema for WebSocket streaming."""
    
    scale: List[float]  # Voxel scale [x, y, z]
    shape: List[int]    # Volume shape [z, y, x]
    dtype: str          # Data type (e.g., "uint8", "float32")
    payload: str        # Base64 encoded preview data
    
    @classmethod
    def from_numpy_array(cls, data: np.ndarray, scale: List[float] = None) -> 'PreviewMessage':
        """
        Create a PreviewMessage from a numpy array.
        
        Args:
            data: Numpy array containing preview data
            scale: Voxel scale (defaults to [1.0, 1.0, 1.0])
        
        Returns:
            PreviewMessage instance
        """
        if scale is None:
            scale = [1.0, 1.0, 1.0]
        
        # Encode data as base64
        data_bytes = data.tobytes()
        base64_payload = base64.b64encode(data_bytes).decode('utf-8')
        
        return cls(
            scale=scale,
            shape=list(data.shape),
            dtype=str(data.dtype),
            payload=base64_payload
        )
    
    def to_numpy_array(self) -> np.ndarray:
        """
        Convert the preview message back to a numpy array.
        
        Returns:
            Numpy array containing the preview data
        """
        # Decode base64 payload
        data_bytes = base64.b64decode(self.payload.encode('utf-8'))
        
        # Reconstruct numpy array
        data = np.frombuffer(data_bytes, dtype=self.dtype)
        data = data.reshape(self.shape)
        
        return data
    
    def validate_dimensions(self, expected_shape: tuple) -> bool:
        """
        Validate that the preview data matches expected dimensions.
        
        Args:
            expected_shape: Expected volume shape (z, y, x)
        
        Returns:
            True if dimensions match, False otherwise
        """
        return tuple(self.shape) == expected_shape

class WebSocketMessage(BaseModel):
    """Base WebSocket message schema."""
    
    type: str           # Message type (preview, progress, completed, error)
    job_id: str         # Job ID
    timestamp: str      # ISO timestamp
    data: dict          # Message-specific data

class PreviewWebSocketMessage(WebSocketMessage):
    """WebSocket message containing preview data."""
    
    type: str = "preview"
    data: PreviewMessage

class ProgressWebSocketMessage(WebSocketMessage):
    """WebSocket message containing progress updates."""
    
    type: str = "progress"
    data: dict  # Progress data (percentage, stage, etc.)

class CompletedWebSocketMessage(WebSocketMessage):
    """WebSocket message for job completion."""
    
    type: str = "completed"
    data: dict  # Completion data (artifacts, stats, etc.)

class ErrorWebSocketMessage(WebSocketMessage):
    """WebSocket message for errors."""
    
    type: str = "error"
    data: dict  # Error data (message, code, etc.)
