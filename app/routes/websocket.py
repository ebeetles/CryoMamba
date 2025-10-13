"""
WebSocket routes for real-time job updates and preview streaming.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, Set
import logging
import json
import asyncio
from datetime import datetime
import uuid
from app.models.preview import PreviewMessage, PreviewWebSocketMessage

logger = logging.getLogger(__name__)
router = APIRouter()

# WebSocket connection management
class ConnectionManager:
    """Manages WebSocket connections for job updates."""
    
    def __init__(self):
        # Map job_id to set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Map WebSocket to job_id for cleanup
        self.connection_jobs: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept a WebSocket connection for a specific job."""
        await websocket.accept()
        
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        
        self.active_connections[job_id].add(websocket)
        self.connection_jobs[websocket] = job_id
        
        logger.info(f"WebSocket connected for job {job_id}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.connection_jobs:
            job_id = self.connection_jobs[websocket]
            
            if job_id in self.active_connections:
                self.active_connections[job_id].discard(websocket)
                
                # Clean up empty job connections
                if not self.active_connections[job_id]:
                    del self.active_connections[job_id]
            
            del self.connection_jobs[websocket]
            logger.info(f"WebSocket disconnected for job {job_id}")
    
    async def send_message(self, job_id: str, message: dict):
        """Send a message to all connections for a specific job."""
        if job_id not in self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = set()
        
        for connection in self.active_connections[job_id]:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_to_job(self, job_id: str, message_type: str, data: dict):
        """Broadcast a structured message to all connections for a job."""
        message = {
            "type": message_type,
            "job_id": job_id,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        await self.send_message(job_id, message)

# Global connection manager
manager = ConnectionManager()

@router.websocket("/ws/jobs/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for job updates and preview streaming.
    
    Accepts connections on /ws/jobs/{job_id} and streams:
    - progress updates
    - preview data
    - completion notifications
    - error messages
    """
    try:
        # Validate job_id format (basic UUID validation)
        try:
            uuid.UUID(job_id)
        except ValueError:
            await websocket.close(code=4000, reason="Invalid job_id format")
            return
        
        # Connect to the job's WebSocket channel
        await manager.connect(websocket, job_id)
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "job_id": job_id,
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to job updates"
        }))
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (ping/pong, etc.)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle client messages if needed
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from WebSocket for job {job_id}")
                continue
            except Exception as e:
                logger.error(f"Error handling WebSocket message for job {job_id}: {e}")
                break
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        manager.disconnect(websocket)

# Helper function to broadcast messages from other parts of the application
async def broadcast_job_update(job_id: str, message_type: str, data: dict):
    """Broadcast a job update to all connected WebSocket clients."""
    await manager.broadcast_to_job(job_id, message_type, data)

# Helper function to start fake preview streaming for testing
async def start_fake_preview_streaming(job_id: str, volume_shape: tuple, frequency: float = 1.0):
    """
    Start fake preview streaming for testing purposes.
    
    Args:
        job_id: Job ID to stream previews for
        volume_shape: Shape of the volume (z, y, x)
        frequency: Streaming frequency in Hz (default 1.0)
    """
    logger.info(f"Starting fake preview streaming for job {job_id}")
    
    try:
        while True:
            # Generate fake preview data
            preview_data = generate_fake_preview_data(volume_shape)
            
            # Send preview message
            await broadcast_job_update(job_id, "preview", preview_data)
            
            # Wait for next update
            await asyncio.sleep(1.0 / frequency)
            
    except asyncio.CancelledError:
        logger.info(f"Fake preview streaming cancelled for job {job_id}")
    except Exception as e:
        logger.error(f"Error in fake preview streaming for job {job_id}: {e}")
        await broadcast_job_update(job_id, "error", {
            "message": f"Preview streaming error: {str(e)}"
        })

def generate_fake_preview_data(volume_shape: tuple) -> dict:
    """
    Generate fake preview data matching the volume dimensions.
    
    Args:
        volume_shape: Shape of the volume (z, y, x)
    
    Returns:
        Dictionary with preview message data
    """
    import numpy as np
    
    # Generate random mask data matching volume dimensions
    mask_data = np.random.randint(0, 2, size=volume_shape, dtype=np.uint8)
    
    # Create PreviewMessage using the model
    preview_message = PreviewMessage.from_numpy_array(mask_data, scale=[1.0, 1.0, 1.0])
    
    return preview_message.model_dump()
