"""
WebSocket client for connecting to CryoMamba server and receiving preview updates.
"""

import asyncio
import json
import logging
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import websockets
import numpy as np
from qtpy.QtCore import QObject, Signal, QThread, pyqtSignal
from qtpy.QtWidgets import QApplication

logger = logging.getLogger(__name__)

class WebSocketClient(QObject):
    """WebSocket client for receiving job updates and preview data."""
    
    # Signals for UI updates
    connected = Signal(str)  # job_id
    disconnected = Signal(str)  # job_id
    preview_received = Signal(str, dict)  # job_id, preview_data
    progress_received = Signal(str, dict)  # job_id, progress_data
    error_received = Signal(str, dict)  # job_id, error_data
    job_completed = Signal(str, dict)  # job_id, completion_data
    
    def __init__(self, server_url: str = "ws://localhost:8000"):
        super().__init__()
        self.server_url = server_url
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.job_id: Optional[str] = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0
        
    async def connect_to_job(self, job_id: str):
        """
        Connect to WebSocket for a specific job.
        
        Args:
            job_id: Job ID to connect to
        """
        if self.is_connected and self.job_id == job_id:
            logger.info(f"Already connected to job {job_id}")
            return
        
        if self.is_connected:
            await self.disconnect()
        
        self.job_id = job_id
        websocket_url = f"{self.server_url}/ws/jobs/{job_id}"
        
        try:
            logger.info(f"Connecting to WebSocket: {websocket_url}")
            self.websocket = await websockets.connect(websocket_url)
            self.is_connected = True
            self.reconnect_attempts = 0
            
            # Emit connected signal
            self.connected.emit(job_id)
            
            # Start listening for messages
            await self._listen_for_messages()
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            await self._handle_reconnect()
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        if self.is_connected:
            self.is_connected = False
            if self.job_id:
                self.disconnected.emit(self.job_id)
                self.job_id = None
    
    async def _listen_for_messages(self):
        """Listen for incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON received: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            await self._handle_reconnect()
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
            await self._handle_reconnect()
    
    async def _handle_message(self, data: dict):
        """Handle incoming WebSocket message."""
        message_type = data.get("type")
        job_id = data.get("job_id")
        message_data = data.get("data", {})
        
        if not job_id or job_id != self.job_id:
            logger.warning(f"Received message for different job: {job_id}")
            return
        
        # Emit appropriate signal based on message type
        if message_type == "preview":
            self.preview_received.emit(job_id, message_data)
        elif message_type == "progress":
            self.progress_received.emit(job_id, message_data)
        elif message_type == "completed":
            self.job_completed.emit(job_id, message_data)
        elif message_type == "error":
            self.error_received.emit(job_id, message_data)
        elif message_type == "connected":
            logger.info(f"Connected to job {job_id}")
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    async def _handle_reconnect(self):
        """Handle reconnection logic."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self.is_connected = False
            if self.job_id:
                self.disconnected.emit(self.job_id)
            return
        
        self.reconnect_attempts += 1
        delay = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
        
        logger.info(f"Attempting to reconnect in {delay} seconds (attempt {self.reconnect_attempts})")
        await asyncio.sleep(delay)
        
        if self.job_id:
            await self.connect_to_job(self.job_id)
    
    async def send_ping(self):
        """Send ping message to keep connection alive."""
        if self.websocket and self.is_connected:
            try:
                ping_message = {
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                }
                await self.websocket.send(json.dumps(ping_message))
            except Exception as e:
                logger.error(f"Failed to send ping: {e}")

class WebSocketWorker(QThread):
    """Worker thread for WebSocket operations."""
    
    def __init__(self, client: WebSocketClient):
        super().__init__()
        self.client = client
        self.loop: Optional[asyncio.AbstractEventLoop] = None
    
    def run(self):
        """Run the asyncio event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()
    
    def stop(self):
        """Stop the event loop."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
    
    async def connect_to_job(self, job_id: str):
        """Connect to a job (thread-safe)."""
        if self.loop:
            await self.client.connect_to_job(job_id)
    
    async def disconnect(self):
        """Disconnect (thread-safe)."""
        if self.loop:
            await self.client.disconnect()

class PreviewDataProcessor:
    """Processes preview data for napari display."""
    
    @staticmethod
    def decode_preview_data(preview_data: dict) -> np.ndarray:
        """
        Decode preview data from WebSocket message.
        
        Args:
            preview_data: Preview message data containing base64 payload
        
        Returns:
            Numpy array containing the preview data
        """
        import base64
        
        # Extract data from preview message
        payload = preview_data.get("payload")
        shape = preview_data.get("shape")
        dtype = preview_data.get("dtype")
        
        if not all([payload, shape, dtype]):
            raise ValueError("Invalid preview data format")
        
        # Decode base64 payload
        data_bytes = base64.b64decode(payload.encode('utf-8'))
        
        # Reconstruct numpy array
        data = np.frombuffer(data_bytes, dtype=dtype)
        data = data.reshape(shape)
        
        return data
    
    @staticmethod
    def get_preview_metadata(preview_data: dict) -> dict:
        """
        Extract metadata from preview data.
        
        Args:
            preview_data: Preview message data
        
        Returns:
            Dictionary containing metadata
        """
        return {
            "scale": preview_data.get("scale", [1.0, 1.0, 1.0]),
            "shape": preview_data.get("shape"),
            "dtype": preview_data.get("dtype"),
            "timestamp": datetime.now().isoformat()
        }
