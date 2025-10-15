"""
Tests for WebSocket functionality.
"""

import pytest
import pytest_asyncio
import asyncio
import json
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import websockets
from app.main import app
from app.models.preview import PreviewMessage

class TestWebSocketEndpoints:
    """Test WebSocket endpoint functionality."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_websocket_endpoint_invalid_job_id(self):
        """Test WebSocket endpoint with invalid job ID."""
        with self.client.websocket_connect("/ws/jobs/invalid-id") as websocket:
            # Should close with error code
            pass
    
    def test_websocket_endpoint_valid_job_id(self):
        """Test WebSocket endpoint with valid job ID."""
        import uuid
        job_id = str(uuid.uuid4())
        
        with self.client.websocket_connect(f"/ws/jobs/{job_id}") as websocket:
            # Should receive connection confirmation
            data = websocket.receive_text()
            message = json.loads(data)
            
            assert message["type"] == "connected"
            assert message["job_id"] == job_id
            assert "timestamp" in message
    
    def test_websocket_ping_pong(self):
        """Test WebSocket ping/pong functionality."""
        import uuid
        job_id = str(uuid.uuid4())
        
        with self.client.websocket_connect(f"/ws/jobs/{job_id}") as websocket:
            # Send ping
            ping_message = {"type": "ping"}
            websocket.send_text(json.dumps(ping_message))
            
            # Should receive pong
            data = websocket.receive_text()
            message = json.loads(data)
            
            assert message["type"] == "pong"
            assert "timestamp" in message

class TestPreviewMessage:
    """Test preview message functionality."""
    
    def test_preview_message_creation(self):
        """Test creating preview message from numpy array."""
        # Create test data
        test_data = np.random.randint(0, 2, size=(32, 32, 32), dtype=np.uint8)
        scale = [1.0, 1.0, 1.0]
        
        # Create preview message
        preview = PreviewMessage.from_numpy_array(test_data, scale)
        
        # Verify properties
        assert preview.scale == scale
        assert preview.shape == list(test_data.shape)
        assert preview.dtype == "uint8"
        assert isinstance(preview.payload, str)
    
    def test_preview_message_roundtrip(self):
        """Test encoding and decoding preview message."""
        # Create test data
        original_data = np.random.randint(0, 2, size=(16, 16, 16), dtype=np.uint8)
        scale = [2.0, 2.0, 2.0]
        
        # Create and decode preview message
        preview = PreviewMessage.from_numpy_array(original_data, scale)
        decoded_data = preview.to_numpy_array()
        
        # Verify data integrity
        np.testing.assert_array_equal(original_data, decoded_data)
        assert preview.scale == scale
    
    def test_preview_message_validation(self):
        """Test preview message validation."""
        # Create test data
        test_data = np.random.randint(0, 2, size=(8, 8, 8), dtype=np.uint8)
        preview = PreviewMessage.from_numpy_array(test_data)
        
        # Test validation
        assert preview.validate_dimensions((8, 8, 8)) == True
        assert preview.validate_dimensions((16, 16, 16)) == False

class TestFakePreviewGeneration:
    """Test fake preview data generation."""
    
    def test_generate_fake_preview_data(self):
        """Test fake preview data generation."""
        from app.routes.websocket import generate_fake_preview_data
        
        volume_shape = (64, 64, 64)
        preview_data = generate_fake_preview_data(volume_shape)
        
        # Verify structure
        assert "scale" in preview_data
        assert "shape" in preview_data
        assert "dtype" in preview_data
        assert "payload" in preview_data
        
        # Verify values
        assert preview_data["scale"] == [1.0, 1.0, 1.0]
        assert preview_data["shape"] == list(volume_shape)
        assert preview_data["dtype"] == "uint8"
        assert isinstance(preview_data["payload"], str)
    
    def test_fake_preview_data_decoding(self):
        """Test that fake preview data can be decoded."""
        from app.routes.websocket import generate_fake_preview_data
        
        volume_shape = (32, 32, 32)
        preview_data = generate_fake_preview_data(volume_shape)
        
        # Create PreviewMessage and decode
        preview = PreviewMessage(**preview_data)
        decoded_data = preview.to_numpy_array()
        
        # Verify decoded data
        assert decoded_data.shape == volume_shape
        assert decoded_data.dtype == np.uint8
        assert np.all((decoded_data == 0) | (decoded_data == 1))  # Binary data

class TestJobIntegration:
    """Test job creation and WebSocket integration."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_create_job_starts_preview_streaming(self):
        """Test that creating a job starts preview streaming."""
        response = self.client.post("/v1/jobs")
        assert response.status_code == 200
        
        job_data = response.json()
        job_id = job_data["job_id"]
        
        # Verify job was created
        assert "job_id" in job_data
        assert "status" in job_data
        assert job_data["status"] == "created"
    
    def test_job_status_endpoint(self):
        """Test job status endpoint."""
        # Create a job first
        response = self.client.post("/v1/jobs")
        job_id = response.json()["job_id"]
        
        # Get job status
        response = self.client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        
        job_data = response.json()
        assert job_data["job_id"] == job_id
        assert "state" in job_data
        assert "created_at" in job_data
        assert "updated_at" in job_data
    
    def test_cancel_job_stops_preview_streaming(self):
        """Test that cancelling a job stops preview streaming."""
        # Create a job first
        response = self.client.post("/v1/jobs")
        job_id = response.json()["job_id"]
        
        # Cancel the job
        response = self.client.delete(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        
        job_data = response.json()
        assert job_data["job_id"] == job_id
        assert job_data["status"] == "cancelled"

@pytest.mark.asyncio
class TestWebSocketClient:
    """Test WebSocket client functionality."""
    
    async def test_websocket_client_connection(self):
        """Test WebSocket client connection."""
        from napari_cryomamba.napari_cryomamba.websocket_client import WebSocketClient
        
        client = WebSocketClient("ws://localhost:8000")
        
        # Mock the websockets.connect to avoid actual connection
        with patch('websockets.connect') as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value = mock_websocket
            
            # Mock message iteration
            mock_websocket.__aiter__.return_value = []
            
            # Test connection
            job_id = "test-job-id"
            await client.connect_to_job(job_id)
            
            # Verify connection was attempted
            mock_connect.assert_called_once_with(f"ws://localhost:8000/ws/jobs/{job_id}")
    
    async def test_preview_data_processor(self):
        """Test preview data processing."""
        from napari_cryomamba.napari_cryomamba.websocket_client import PreviewDataProcessor
        
        # Create test preview data
        test_data = np.random.randint(0, 2, size=(16, 16, 16), dtype=np.uint8)
        preview_message = PreviewMessage.from_numpy_array(test_data)
        preview_data = preview_message.model_dump()
        
        # Test decoding
        decoded_data = PreviewDataProcessor.decode_preview_data(preview_data)
        np.testing.assert_array_equal(test_data, decoded_data)
        
        # Test metadata extraction
        metadata = PreviewDataProcessor.get_preview_metadata(preview_data)
        assert "scale" in metadata
        assert "shape" in metadata
        assert "dtype" in metadata
        assert "timestamp" in metadata

if __name__ == "__main__":
    pytest.main([__file__])
