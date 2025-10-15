"""
End-to-end integration tests for CryoMamba foundation components.
Tests the complete workflow from desktop to server communication.
"""

import pytest
import pytest_asyncio
import asyncio
import websockets
import json
import requests
import numpy as np
from pathlib import Path
import tempfile
import mrcfile
import httpx
from httpx import ASGITransport
from app.main import app


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def server_client(self):
        """Create HTTP client for server testing."""
        transport = ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")
    
    @pytest.fixture
    def test_mrc_file(self):
        """Create a temporary MRC file for testing."""
        # Create a small test volume
        test_data = np.random.randint(0, 255, size=(32, 32, 32), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.mrc', delete=False) as tmp_file:
            with mrcfile.new(tmp_file.name, overwrite=True) as mrc:
                mrc.set_data(test_data)
                mrc.header.cella.x = 32.0
                mrc.header.cella.y = 32.0
                mrc.header.cella.z = 32.0
            
            yield tmp_file.name
        
        # Cleanup
        Path(tmp_file.name).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_server_health_endpoints(self, server_client):
        """Test server health and info endpoints."""
        # Test health endpoint
        response = await server_client.get("/v1/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"
        assert "timestamp" in data
        assert data["service"] == "cryomamba-inference-server"
        
        # Test server info endpoint
        response = await server_client.get("/v1/server/info")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "cryomamba-inference-server"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
    
    @pytest.mark.asyncio
    async def test_job_lifecycle(self, server_client):
        """Test complete job lifecycle."""
        # Create job
        response = await server_client.post("/v1/jobs")
        assert response.status_code == 200
        job_data = response.json()
        job_id = job_data["job_id"]
        assert job_data["status"] == "created"
        
        # Get job status
        response = await server_client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        job_info = response.json()
        assert job_info["job_id"] == job_id
        assert job_info["state"] in ["pending", "running"]
        
        # Cancel job
        response = await server_client.delete(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        cancel_data = response.json()
        assert cancel_data["job_id"] == job_id
        assert cancel_data["status"] == "cancelled"
    
    @pytest.mark.asyncio
    async def test_job_with_volume_parameters(self, server_client):
        """Test job creation with volume parameters."""
        volume_params = {
            "volume_shape": [64, 64, 64],
            "volume_dtype": "uint8",
            "has_volume": True
        }
        
        response = await server_client.post("/v1/jobs", json=volume_params)
        assert response.status_code == 200
        job_data = response.json()
        job_id = job_data["job_id"]
        
        # Verify job was created with correct parameters
        response = await server_client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        job_info = response.json()
        assert job_info["params"]["has_volume"] is True
        assert job_info["params"]["volume_shape"] == [64, 64, 64]
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, server_client):
        """Test WebSocket connection and message handling."""
        # First create a job
        response = await server_client.post("/v1/jobs")
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        
        # Connect to WebSocket
        websocket_url = f"ws://localhost:8000/ws/jobs/{job_id}"
        
        try:
            async with websockets.connect(websocket_url) as websocket:
                # Wait for initial connection message
                message = await websocket.receive()
                data = json.loads(message)
                assert data["type"] == "connected"
                assert data["job_id"] == job_id
                
                # Wait for job creation message
                message = await websocket.receive()
                data = json.loads(message)
                assert data["type"] == "job_created"
                assert data["job_id"] == job_id
                
                # Wait for job started message
                message = await websocket.receive()
                data = json.loads(message)
                assert data["type"] == "job_started"
                assert data["job_id"] == job_id
                
                # Wait for preview messages (should receive at least one)
                preview_received = False
                for _ in range(3):  # Wait for up to 3 seconds
                    try:
                        message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                        data = json.loads(message)
                        if data["type"] == "preview":
                            assert data["job_id"] == job_id
                            assert "data" in data
                            preview_data = data["data"]
                            assert "shape" in preview_data
                            assert "payload" in preview_data
                            preview_received = True
                            break
                    except asyncio.TimeoutError:
                        continue
                
                assert preview_received, "No preview message received"
                
        except OSError:
            pytest.skip("WebSocket server not running - start server with 'python app/main.py'")
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling(self, server_client):
        """Test WebSocket error handling with invalid job ID."""
        invalid_job_id = "invalid-job-id"
        websocket_url = f"ws://localhost:8000/ws/jobs/{invalid_job_id}"
        
        try:
            async with websockets.connect(websocket_url) as websocket:
                # Should receive connection confirmation
                message = await websocket.receive()
                data = json.loads(message)
                assert data["type"] == "connected"
                
                # No further messages should be received for invalid job
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=2.0)
                    # If we get here, something unexpected happened
                    pytest.fail("Unexpected message received for invalid job")
                except asyncio.TimeoutError:
                    # This is expected - no messages for invalid job
                    pass
                    
        except OSError:
            pytest.skip("WebSocket server not running - start server with 'python app/main.py'")
    
    def test_preview_data_encoding_decoding(self):
        """Test preview data encoding and decoding."""
        from app.models.preview import PreviewMessage
        
        # Create test data
        test_shape = (32, 32, 32)
        test_data = np.random.randint(0, 2, size=test_shape, dtype=np.uint8)
        
        # Encode data
        preview_message = PreviewMessage.from_numpy_array(test_data, scale=[1.0, 1.0, 1.0])
        
        # Verify encoding
        assert preview_message.shape == list(test_shape)
        assert preview_message.dtype == "uint8"
        assert preview_message.scale == [1.0, 1.0, 1.0]
        assert len(preview_message.payload) > 0
        
        # Decode data
        decoded_data = preview_message.to_numpy_array()
        
        # Verify decoding
        assert decoded_data.shape == test_shape
        assert decoded_data.dtype == test_data.dtype
        np.testing.assert_array_equal(decoded_data, test_data)
    
    def test_mrc_file_loading(self, test_mrc_file):
        """Test MRC file loading functionality."""
        import mrcfile
        
        # Load the test MRC file
        with mrcfile.open(test_mrc_file) as mrc:
            data = mrc.data.copy()
            header = mrc.header
        
        # Verify data properties
        assert data.shape == (32, 32, 32)
        # MRC files may use different dtypes, check it's a valid integer type
        assert np.issubdtype(data.dtype, np.integer)
        assert data.size > 0
        
        # Verify header properties
        assert header.nx == 32
        assert header.ny == 32
        assert header.nz == 32
        assert header.cella.x == 32.0
        assert header.cella.y == 32.0
        assert header.cella.z == 32.0
    
    @pytest.mark.asyncio
    async def test_error_scenarios(self, server_client):
        """Test various error scenarios."""
        # Test getting non-existent job
        response = await server_client.get("/v1/jobs/non-existent-id")
        assert response.status_code == 404
        data = response.json()
        assert data["message"] == "Job not found"
        
        # Test cancelling non-existent job
        response = await server_client.delete("/v1/jobs/non-existent-id")
        assert response.status_code == 404
        data = response.json()
        assert data["message"] == "Job not found"
    
    @pytest.mark.asyncio
    async def test_concurrent_jobs(self, server_client):
        """Test handling multiple concurrent jobs."""
        # Create multiple jobs
        job_ids = []
        for i in range(3):
            response = await server_client.post("/v1/jobs")
            assert response.status_code == 200
            job_ids.append(response.json()["job_id"])
        
        # Verify all jobs exist
        for job_id in job_ids:
            response = await server_client.get(f"/v1/jobs/{job_id}")
            assert response.status_code == 200
            assert response.json()["job_id"] == job_id
        
        # Cancel all jobs
        for job_id in job_ids:
            response = await server_client.delete(f"/v1/jobs/{job_id}")
            assert response.status_code == 200
            assert response.json()["status"] == "cancelled"


class TestDesktopIntegration:
    """Test desktop application integration components."""
    
    def test_websocket_client_initialization(self):
        """Test WebSocket client initialization."""
        from napari_cryomamba.napari_cryomamba.websocket_client import WebSocketClient
        
        client = WebSocketClient("ws://localhost:8000")
        assert client.server_url == "ws://localhost:8000"
        assert client.job_id is None
        assert client.is_connected is False
        assert client.reconnect_attempts == 0
        assert client.max_reconnect_attempts == 5
    
    def test_preview_data_processor(self):
        """Test preview data processing."""
        from napari_cryomamba.napari_cryomamba.websocket_client import PreviewDataProcessor
        
        # Create test data
        test_shape = (16, 16, 16)
        test_data = np.random.randint(0, 2, size=test_shape, dtype=np.uint8)
        
        # Create preview message
        preview_data = {
            "scale": [1.0, 1.0, 1.0],
            "shape": list(test_shape),
            "dtype": "uint8",
            "payload": PreviewDataProcessor.encode_data(test_data)
        }
        
        # Decode data
        decoded_data = PreviewDataProcessor.decode_preview_data(preview_data)
        
        # Verify
        assert decoded_data.shape == test_shape
        assert decoded_data.dtype == test_data.dtype
        np.testing.assert_array_equal(decoded_data, test_data)
    
    def test_volume_metadata_extraction(self):
        """Test volume metadata extraction."""
        import tempfile
        import mrcfile
        from pathlib import Path
        
        # Create test volume
        test_data = np.random.randint(0, 255, size=(64, 64, 64), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.mrc', delete=False) as tmp_file:
            with mrcfile.new(tmp_file.name, overwrite=True) as mrc:
                mrc.set_data(test_data)
                mrc.header.cella.x = 64.0
                mrc.header.cella.y = 64.0
                mrc.header.cella.z = 64.0
            
            # Test metadata extraction
            with mrcfile.open(tmp_file.name) as mrc:
                header = mrc.header
                
            metadata = {
                'filename': Path(tmp_file.name).name,
                'shape': test_data.shape,
                'dtype': str(test_data.dtype),
                'voxel_size': (header.cella.x / header.nx, 
                              header.cella.y / header.ny, 
                              header.cella.z / header.nz),
                'min_intensity': float(np.min(test_data)),
                'max_intensity': float(np.max(test_data)),
                'mean_intensity': float(np.mean(test_data)),
                'std_intensity': float(np.std(test_data))
            }
            
            # Verify metadata
            assert metadata['shape'] == (64, 64, 64)
            assert metadata['dtype'] == 'uint8'
            assert metadata['voxel_size'] == (1.0, 1.0, 1.0)
            assert 0 <= metadata['min_intensity'] <= 255
            assert 0 <= metadata['max_intensity'] <= 255
            assert metadata['min_intensity'] <= metadata['mean_intensity'] <= metadata['max_intensity']
            assert metadata['std_intensity'] >= 0
        
        # Cleanup
        Path(tmp_file.name).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
