import pytest
import httpx
from httpx import ASGITransport
from app.main import app

# Create a test client using httpx with ASGITransport
transport = ASGITransport(app=app)
client = httpx.AsyncClient(transport=transport, base_url="http://test")

@pytest.mark.asyncio
async def test_root_endpoint():
    """Test root endpoint"""
    response = await client.get("/")
    assert response.status_code == 200
    assert "CryoMamba" in response.json()["message"]

@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check endpoint"""
    response = await client.get("/v1/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert "timestamp" in data
    assert data["service"] == "cryomamba-inference-server"

@pytest.mark.asyncio
async def test_server_info_endpoint():
    """Test server info endpoint"""
    response = await client.get("/v1/server/info")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "cryomamba-inference-server"
    assert data["version"] == "1.0.0"
    assert "system" in data
    assert data["status"] == "running"

@pytest.mark.asyncio
async def test_create_job():
    """Test job creation endpoint"""
    response = await client.post("/v1/jobs")
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "created"
    assert "Dummy job created successfully" in data["message"]
    return data["job_id"]

@pytest.mark.asyncio
async def test_get_job():
    """Test job retrieval endpoint"""
    # First create a job
    job_id = await test_create_job()
    
    # Then retrieve it
    response = await client.get(f"/v1/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["state"] in ["pending", "running"]

@pytest.mark.asyncio
async def test_get_nonexistent_job():
    """Test retrieving non-existent job"""
    response = await client.get("/v1/jobs/nonexistent-id")
    assert response.status_code == 404
    data = response.json()
    assert data["message"] == "Job not found"

@pytest.mark.asyncio
async def test_cancel_job():
    """Test job cancellation endpoint"""
    # First create a job
    job_id = await test_create_job()
    
    # Then cancel it
    response = await client.delete(f"/v1/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "cancelled"
