import pytest
from httpx import AsyncClient
from app.main import app

client = AsyncClient(app=app, base_url="http://test")

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "CryoMamba" in response.json()["message"]

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/v1/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert "timestamp" in data
    assert data["service"] == "cryomamba-inference-server"

def test_server_info_endpoint():
    """Test server info endpoint"""
    response = client.get("/v1/server/info")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "cryomamba-inference-server"
    assert data["version"] == "1.0.0"
    assert "system" in data
    assert data["status"] == "running"

def test_create_job():
    """Test job creation endpoint"""
    response = client.post("/v1/jobs")
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "created"
    assert data["message"] == "Dummy job created successfully"
    return data["job_id"]

def test_get_job():
    """Test job retrieval endpoint"""
    # First create a job
    job_id = test_create_job()
    
    # Then retrieve it
    response = client.get(f"/v1/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["state"] == "pending"
    assert data["params"]["dummy"] is True

def test_get_nonexistent_job():
    """Test retrieving non-existent job"""
    response = client.get("/v1/jobs/nonexistent-id")
    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "Job not found"

def test_cancel_job():
    """Test job cancellation endpoint"""
    # First create a job
    job_id = test_create_job()
    
    # Then cancel it
    response = client.delete(f"/v1/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "cancelled"
