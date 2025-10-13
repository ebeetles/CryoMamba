import requests
import json

BASE_URL = "http://localhost:8000"

def test_server():
    """Test the FastAPI server endpoints"""
    print("Testing CryoMamba FastAPI Server...")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/v1/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    print(f"✓ Health check: {data}")
    
    # Test server info endpoint
    print("\n2. Testing server info endpoint...")
    response = requests.get(f"{BASE_URL}/v1/server/info")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "cryomamba-inference-server"
    print(f"✓ Server info: {data['service']} v{data['version']}")
    
    # Test job creation
    print("\n3. Testing job creation...")
    response = requests.post(f"{BASE_URL}/v1/jobs")
    assert response.status_code == 200
    data = response.json()
    job_id = data["job_id"]
    assert data["status"] == "created"
    print(f"✓ Job created: {job_id}")
    
    # Test job retrieval
    print("\n4. Testing job retrieval...")
    response = requests.get(f"{BASE_URL}/v1/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["state"] == "pending"
    print(f"✓ Job retrieved: {data['state']}")
    
    # Test job cancellation
    print("\n5. Testing job cancellation...")
    response = requests.delete(f"{BASE_URL}/v1/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cancelled"
    print(f"✓ Job cancelled: {data['status']}")
    
    # Test non-existent job
    print("\n6. Testing non-existent job...")
    response = requests.get(f"{BASE_URL}/v1/jobs/nonexistent")
    assert response.status_code == 404
    print("✓ Non-existent job returns 404")
    
    print("\n🎉 All tests passed!")

if __name__ == "__main__":
    test_server()
