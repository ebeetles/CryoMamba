#!/usr/bin/env python3
"""
CryoMamba Foundation Validation Script

This script validates that the complete CryoMamba foundation setup is working correctly.
It tests server startup, desktop components, and end-to-end integration.
"""

import subprocess
import sys
import time
import requests
import asyncio
import websockets
import json
import tempfile
import mrcfile
import numpy as np
from pathlib import Path


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def run_command(cmd, timeout=30):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"


def test_python_version():
    """Test Python version compatibility."""
    print("🔍 Testing Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        raise ValidationError(f"Python 3.8+ required, found {version.major}.{version.minor}")
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")


def test_dependencies():
    """Test that all required dependencies are installed."""
    print("🔍 Testing dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'napari', 'mrcfile', 'numpy', 
        'httpx', 'websockets', 'pytest', 'qtpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ValidationError(f"Missing packages: {', '.join(missing_packages)}")
    
    print("✅ All dependencies installed - OK")


def test_server_startup():
    """Test that the server can start up."""
    print("🔍 Testing server startup...")
    
    # Start server in background
    server_process = subprocess.Popen(
        [sys.executable, "app/main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    max_wait = 15
    for i in range(max_wait):
        try:
            response = requests.get("http://localhost:8000/v1/healthz", timeout=2)
            if response.status_code == 200:
                print("✅ Server started successfully - OK")
                return server_process
        except requests.exceptions.RequestException:
            time.sleep(1)
    
    # If we get here, server didn't start
    server_process.terminate()
    raise ValidationError("Server failed to start within 15 seconds")


def test_server_endpoints(server_process):
    """Test server endpoints."""
    print("🔍 Testing server endpoints...")
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    response = requests.get(f"{base_url}/v1/healthz")
    if response.status_code != 200:
        raise ValidationError("Health endpoint failed")
    
    # Test server info endpoint
    response = requests.get(f"{base_url}/v1/server/info")
    if response.status_code != 200:
        raise ValidationError("Server info endpoint failed")
    
    # Test job creation
    response = requests.post(f"{base_url}/v1/jobs")
    if response.status_code != 200:
        raise ValidationError("Job creation failed")
    
    job_data = response.json()
    job_id = job_data["job_id"]
    
    # Test job retrieval
    response = requests.get(f"{base_url}/v1/jobs/{job_id}")
    if response.status_code != 200:
        raise ValidationError("Job retrieval failed")
    
    print("✅ All server endpoints working - OK")
    return job_id


async def test_websocket_connection(job_id):
    """Test WebSocket connection and streaming."""
    print("🔍 Testing WebSocket connection...")
    
    websocket_url = f"ws://localhost:8000/ws/jobs/{job_id}"
    
    try:
        async with websockets.connect(websocket_url) as websocket:
            # Wait for initial connection message
            message = await websocket.receive()
            data = json.loads(message)
            if data["type"] != "connected":
                raise ValidationError("WebSocket connection message invalid")
            
            # Wait for preview message
            preview_received = False
            for _ in range(5):  # Wait up to 5 seconds
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                    data = json.loads(message)
                    if data["type"] == "preview":
                        preview_received = True
                        break
                except asyncio.TimeoutError:
                    continue
            
            if not preview_received:
                raise ValidationError("No preview message received")
            
            print("✅ WebSocket streaming working - OK")
            
    except websockets.exceptions.ConnectionRefused:
        raise ValidationError("WebSocket connection refused")


def test_mrc_file_handling():
    """Test MRC file creation and loading."""
    print("🔍 Testing MRC file handling...")
    
    # Create a test MRC file
    test_data = np.random.randint(0, 255, size=(32, 32, 32), dtype=np.uint8)
    
    with tempfile.NamedTemporaryFile(suffix='.mrc', delete=False) as tmp_file:
        with mrcfile.new(tmp_file.name, overwrite=True) as mrc:
            mrc.set_data(test_data)
            mrc.header.cella.x = 32.0
            mrc.header.cella.y = 32.0
            mrc.header.cella.z = 32.0
        
        # Test loading the file
        with mrcfile.open(tmp_file.name) as mrc:
            loaded_data = mrc.data.copy()
        
        # Verify data integrity
        if not np.array_equal(loaded_data, test_data):
            raise ValidationError("MRC file data integrity check failed")
        
        # Cleanup
        Path(tmp_file.name).unlink(missing_ok=True)
    
    print("✅ MRC file handling working - OK")


def test_desktop_components():
    """Test desktop application components."""
    print("🔍 Testing desktop components...")
    
    try:
        # Test WebSocket client initialization
        from napari_cryomamba.napari_cryomamba.websocket_client import WebSocketClient
        client = WebSocketClient("ws://localhost:8000")
        if client.server_url != "ws://localhost:8000":
            raise ValidationError("WebSocket client initialization failed")
        
        # Test preview data processor
        from napari_cryomamba.napari_cryomamba.websocket_client import PreviewDataProcessor
        test_data = np.random.randint(0, 2, size=(16, 16, 16), dtype=np.uint8)
        encoded = PreviewDataProcessor.encode_data(test_data)
        decoded = PreviewDataProcessor.decode_preview_data({
            "scale": [1.0, 1.0, 1.0],
            "shape": list(test_data.shape),
            "dtype": "uint8",
            "payload": encoded
        })
        
        if not np.array_equal(decoded, test_data):
            raise ValidationError("Preview data processing failed")
        
        print("✅ Desktop components working - OK")
        
    except ImportError as e:
        raise ValidationError(f"Desktop component import failed: {e}")


def test_integration_tests():
    """Test that integration tests can run."""
    print("🔍 Testing integration tests...")
    
    success, stdout, stderr = run_command(
        f"{sys.executable} run_e2e_tests.py --server-only",
        timeout=60
    )
    
    if not success:
        raise ValidationError(f"Integration tests failed: {stderr}")
    
    print("✅ Integration tests passing - OK")


def cleanup_server(server_process):
    """Clean up server process."""
    if server_process:
        server_process.terminate()
        server_process.wait(timeout=5)


def main():
    """Main validation function."""
    print("🚀 CryoMamba Foundation Validation")
    print("=" * 50)
    
    server_process = None
    
    try:
        # Run all validation tests
        test_python_version()
        test_dependencies()
        test_mrc_file_handling()
        test_desktop_components()
        
        # Start server for integration tests
        server_process = test_server_startup()
        job_id = test_server_endpoints(server_process)
        
        # Test WebSocket
        asyncio.run(test_websocket_connection(job_id))
        
        # Test integration tests
        test_integration_tests()
        
        print("\n" + "=" * 50)
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ CryoMamba foundation is ready for development")
        print("\nNext steps:")
        print("1. Start server: python app/main.py")
        print("2. Start desktop: python napari_cryomamba/main.py")
        print("3. Follow the workflow in SETUP.md")
        
        return True
        
    except ValidationError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return False
        
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        return False
        
    finally:
        cleanup_server(server_process)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
