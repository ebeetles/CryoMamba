#!/usr/bin/env python3
"""
CryoMamba Development Script
Provides common development tasks and utilities
"""

import subprocess
import sys
import json
import requests
import time
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command with improved error handling"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if check and result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)
        return result
    except subprocess.TimeoutExpired:
        print(f"Command timed out after 5 minutes: {cmd}")
        sys.exit(1)
    except Exception as e:
        print(f"Command failed with exception: {e}")
        sys.exit(1)

def check_docker():
    """Check if Docker is running"""
    try:
        run_command("docker --version")
        run_command("docker-compose --version")
        return True
    except:
        print("Docker or Docker Compose not found. Please install Docker.")
        return False

def build_image():
    """Build Docker image"""
    print("Building Docker image...")
    run_command("docker-compose build")

def start_server():
    """Start development server"""
    print("Starting development server...")
    run_command("docker-compose up -d")

def stop_server():
    """Stop development server"""
    print("Stopping development server...")
    run_command("docker-compose down")

def check_health():
    """Check server health"""
    print("Checking server health...")
    try:
        response = requests.get("http://localhost:8000/v1/healthz", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")

def check_server_info():
    """Check server info"""
    print("Checking server info...")
    try:
        response = requests.get("http://localhost:8000/v1/server/info", timeout=5)
        if response.status_code == 200:
            print("✅ Server info retrieved")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Server info check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")

def wait_for_server(max_attempts=30):
    """Wait for server to be ready"""
    print("Waiting for server to be ready...")
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/v1/healthz", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except:
            pass
        print(f"Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(2)
    
    print("❌ Server failed to start within timeout")
    return False

def show_logs():
    """Show server logs"""
    print("Showing server logs...")
    run_command("docker-compose logs -f cryomamba-server", check=False)

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python dev.py <command>")
        print("Commands:")
        print("  build     - Build Docker image")
        print("  start     - Start development server")
        print("  stop      - Stop development server")
        print("  restart   - Restart development server")
        print("  health    - Check server health")
        print("  info      - Check server info")
        print("  logs      - Show server logs")
        print("  test      - Run full test suite")
        print("  dev       - Start development workflow")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "build":
        if not check_docker():
            sys.exit(1)
        build_image()
    
    elif command == "start":
        if not check_docker():
            sys.exit(1)
        start_server()
        wait_for_server()
        check_health()
    
    elif command == "stop":
        stop_server()
    
    elif command == "restart":
        stop_server()
        time.sleep(2)
        start_server()
        wait_for_server()
        check_health()
    
    elif command == "health":
        check_health()
    
    elif command == "info":
        check_server_info()
    
    elif command == "logs":
        show_logs()
    
    elif command == "test":
        check_health()
        check_server_info()
    
    elif command == "dev":
        if not check_docker():
            sys.exit(1)
        build_image()
        start_server()
        wait_for_server()
        check_health()
        check_server_info()
        print("\n🚀 Development server is ready!")
        print("Access the server at: http://localhost:8000")
        print("API docs at: http://localhost:8000/docs")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
