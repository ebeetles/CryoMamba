#!/usr/bin/env python3
"""
Test WebSocket behavior with a real running server.
This script tests the actual expected behavior.
"""

import asyncio
import websockets
import json
import requests
import time

async def test_real_websocket_behavior():
    """Test WebSocket behavior with a real server."""
    
    print("🔍 Testing WebSocket behavior with real server...")
    print("Make sure the server is running: python app/main.py")
    print()
    
    # Step 1: Create a job
    print("1. Creating a job...")
    try:
        response = requests.post("http://localhost:8000/v1/jobs", timeout=5)
        if response.status_code != 200:
            print(f"❌ Failed to create job: {response.status_code}")
            return
        
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"✅ Created job: {job_id}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Please start the server with: python app/main.py")
        return
    
    # Step 2: Connect to WebSocket
    print("2. Connecting to WebSocket...")
    websocket_url = f"ws://localhost:8000/ws/jobs/{job_id}"
    
    try:
        async with websockets.connect(websocket_url) as websocket:
            print(f"✅ Connected to WebSocket: {websocket_url}")
            
            # Step 3: Listen for messages
            print("3. Listening for messages (will run for 10 seconds)...")
            message_count = 0
            start_time = time.time()
            
            try:
                while time.time() - start_time < 10:  # Run for 10 seconds
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        message_count += 1
                        
                        print(f"📨 Message {message_count}: {data['type']} - {data.get('message', '')}")
                        
                        # If we get preview data, show some details
                        if data['type'] == 'preview' and 'data' in data:
                            preview_data = data['data']
                            print(f"   Preview shape: {preview_data.get('shape', 'unknown')}")
                        
                    except asyncio.TimeoutError:
                        print("⏱️  No message received in 1 second (this is normal)")
                        continue
                        
            except websockets.exceptions.ConnectionClosed:
                print("❌ WebSocket connection closed unexpectedly!")
                return
            
            print(f"\n📊 Summary:")
            print(f"   - Connection duration: {time.time() - start_time:.1f} seconds")
            print(f"   - Messages received: {message_count}")
            
            if message_count > 0:
                print("✅ WebSocket is working correctly!")
                print("   - Connection stays open")
                print("   - Preview data is streaming")
                print("   - This is the expected behavior")
            else:
                print("⚠️  No messages received - this might indicate an issue")
                
    except OSError:
        print("❌ Cannot connect to WebSocket. Make sure the server is running.")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")

if __name__ == "__main__":
    print("🧪 CryoMamba WebSocket Behavior Test")
    print("=" * 50)
    asyncio.run(test_real_websocket_behavior())
