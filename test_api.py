# Test script for Flask API
# File: test_api.py

import requests
import json
import time
import threading
from app import app

def test_api_endpoints():
    """Test all API endpoints"""
    
    # Start Flask app in a separate thread
    def run_app():
        app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
    
    # Start the app in background
    server_thread = threading.Thread(target=run_app)
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    base_url = "http://127.0.0.1:5001"
    
    print("🧪 Testing Flask API Endpoints...")
    print("=" * 50)
    
    # Test 1: Home endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ GET / - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Message: {data.get('message', 'N/A')}")
    except Exception as e:
        print(f"❌ GET / - Error: {e}")
    
    # Test 2: Health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ GET /health - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status', 'N/A')}")
    except Exception as e:
        print(f"❌ GET /health - Error: {e}")
    
    # Test 3: Info endpoint
    try:
        response = requests.get(f"{base_url}/info")
        print(f"✅ GET /info - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Version: {data.get('version', 'N/A')}")
    except Exception as e:
        print(f"❌ GET /info - Error: {e}")
    
    # Test 4: Chat endpoint (without API keys - should still work for basic functionality)
    try:
        chat_data = {
            "message": "Hello, can you help me with a diet plan?",
            "session_id": "test_session_123"
        }
        response = requests.post(f"{base_url}/chat", json=chat_data)
        print(f"✅ POST /chat - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {data.get('response', 'N/A')[:100]}...")
            print(f"   Session ID: {data.get('session_id', 'N/A')}")
        else:
            print(f"   Error: {response.text[:100]}...")
    except Exception as e:
        print(f"❌ POST /chat - Error: {e}")
    
    # Test 5: Clear endpoint
    try:
        clear_data = {"session_id": "test_session_123"}
        response = requests.post(f"{base_url}/clear", json=clear_data)
        print(f"✅ POST /clear - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Message: {data.get('message', 'N/A')}")
    except Exception as e:
        print(f"❌ POST /clear - Error: {e}")
    
    print("=" * 50)
    print("🎯 API Testing Complete!")

if __name__ == "__main__":
    test_api_endpoints()
