# Quick Start Script for Ayurvedic Chatbot API
# File: start_api.py

import os
import sys
import subprocess
import time

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    
    try:
        import flask
        import flask_cors
        import pandas
        import numpy
        import langchain
        print("✅ All core dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has required variables"""
    print("🔍 Checking environment configuration...")
    
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("💡 Create a .env file with your API keys:")
        print("   GROQ_API_KEY=your_groq_api_key_here")
        print("   HF_TOKEN=your_huggingface_token_here")
        return False
    
    # Load and check .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ["GROQ_API_KEY", "HF_TOKEN"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("💡 Add these to your .env file")
        return False
    
    print("✅ Environment configuration is valid")
    return True

def get_local_ip():
    """Get the local IP address"""
    import socket
    try:
        # Connect to a remote server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def main():
    """Main startup function"""
    print("🌿 Ayurvedic Chatbot API - Quick Start")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please install dependencies first.")
        return
    
    # Check environment
    if not check_env_file():
        print("\n❌ Environment check failed. Please configure your .env file.")
        return
    
    # Get local IP
    local_ip = get_local_ip()
    
    print("\n🚀 Starting Flask API Server...")
    print("=" * 50)
    print(f"📡 Local access: http://localhost:5000")
    print(f"🌐 Network access: http://{local_ip}:5000")
    print("=" * 50)
    print("📋 Available endpoints:")
    print("   GET  /          - API information")
    print("   POST /chat      - Send message to chatbot")
    print("   GET  /health    - Health check")
    print("   GET  /info      - System information")
    print("   POST /clear     - Clear conversation history")
    print("=" * 50)
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask app
    try:
        from app import app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("💡 Check your configuration and try again")

if __name__ == "__main__":
    main()
