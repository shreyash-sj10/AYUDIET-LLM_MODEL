# Flask Web API for Ayurvedic Chatbot
# File: app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from datetime import datetime, timezone
import time
from collections import deque
from chatbot_core import get_chatbot_response, clear_session, get_system_info

# Ensure Unicode log/output works on Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Initialize Flask app
app = Flask(__name__)

cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").strip()
allowed_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip() and origin.strip() != "*"]
if not allowed_origins:
    allowed_origins = ["http://localhost:3000"]
CORS(app, resources={r"/*": {"origins": allowed_origins}})

API_KEY = os.getenv("AYUDIET_API_KEY", "").strip()
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
_RATE_BUCKETS = {}


def _enforce_api_key():
    if not API_KEY:
        return None
    provided = request.headers.get("X-API-Key", "").strip()
    if provided != API_KEY:
        return jsonify({"error": "Invalid API key", "status": "error"}), 401
    return None


def _enforce_rate_limit():
    ip = request.remote_addr or "unknown"
    now = time.time()
    bucket = _RATE_BUCKETS.setdefault(ip, deque())
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
        return jsonify({"error": "Rate limit exceeded", "status": "error"}), 429
    bucket.append(now)
    return None


@app.before_request
def _security_guard():
    if request.path in ["/chat", "/clear"]:
        auth_error = _enforce_api_key()
        if auth_error:
            return auth_error
        rate_error = _enforce_rate_limit()
        if rate_error:
            return rate_error

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    try:
        system_info = get_system_info()
        return jsonify({
            "message": "🌿 Ayurvedic Dietitian Chatbot API",
            "deprecation_notice": "Use FastAPI production entrypoint: uvicorn main:app",
            "version": system_info.get('version', '2.0 Enhanced - API'),
            "status": "active",
            "endpoints": {
                "POST /chat": "Send a message to the chatbot",
                "GET /health": "Check API health status",
                "GET /info": "Get system information",
                "POST /clear": "Clear conversation history for a session"
            },
            "capabilities": system_info.get('capabilities', []),
            "data_sources": system_info.get('data_sources', [])
        })
    except Exception as e:
        return jsonify({
            "message": "🌿 Ayurvedic Dietitian Chatbot API",
            "status": "active",
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Try to initialize the agent to check if everything is working
        from chatbot_core import initialize_agent
        agent = initialize_agent()
        
        return jsonify({
            "status": "healthy",
            "message": "API is running and agent is initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "message": "API is running but agent initialization failed",
            "error": str(e)
        }), 500

@app.route('/info', methods=['GET'])
def info():
    """Get system information"""
    try:
        system_info = get_system_info()
        return jsonify(system_info)
    except Exception as e:
        return jsonify({
            "error": "Failed to get system information",
            "details": str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint
    
    Expected JSON payload:
    {
        "message": "user's message here",
        "session_id": "optional_session_id"
    }
    
    Returns:
    {
        "response": "chatbot's response",
        "session_id": "session_id",
        "status": "success/error"
    }
    """
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON",
                "status": "error"
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'message' not in data:
            return jsonify({
                "error": "Missing required field: 'message'",
                "status": "error"
            }), 400
        
        user_message = data['message'].strip()
        session_id = data.get('session_id')
        
        if not user_message:
            return jsonify({
                "error": "Message cannot be empty",
                "status": "error"
            }), 400
        
        # Get chatbot response
        result = get_chatbot_response(user_message, session_id)
        
        # Return response
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "status": "error"
        }), 500

@app.route('/clear', methods=['POST'])
def clear_conversation():
    """
    Clear conversation history for a session
    
    Expected JSON payload:
    {
        "session_id": "session_id_to_clear"
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON",
                "status": "error"
            }), 400
        
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                "error": "Missing required field: 'session_id'",
                "status": "error"
            }), 400
        
        success = clear_session(session_id)
        
        if success:
            return jsonify({
                "message": "Conversation history cleared successfully",
                "session_id": session_id,
                "status": "success"
            })
        else:
            return jsonify({
                "message": "Session not found or already cleared",
                "session_id": session_id,
                "status": "success"
            })
            
    except Exception as e:
        return jsonify({
            "error": "Failed to clear conversation",
            "details": str(e),
            "status": "error"
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "Available endpoints: GET /, POST /chat, GET /health, GET /info, POST /clear",
        "status": "error"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        "error": "Method not allowed",
        "message": "Check the API documentation for correct HTTP methods",
        "status": "error"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on our end",
        "status": "error"
    }), 500

if __name__ == '__main__':
    print("🌿 Starting Ayurvedic Dietitian Chatbot API...")
    print("=" * 50)
    print("API Endpoints:")
    print("  GET  /          - API information")
    print("  POST /chat      - Send message to chatbot")
    print("  GET  /health    - Health check")
    print("  GET  /info      - System information")
    print("  POST /clear     - Clear conversation history")
    print("=" * 50)
    print("Server will start on http://0.0.0.0:5000")
    print("Access from other devices using your local IP address")
    print("=" * 50)
    
    # Run the Flask app
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(
        host='0.0.0.0',  # Allow access from other devices on the network
        port=5000,       # Default Flask port
        debug=debug_mode
    )


