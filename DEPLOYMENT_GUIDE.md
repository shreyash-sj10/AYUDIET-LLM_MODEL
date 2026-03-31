# 🌿 Ayurvedic Chatbot Flask API - Deployment Guide

## ✅ Deployment Readiness Status

**STATUS: READY FOR DEPLOYMENT** ✅

All components have been tested and are working correctly:
- ✅ Flask application imports successfully
- ✅ All API endpoints respond correctly
- ✅ Agent initialization works
- ✅ Chat functionality operational
- ✅ All dependencies installed

## 📋 Prerequisites

### Required Files
- `app.py` - Main Flask application
- `chatbot_core.py` - Core chatbot functionality
- `requirements.txt` - All dependencies
- `.env` - Environment variables (you need to create this)

### Required Environment Variables
Create a `.env` file in your project directory with:
```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

## 🚀 Deployment Instructions

### Step 1: Install Dependencies
```bash
# Navigate to your project directory
cd C:\Users\Atharv\Documents\SIH4

# Install all required packages
pip install -r requirements.txt
```

### Step 2: Set Up Environment Variables
1. Create a `.env` file in your project directory
2. Add your API keys:
   ```
   GROQ_API_KEY=your_actual_groq_api_key
   HF_TOKEN=your_actual_huggingface_token
   ```

### Step 3: Start the Flask Server
```bash
# Start the Flask application
python app.py
```

The server will start on `http://0.0.0.0:5000`

### Step 4: Find Your Local IP Address

#### Windows (Command Prompt):
```cmd
ipconfig
```
Look for "IPv4 Address" under your active network adapter.

#### Windows (PowerShell):
```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*" -or $_.IPAddress -like "172.16.*"}
```

#### Alternative (Works on both):
```cmd
ipconfig | findstr "IPv4"
```

### Step 5: Access the API

#### From the same computer:
- `http://localhost:5000` or `http://127.0.0.1:5000`

#### From other devices on the same network:
- `http://YOUR_IP_ADDRESS:5000` (replace with your actual IP)

## 🧪 Testing Instructions

### 1. Basic Health Check
```bash
# Test if the API is running
curl http://localhost:5000/health
```

### 2. Test Chat Endpoint
```bash
# Send a message to the chatbot
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, I need help with my diet plan"}'
```

### 3. Using Python requests
```python
import requests

# Test basic endpoints
response = requests.get("http://localhost:5000/")
print(response.json())

# Test chat
chat_data = {
    "message": "Create a diet plan for Vata constitution",
    "session_id": "test_session"
}
response = requests.post("http://localhost:5000/chat", json=chat_data)
print(response.json())
```

### 4. Using the provided test script
```bash
python test_api.py
```

## 📡 API Endpoints

### GET `/`
- **Purpose**: API information and status
- **Response**: System information, capabilities, and available endpoints

### POST `/chat`
- **Purpose**: Send messages to the chatbot
- **Request Body**:
  ```json
  {
    "message": "Your message here",
    "session_id": "optional_session_id"
  }
  ```
- **Response**:
  ```json
  {
    "response": "Chatbot's response",
    "session_id": "session_id",
    "status": "success"
  }
  ```

### GET `/health`
- **Purpose**: Health check endpoint
- **Response**: Server health status

### GET `/info`
- **Purpose**: System information
- **Response**: Version, capabilities, and data sources

### POST `/clear`
- **Purpose**: Clear conversation history
- **Request Body**:
  ```json
  {
    "session_id": "session_to_clear"
  }
  ```

## 🌐 Network Access

### For Local Network Access:
1. Find your computer's IP address (see Step 4 above)
2. Ensure Windows Firewall allows Python/Flask through port 5000
3. Other devices can access: `http://YOUR_IP:5000`

### Windows Firewall Configuration:
1. Open Windows Defender Firewall
2. Click "Allow an app or feature through Windows Defender Firewall"
3. Click "Change settings" → "Allow another app"
4. Browse to your Python executable
5. Check both "Private" and "Public" networks

## 🔧 Troubleshooting

### Common Issues:

1. **"Module not found" errors**:
   ```bash
   pip install -r requirements.txt
   ```

2. **API key errors**:
   - Check your `.env` file exists
   - Verify API keys are correct
   - Ensure no extra spaces in the `.env` file

3. **Port already in use**:
   - Change port in `app.py`: `app.run(host='0.0.0.0', port=5001)`
   - Or kill the process using port 5000

4. **Cannot access from other devices**:
   - Check Windows Firewall settings
   - Verify IP address is correct
   - Ensure devices are on the same network

5. **Agent initialization fails**:
   - Check if data files exist in the `data/` directory
   - Verify API keys are working
   - Check internet connection for model downloads

## 📊 Performance Notes

- **First startup**: May take 1-2 minutes to initialize the agent and load data
- **Memory usage**: ~2-4GB RAM (due to ML models)
- **Response time**: 2-10 seconds per request (depending on complexity)
- **Concurrent users**: Supports multiple simultaneous sessions

## 🔒 Security Considerations

- This is a development server - not suitable for production
- For production, use a proper WSGI server like Gunicorn
- Consider adding authentication if exposing to the internet
- API keys are loaded from environment variables (secure)

## 📱 Frontend Integration

To connect a website to this API:

```javascript
// Example JavaScript frontend code
async function sendMessage(message) {
    const response = await fetch('http://YOUR_IP:5000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: message,
            session_id: 'user_session_123'
        })
    });
    
    const data = await response.json();
    return data.response;
}
```

## 🎯 Success Indicators

Your deployment is successful when:
- ✅ `python app.py` starts without errors
- ✅ `http://localhost:5000/health` returns "healthy"
- ✅ Chat endpoint responds with AI-generated content
- ✅ Other devices can access the API using your IP address

---

**Ready to deploy! 🚀**

Your Ayurvedic Chatbot API is now ready for use. Follow the steps above to get it running and accessible from your local network.
