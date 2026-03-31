# Core Chatbot Function for Flask API
# File: chatbot_core.py

import os
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import the fixed graph builder (using the working version)
from graph_builder_fixed_final import build_agent, get_system_info

# Load environment variables
load_dotenv()

# Global agent instance for reuse
_agent = None
_conversation_histories = {}

def setup_environment():
    """Setup and validate environment variables"""
    required_vars = ["GROQ_API_KEY", "HF_TOKEN"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Set environment variables for compatibility
    os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    
    return True

def initialize_agent():
    """Initialize the Enhanced LangGraph agent (singleton pattern)"""
    global _agent
    
    if _agent is None:
        try:
            print("🔧 Initializing Enhanced Ayurvedic Agent...")
            _agent = build_agent()
            print("✅ Agent initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            raise e
    
    return _agent

def get_chatbot_response(user_message, session_id=None):
    """
    Main function to get chatbot response for Flask API
    
    Args:
        user_message (str): The user's input message
        session_id (str, optional): Session ID for conversation continuity
    
    Returns:
        dict: Response containing the chatbot's reply and session info
    """
    try:
        # Setup environment if not already done
        if not os.getenv("GROQ_API_KEY") or not os.getenv("HF_TOKEN"):
            setup_environment()
        
        # Initialize agent
        agent = initialize_agent()
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        
        # Get or create conversation history for this session
        if session_id not in _conversation_histories:
            _conversation_histories[session_id] = []
        
        conversation_history = _conversation_histories[session_id]
        
        # Add user message to conversation history
        conversation_history.append(HumanMessage(content=user_message))
        
        # Prepare state for the agent
        config = {"configurable": {"thread_id": session_id}}
        
        state = {
            "messages": conversation_history,
            "user_profile": {},
            "route_to": "",
            "final_response": "",
            "session_id": session_id,
            "context": {}
        }
        
        # Invoke the agent
        result = agent.invoke(state, config=config)
        
        # Get the response
        response = result.get("final_response", "I apologize, but I couldn't process that request.")
        
        # Add assistant response to conversation history
        conversation_history.append(AIMessage(content=response))
        
        # Keep conversation history manageable (last 20 messages)
        if len(conversation_history) > 20:
            _conversation_histories[session_id] = conversation_history[-20:]
        
        return {
            "response": response,
            "session_id": session_id,
            "status": "success"
        }
        
    except Exception as e:
        error_msg = f"I encountered an issue processing your request: {str(e)}"
        return {
            "response": error_msg,
            "session_id": session_id or "error",
            "status": "error",
            "error": str(e)
        }

def clear_session(session_id):
    """Clear conversation history for a specific session"""
    if session_id in _conversation_histories:
        del _conversation_histories[session_id]
        return True
    return False

def get_system_info():
    """Get system information for the API"""
    try:
        return get_system_info()
    except:
        return {
            'version': '2.0 Enhanced - API',
            'capabilities': [
                "Personalized diet plans based on your Ayurvedic constitution",
                "Recipe recommendations with nutritional analysis", 
                "Dosha assessment and lifestyle guidance",
                "Food properties and Ayurvedic principles",
                "Health consultations and dietary modifications"
            ],
            'data_sources': [
                "ICMR-NIN Food Composition Tables",
                "UK Food Composition Database", 
                "USDA Food Database",
                "Indian Recipe Database"
            ]
        }

