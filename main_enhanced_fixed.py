# Fixed Enhanced Main Application with Session Management
# File: main_enhanced_fixed.py

import os
import sys
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import the fixed graph builder
from graph_builder_fixed_final import build_agent, get_system_info

# Load environment variables
load_dotenv()

def setup_environment():
    """Setup and validate environment variables"""
    required_vars = ["GROQ_API_KEY", "HF_TOKEN"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and ensure all required variables are set.")
        print("\nExample .env file:")
        print("GROQ_API_KEY=your_groq_api_key_here")
        print("HF_TOKEN=your_huggingface_token_here")
        print("DATA_DIRECTORY=data  # Optional, defaults to 'data'")
        print("GROQ_MODEL=llama-3.1-8b-instant  # Optional, fallback models are tried automatically")
        return False
    
    # Set environment variables for compatibility
    os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    
    print("✅ Environment variables configured")
    return True

def initialize_agent():
    """Initialize the Enhanced LangGraph agent"""
    try:
        print("🔧 Initializing Enhanced Ayurvedic Agent...")
        agent = build_agent()
        print("✅ Agent initialized successfully!")
        return agent
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print(f"Full error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def print_welcome_message():
    """Print enhanced welcome message"""
    try:
        system_info = get_system_info()
    except:
        system_info = {
            'version': '2.0 Enhanced - Fixed',
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
    
    print("\n" + "="*70)
    print("🌿 ENHANCED AYURVEDIC DIETITIAN CHATBOT 🌿")
    print("Smart Indian Hackathon 2025 - Ministry of Ayush")
    print("="*70)
    
    print(f"\n🚀 System Version: {system_info['version']}")
    print("📋 I can help you with:")
    
    capabilities = [
        "🔸 Personalized diet plans based on your Ayurvedic constitution",
        "🔸 Recipe recommendations with nutritional analysis", 
        "🔸 Dosha assessment and lifestyle guidance",
        "🔸 Food properties and Ayurvedic principles",
        "🔸 Health consultations and dietary modifications",
        "🔸 Traditional recipes from 1000+ Indian recipes database"
    ]
    
    for capability in capabilities:
        print(f"   {capability}")
    
    print(f"\n📚 Knowledge Sources:")
    for i, source in enumerate(system_info['data_sources'][:4], 1):
        print(f"   {i}. {source}")
    if len(system_info['data_sources']) > 4:
        print("   ... and more")
    
    print("\n💡 Sample Questions:")
    examples = [
        "👋 'Hello, I'm 25 years old and want to lose weight'",
        "🍽️ 'Create a 7-day diet plan for Pitta constitution'", 
        "🥘 'How do I make moong dal khichdi?'",
        "❓ 'What foods are good for Vata dosha?'",
        "⚖️ 'I have diabetes, what should I eat?'"
    ]
    
    for example in examples:
        print(f"   {example}")
    
    print("\n" + "="*70)
    print("🎯 Ready to provide personalized Ayurvedic nutrition guidance!")
    print("Type 'quit', 'exit', or 'bye' to end the session")
    print("="*70 + "\n")

def print_help():
    """Print help information"""
    print("\n" + "="*50)
    print("📖 AYURVEDIC CHATBOT HELP")
    print("="*50)
    
    print("\n🔹 Getting Started:")
    print("   • Introduce yourself: age, weight, height, health goals")
    print("   • Tell me about dietary preferences and restrictions")
    print("   • Mention any health conditions or concerns")
    
    print("\n🔹 Diet Planning:")
    print("   • 'Create a 7-day diet plan for me'")
    print("   • 'I want to lose weight, suggest a meal plan'") 
    print("   • 'Diet plan for Pitta constitution'")
    
    print("\n🔹 Recipe Requests:")
    print("   • 'How to make palak paneer?'")
    print("   • 'Recipe for moong dal soup'")
    print("   • 'Breakfast recipes for Kapha dosha'")
    
    print("\n🔹 Ayurvedic Knowledge:")
    print("   • 'What is my dosha?'")
    print("   • 'Foods to avoid for Vata imbalance'")
    print("   • 'Benefits of turmeric in Ayurveda'")
    
    print("\n🔹 Profile Management:")
    print("   • 'Update my weight to 70 kg'")
    print("   • 'I am allergic to nuts'")
    print("   • 'My activity level is moderate'")
    
    print("\n📝 Commands:")
    print("   • 'help' - Show this help message")
    print("   • 'profile' - Show your current profile") 
    print("   • 'clear' - Clear conversation history")
    print("   • 'quit' - Exit the chatbot")
    
    print("="*50 + "\n")

def show_profile(agent, session_id):
    """Show current user profile"""
    try:
        from agents_enhanced import PROFILE_STORAGE
        
        profile = PROFILE_STORAGE.get(session_id, {})
        
        if not profile:
            print("\n📋 **Your Profile:** No information stored yet.")
            print("💡 Tell me about yourself to build your profile!")
            return
        
        print("\n📋 **Your Current Profile:**")
        print("="*40)
        
        # Basic info
        if profile.get('name'):
            print(f"👤 Name: {profile['name']}")
        if profile.get('age'):
            print(f"🎂 Age: {profile['age']} years")
        if profile.get('gender'):
            print(f"⚥ Gender: {profile['gender']}")
        if profile.get('weight'):
            print(f"⚖️ Weight: {profile['weight']} kg")
        if profile.get('height'):
            print(f"📏 Height: {profile['height']}")
        
        # Health info
        if profile.get('health_goals'):
            print(f"🎯 Goals: {', '.join(profile['health_goals'])}")
        if profile.get('activity_level'):
            print(f"🏃 Activity: {profile['activity_level']}")
        
        # Ayurvedic info
        if profile.get('primary_dosha'):
            constitution = profile.get('constitution', profile['primary_dosha'])
            print(f"🧘 Constitution: {constitution.title()}")
        
        # Dietary preferences
        if profile.get('dietary_preferences'):
            print(f"🥗 Diet: {', '.join(profile['dietary_preferences'])}")
        if profile.get('allergies'):
            print(f"⚠️ Allergies: {', '.join(profile['allergies'])}")
        
        print("="*40)
        
    except Exception as e:
        print(f"❌ Error showing profile: {e}")

def clear_conversation(session_id):
    """Clear conversation history"""
    try:
        from agents_enhanced import PROFILE_STORAGE
        
        if session_id in PROFILE_STORAGE:
            del PROFILE_STORAGE[session_id]
        
        print("✅ Conversation history and profile cleared!")
        print("💡 Start fresh by telling me about yourself.")
        
    except Exception as e:
        print(f"❌ Error clearing conversation: {e}")

def main():
    """Enhanced main function with better error handling and features"""
    
    # Setup environment
    if not setup_environment():
        return
    
    # Initialize agent
    agent = initialize_agent()
    if not agent:
        print("❌ Failed to start the chatbot. Please check your configuration.")
        print("💡 Make sure you have:")
        print("   1. Created .env file with API keys")
        print("   2. Installed requirements: pip install -r requirements.txt")
        print("   3. Created data/ directory with CSV files (optional)")
        return
    
    # Print welcome message
    print_welcome_message()
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())[:8]
    print(f"🔑 Session ID: {session_id}")
    
    # Main conversation loop
    conversation_history = []
    
    try:
        while True:
            # Get user input
            try:
                user_input = input("\n🔥 You: ").strip()
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Take care of your health!")
                break
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n👋 Thank you for using the Ayurvedic Chatbot!")
                print("🌿 Remember to follow Ayurvedic principles for optimal health!")
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'profile':
                show_profile(agent, session_id)
                continue
            
            elif user_input.lower() in ['clear', 'reset']:
                clear_conversation(session_id)
                conversation_history = []
                continue
            
            # Add user message to conversation history
            conversation_history.append(HumanMessage(content=user_input))
            
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
            
            try:
                print("\n🔄 Processing your request...")
                
                # Invoke the agent
                result = agent.invoke(state, config=config)
                
                # Get the response
                response = result.get("final_response", "I apologize, but I couldn't process that request.")
                
                print(f"\n🤖 **Ayurvedic Assistant:** {response}")
                
                # Add assistant response to conversation history
                conversation_history.append(AIMessage(content=response))
                
                # Keep conversation history manageable
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]
                
            except Exception as e:
                print(f"\n❌ **Error:** I encountered an issue processing your request.")
                print(f"**Details:** {str(e)}")
                print("💡 **Suggestion:** Try rephrasing your question or type 'help' for guidance.")
                
                # Debug information
                if "name 'data' is not defined" in str(e):
                    print("🔧 **Debug Info:** This appears to be a data loading issue.")
                    print("   Check that your CSV files are in the 'data/' directory.")
            
    except Exception as e:
        print(f"\n❌ Unexpected error in main loop: {e}")
        print("💡 Please restart the application.")

if __name__ == "__main__":
    print("DEPRECATED entrypoint: use FastAPI service via `uvicorn main:app` for production.")
    main()
