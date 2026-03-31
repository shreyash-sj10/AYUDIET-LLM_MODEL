# Fixed Enhanced Graph Builder with Professional Architecture
# File: graph_builder_fixed_final.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Import the enhanced agents
from agents_enhanced import (
    route_query, knowledge_retrieval_agent, diet_plan_agent,
    recipe_agent, profile_agent, greeting_handler, error_handler,
    feedback_agent, State, set_global_config
)

from data_ingestion_fixed_final import AyurvedicDataIngestion

def build_agent():
    """
    Builds and compiles the enhanced LangGraph agent with comprehensive functionality.
    """
    print("🚀 Building Enhanced Ayurvedic LangGraph Agent...")
    
    # Load all the data and create the knowledge retriever
    try:
        ingestion = AyurvedicDataIngestion()
        datasets, knowledge_retriever = ingestion.load_and_process_all_data()
        
        if not datasets:
            print("⚠️  Warning: Failed to load datasets. Creating agent with limited functionality.")
            datasets = {}
        else:
            print(f"✅ Loaded {len(datasets)} datasets successfully")
            ingestion.get_dataset_info()
        
        if not knowledge_retriever:
            print("⚠️  Warning: Failed to create knowledge base. PDF retrieval may be limited.")
        else:
            print("✅ Knowledge retriever created successfully")
    
    except Exception as e:
        print(f"❌ Error during data loading: {e}")
        print(f"Full error details: {str(e)}")
        datasets = {}
        knowledge_retriever = None
    
    # Set global configuration for agents
    config = {
        "datasets": datasets,
        "knowledge_retriever": knowledge_retriever
    }
    set_global_config(config)
    
    # Create the state graph
    print("🔗 Building conversation flow graph...")
    
    workflow = StateGraph(State)
    
    # Add all agent nodes
    workflow.add_node("route_query", route_query)
    workflow.add_node("profile_agent", profile_agent)
    workflow.add_node("knowledge_agent", knowledge_retrieval_agent)
    workflow.add_node("diet_plan_agent", diet_plan_agent)
    workflow.add_node("recipe_agent", recipe_agent)
    workflow.add_node("greeting_handler", greeting_handler)
    workflow.add_node("feedback_agent", feedback_agent)
    workflow.add_node("error_handler", error_handler)
    
    # Set entry point
    workflow.set_entry_point("route_query")
    
    # Add conditional edges based on routing decisions
    def should_continue(state):
        """Determine which agent to call next based on routing"""
        route_to = state.get("route_to", "error")
        
        # Map routes to agent nodes
        route_mapping = {
            "greeting": "greeting_handler",
            "profile_agent": "profile_agent", 
            "knowledge_agent": "knowledge_agent",
            "diet_plan_agent": "diet_plan_agent",
            "recipe_agent": "recipe_agent",
            "feedback": "feedback_agent",
            "error": "error_handler"
        }
        
        return route_mapping.get(route_to, "error_handler")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "route_query",
        should_continue,
        {
            "greeting_handler": "greeting_handler",
            "profile_agent": "profile_agent",
            "knowledge_agent": "knowledge_agent", 
            "diet_plan_agent": "diet_plan_agent",
            "recipe_agent": "recipe_agent",
            "feedback_agent": "feedback_agent",
            "error_handler": "error_handler"
        }
    )
    
    # All agents end the conversation
    workflow.add_edge("profile_agent", END)
    workflow.add_edge("knowledge_agent", END)
    workflow.add_edge("diet_plan_agent", END)
    workflow.add_edge("recipe_agent", END)
    workflow.add_edge("greeting_handler", END)
    workflow.add_edge("feedback_agent", END)
    workflow.add_edge("error_handler", END)
    
    # Add memory for conversation persistence
    memory = MemorySaver()
    
    # Compile the graph
    try:
        app = workflow.compile(checkpointer=memory)
        print("✅ LangGraph agent compiled successfully!")
        
        # Test the agent with better error handling
        print("🧪 Testing agent functionality...")
        test_config = {"configurable": {"thread_id": "test"}}
        test_input = {
            "messages": [HumanMessage(content="Hello")],
            "user_profile": {},
            "route_to": "",
            "final_response": "",
            "session_id": "test",
            "context": {}
        }
        
        try:
            result = app.invoke(test_input, config=test_config)
            if result.get("final_response"):
                print("✅ Agent test successful!")
            else:
                print("⚠️  Agent test completed but no response generated")
        except Exception as test_error:
            print(f"⚠️  Agent test failed: {test_error}")
            print("This may be normal if data files are missing")
        
        return app
        
    except Exception as e:
        print(f"❌ Error compiling LangGraph: {e}")
        raise e

def get_system_info():
    """Get information about the system capabilities"""
    info = {
        "version": "2.0 Enhanced - Fixed",
        "capabilities": [
            "Comprehensive Ayurvedic profile management",
            "Multi-dataset nutrition analysis", 
            "PDF-based knowledge retrieval",
            "Personalized diet plan generation",
            "Recipe recommendations with Ayurvedic properties",
            "Dosha assessment and constitution analysis",
            "Session persistence and memory",
            "Error handling and fallback responses"
        ],
        "supported_queries": [
            "Profile information and health assessment",
            "Diet plans for various timeframes (days/weeks/months)",
            "Recipe requests with ingredients and methods",
            "Ayurvedic knowledge and principles",
            "Food properties and recommendations",
            "General health and nutrition guidance"
        ],
        "data_sources": [
            "ICMR-NIN Food Composition Tables",
            "UK Food Composition Database",
            "USDA Food Database",
            "Indian Recipe Database (1000+ recipes)",
            "Ayurvedic texts (PDF knowledge base)",
            "Enhanced Ayurvedic food properties"
        ]
    }
    return info

if __name__ == "__main__":
    """Test the graph builder"""
    print("🧪 Testing Enhanced Graph Builder...")
    
    try:
        app = build_agent()
        
        if app:
            print("\n" + "="*50)
            print("🎉 ENHANCED AYURVEDIC AGENT READY!")
            print("="*50)
            
            system_info = get_system_info()
            print(f"\n📊 System Version: {system_info['version']}")
            print(f"\n🔧 Capabilities ({len(system_info['capabilities'])}):")
            for i, capability in enumerate(system_info['capabilities'], 1):
                print(f"   {i}. {capability}")
            
            print(f"\n💬 Supported Query Types ({len(system_info['supported_queries'])}):")
            for i, query_type in enumerate(system_info['supported_queries'], 1):
                print(f"   {i}. {query_type}")
            
            print(f"\n📚 Data Sources ({len(system_info['data_sources'])}):")
            for i, source in enumerate(system_info['data_sources'], 1):
                print(f"   {i}. {source}")
                
            print("\n✨ Ready to use with main_enhanced_fixed.py!")
        else:
            print("❌ Failed to build agent")
            
    except Exception as e:
        print(f"❌ Error in graph builder test: {e}")
        import traceback
        traceback.print_exc()