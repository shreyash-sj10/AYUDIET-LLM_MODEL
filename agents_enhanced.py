# Enhanced Ayurvedic Agents System
# File: agents_enhanced.py

import os
import re
import pandas as pd
from typing import Annotated, TypedDict, Optional, Dict, Any, List
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from llm_strict_wrapper import StrictLLMWrapper
from deterministic_engine import DeterministicDecisionEngine

# Enhanced UserProfile with comprehensive Ayurvedic fields
class UserProfile(BaseModel):
    # Basic Information
    name: Optional[str] = Field(default=None, description="User's name")
    age: Optional[int] = Field(default=None, description="User's age in years")
    gender: Optional[str] = Field(default=None, description="User's gender")
    weight: Optional[float] = Field(default=None, description="Weight in kg")
    height: Optional[str] = Field(default=None, description="Height in cm or feet")
    
    # Health & Lifestyle
    activity_level: Optional[str] = Field(default=None, description="sedentary/moderate/active/very active")
    health_conditions: Optional[List[str]] = Field(default=[], description="Any health issues")
    medications: Optional[List[str]] = Field(default=[], description="Current medications")
    allergies: Optional[List[str]] = Field(default=[], description="Food allergies/intolerances")
    
    # Dietary Preferences
    dietary_preferences: Optional[List[str]] = Field(default=[], description="vegetarian/vegan/non-veg")
    food_dislikes: Optional[List[str]] = Field(default=[], description="Foods to avoid")
    meal_frequency: Optional[int] = Field(default=3, description="Meals per day")
    
    # Ayurvedic Assessment
    primary_dosha: Optional[str] = Field(default=None, description="vata/pitta/kapha")
    secondary_dosha: Optional[str] = Field(default=None, description="Secondary dosha if any")
    constitution: Optional[str] = Field(default=None, description="Prakriti assessment")
    current_imbalance: Optional[str] = Field(default=None, description="Vikriti - current imbalance")
    
    # Goals and Preferences
    health_goals: Optional[List[str]] = Field(default=[], description="Weight loss/gain/maintenance/health")
    preferred_cuisine: Optional[List[str]] = Field(default=[], description="Indian/South Indian/North Indian etc")
    cooking_skill: Optional[str] = Field(default="intermediate", description="beginner/intermediate/expert")
    
    # Digestive Health (Important for Ayurveda)
    digestion_strength: Optional[str] = Field(default=None, description="strong/moderate/weak")
    bowel_movement: Optional[str] = Field(default=None, description="regular/irregular/constipated")
    water_intake: Optional[str] = Field(default=None, description="Daily water intake")
    sleep_pattern: Optional[str] = Field(default=None, description="Sleep quality and timing")

class State(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    user_profile: Annotated[Dict, "The user's extracted profile information"]
    route_to: Annotated[str, "The agent to route to next"]
    final_response: Annotated[str, "The final response to the user"]
    session_id: Annotated[str, "Session identifier for persistence"]
    context: Annotated[Dict, "Additional context for agents"]

# Global configuration - will be set by the graph builder
GLOBAL_CONFIG = {}
_LLM_INSTANCE = None
_LLM_MODEL = None
_STRICT_WRAPPER = None
_DECISION_ENGINE = None

def set_global_config(config):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = config

# Persistent profile storage (In production, use database)
PROFILE_STORAGE = {}

# Enhanced LLM initialization
def get_llm():
    """Get a working Groq LLM instance with model fallback support."""
    global _LLM_INSTANCE, _LLM_MODEL

    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error initializing Groq LLM: GROQ_API_KEY is missing")
        return None

    preferred_model = os.getenv("GROQ_MODEL", "").strip()
    candidate_models = []

    if preferred_model:
        candidate_models.append(preferred_model)

    # Common Groq chat models to try if the preferred model is not accessible.
    fallback_models = [
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
    ]

    for model_name in fallback_models:
        if model_name not in candidate_models:
            candidate_models.append(model_name)

    for model_name in candidate_models:
        try:
            llm = ChatGroq(
                model=model_name,
                temperature=float(os.getenv("TEMPERATURE", "0.1")),
                api_key=api_key
            )
            _LLM_INSTANCE = llm
            _LLM_MODEL = model_name
            print(f"Using Groq model: {model_name}")
            return _LLM_INSTANCE
        except Exception as e:
            print(f"Groq model unavailable ({model_name}): {e}")

    print("Error initializing Groq LLM: no accessible Groq models were available")
    return None


def get_strict_wrapper() -> Optional[StrictLLMWrapper]:
    """Return a cached strict wrapper bound to the configured LLM."""
    global _STRICT_WRAPPER

    if _STRICT_WRAPPER is not None:
        return _STRICT_WRAPPER

    llm = get_llm()
    if llm is None:
        return None

    _STRICT_WRAPPER = StrictLLMWrapper(llm)
    return _STRICT_WRAPPER


def get_decision_engine() -> DeterministicDecisionEngine:
    global _DECISION_ENGINE
    if _DECISION_ENGINE is None:
        _DECISION_ENGINE = DeterministicDecisionEngine()
    return _DECISION_ENGINE

# ROUTING AGENT - Enhanced query classification
def route_query(state: State) -> State:
    """Enhanced routing agent with better query classification"""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Initialize session if needed
    session_id = state.get("session_id", "default")
    if session_id not in PROFILE_STORAGE:
        PROFILE_STORAGE[session_id] = {}
    
    state["user_profile"] = PROFILE_STORAGE[session_id]
    
    # Comprehensive routing logic
    query_lower = last_message.lower().strip()
    
    # Greeting patterns
    greeting_patterns = [
        r'\b(hi|hello|hey|good morning|good afternoon|good evening|namaste)\b',
        r'^(hi|hello|hey)$'
    ]
    
    # Profile patterns
    profile_patterns = [
        r'\b(my name is|i am|i\'m)\b',
        r'\b(age|years old|weight|height)\b',
        r'\b(suffer from|have|condition|disease|diabetes|hypertension)\b',
        r'\b(allergic to|allergy|intolerant)\b',
        r'\b(vegetarian|vegan|non-veg)\b'
    ]
    
    # Diet plan patterns
    diet_patterns = [
        r'\b(diet plan|diet chart|meal plan|nutrition plan)\b',
        r'\b(lose weight|gain weight|weight loss|weight gain)\b',
        r'\b(what should i eat|food recommendation|meal suggestion)\b',
        r'\b(plan for|days?|weeks?|month)\b'
    ]
    
    # Recipe patterns
    recipe_patterns = [
        r'\b(recipe|how to make|preparation|cook|cooking)\b',
        r'\b(ingredients|steps|method|procedure)\b'
    ]
    
    # Knowledge patterns
    knowledge_patterns = [
        r'\b(what is|tell me about|explain|ayurveda|dosha)\b',
        r'\b(vata|pitta|kapha|constitution|prakriti)\b',
        r'\b(properties|benefits|effects|good for)\b'
    ]
    
    # Route based on patterns
    if any(re.search(pattern, query_lower) for pattern in greeting_patterns):
        state["route_to"] = "greeting"
    elif any(re.search(pattern, query_lower) for pattern in profile_patterns):
        state["route_to"] = "profile_agent"
    elif any(re.search(pattern, query_lower) for pattern in diet_patterns):
        state["route_to"] = "diet_plan_agent"
    elif any(re.search(pattern, query_lower) for pattern in recipe_patterns):
        state["route_to"] = "recipe_agent"
    elif any(re.search(pattern, query_lower) for pattern in knowledge_patterns):
        state["route_to"] = "knowledge_agent"
    else:
        # Default to knowledge for general queries
        state["route_to"] = "knowledge_agent"
    
    print(f"🎯 Routing to: {state['route_to']}")
    return state

# PROFILE AGENT - Enhanced profile management
def profile_agent(state: State) -> State:
    """Profile agent backed by strict wrapper and deterministic field extraction."""
    messages = state["messages"]
    session_id = state.get("session_id", "default")
    current_profile = state.get("user_profile", {})

    wrapper = get_strict_wrapper()
    if not wrapper:
        state["final_response"] = "I'm having technical difficulties. Please try again."
        return state

    user_text = messages[-1].content if messages else ""

    try:
        strict_result = wrapper.extract_health_profile(user_text, current_profile=current_profile)
        strict_profile = strict_result.get("data", {})
        deterministic_profile = _extract_basic_profile_fields(user_text)

        updated_profile = {**current_profile}
        for key, value in deterministic_profile.items():
            if value is not None and value != [] and value != "":
                updated_profile[key] = value

        risk_flags = strict_profile.get("risk_flags", [])
        dosha = strict_profile.get("dosha_estimate", {})
        if risk_flags:
            updated_profile["health_conditions"] = sorted(
                list(set(updated_profile.get("health_conditions", []) + risk_flags))
            )
        if dosha:
            updated_profile["dosha_scores"] = dosha
            updated_profile["primary_dosha"] = max(dosha, key=dosha.get)
            updated_profile["constitution"] = updated_profile["primary_dosha"]

        if updated_profile.get("age") and updated_profile.get("weight") and not updated_profile.get("dosha_scores"):
            updated_profile.update(assess_dosha_comprehensive(updated_profile))

        PROFILE_STORAGE[session_id] = updated_profile
        state["user_profile"] = updated_profile

        profile_summary = generate_profile_summary(updated_profile)
        confidence = float(strict_profile.get("confidence", 0.2))
        confidence_label = "high" if confidence >= 0.7 else "moderate" if confidence >= 0.45 else "low"

        state["final_response"] = f"""Thank you for sharing that information! I've updated your profile.

{profile_summary}

Profile confidence: {confidence_label} ({confidence:.2f})

I can now provide you with personalized Ayurvedic recommendations. You can ask me for:
- Personalized diet plans
- Recipe suggestions
- Ayurvedic advice
- Food recommendations based on your constitution

What would you like to know?"""

    except Exception as e:
        print(f"Error in profile agent: {e}")
        state["final_response"] = "I've noted your information. What else can I help you with regarding your Ayurvedic diet and health?"

    return state


def _extract_basic_profile_fields(user_text: str) -> Dict[str, Any]:
    """Extract deterministic profile fields from plain user text."""
    text = (user_text or "").strip()
    lower = text.lower()
    updates: Dict[str, Any] = {}

    age_match = re.search(r"\b(?:age\s*[:=]?\s*)?(\d{1,2})\s*(?:years?|yrs?)\b", lower)
    if age_match:
        updates["age"] = int(age_match.group(1))

    weight_match = re.search(r"\b(?:weight\s*[:=]?\s*)?(\d{2,3}(?:\.\d+)?)\s*(?:kg|kgs)\b", lower)
    if weight_match:
        updates["weight"] = float(weight_match.group(1))

    height_match = re.search(r"\b(?:height\s*[:=]?\s*)?(\d{2,3})\s*cm\b", lower)
    if height_match:
        updates["height"] = f"{height_match.group(1)} cm"

    if "vegetarian" in lower:
        updates["dietary_preferences"] = ["vegetarian"]
    elif "vegan" in lower:
        updates["dietary_preferences"] = ["vegan"]
    elif "non-veg" in lower or "non veg" in lower:
        updates["dietary_preferences"] = ["non-veg"]

    allergies = []
    allergy_match = re.search(r"allergic to ([a-z,\s]+)", lower)
    if allergy_match:
        allergies = [item.strip() for item in allergy_match.group(1).split(",") if item.strip()]
    if allergies:
        updates["allergies"] = allergies

    if "weight loss" in lower or "lose weight" in lower:
        updates["health_goals"] = ["weight_loss"]
    elif "weight gain" in lower or "gain weight" in lower:
        updates["health_goals"] = ["weight_gain"]

    return updates

# ENHANCED DOSHA ASSESSMENT
def assess_dosha_comprehensive(profile: Dict) -> Dict:
    """Comprehensive Dosha assessment based on multiple factors"""
    try:
        age = profile.get("age", 25)
        weight = profile.get("weight", 65)
        height_str = profile.get("height", "165")
        activity = profile.get("activity_level", "moderate")
        digestion = profile.get("digestion_strength", "moderate")
        
        # Calculate BMI
        height_cm = extract_height_in_cm(height_str)
        bmi = weight / ((height_cm / 100) ** 2) if height_cm > 0 else 22
        
        # Dosha scoring based on multiple factors
        vata_score = 0
        pitta_score = 0
        kapha_score = 0
        
        # BMI-based assessment
        if bmi < 18.5:
            vata_score += 3
        elif 18.5 <= bmi < 25:
            pitta_score += 2
            vata_score += 1
        else:
            kapha_score += 3
        
        # Age-based assessment
        if age < 30:
            pitta_score += 2
        elif age < 50:
            pitta_score += 1
            vata_score += 1
        else:
            vata_score += 2
        
        # Activity level assessment
        if activity == "sedentary":
            kapha_score += 2
        elif activity == "very_active":
            vata_score += 2
        else:
            pitta_score += 1
        
        # Digestion assessment
        if digestion == "strong":
            pitta_score += 2
        elif digestion == "weak":
            vata_score += 2
        else:
            kapha_score += 1
        
        # Determine primary and secondary dosha
        scores = {"vata": vata_score, "pitta": pitta_score, "kapha": kapha_score}
        sorted_doshas = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_dosha = sorted_doshas[0][0]
        secondary_dosha = sorted_doshas[1][0] if sorted_doshas[1][1] > 1 else None
        
        # Generate constitution description
        if secondary_dosha and scores[primary_dosha] - scores[secondary_dosha] <= 1:
            constitution = f"{primary_dosha}-{secondary_dosha}"
        else:
            constitution = primary_dosha
        
        return {
            "primary_dosha": primary_dosha,
            "secondary_dosha": secondary_dosha,
            "constitution": constitution,
            "dosha_scores": scores
        }
        
    except Exception as e:
        print(f"Error in dosha assessment: {e}")
        return {"primary_dosha": "pitta", "constitution": "pitta"}

def extract_height_in_cm(height_str: str) -> float:
    """Extract height in centimeters from various formats"""
    try:
        height_str = str(height_str).strip().lower()
        
        # If already in cm
        if "cm" in height_str:
            return float(re.findall(r'\d+', height_str)[0])
        
        # If in feet and inches (e.g., "5'6", "5 feet 6 inches")
        feet_inches = re.findall(r'(\d+)[\'\']\s*(\d+)', height_str)
        if feet_inches:
            feet, inches = map(int, feet_inches[0])
            return (feet * 30.48) + (inches * 2.54)
        
        # If just feet (e.g., "5.5")
        if "." in height_str:
            feet = float(re.findall(r'\d+\.?\d*', height_str)[0])
            return feet * 30.48
        
        # Default assume cm if just number
        return float(height_str)
        
    except:
        return 165  # Default height

def generate_profile_summary(profile: Dict) -> str:
    """Generate a friendly profile summary"""
    summary_parts = []
    
    if profile.get("name"):
        summary_parts.append(f"Name: {profile['name']}")
    
    if profile.get("age"):
        summary_parts.append(f"Age: {profile['age']} years")
    
    if profile.get("primary_dosha"):
        constitution = profile.get("constitution", profile["primary_dosha"])
        summary_parts.append(f"Constitution (Prakriti): {constitution.title()}")
    
    if profile.get("health_goals"):
        goals = ", ".join(profile["health_goals"])
        summary_parts.append(f"Health Goals: {goals}")
    
    if profile.get("dietary_preferences"):
        diet = ", ".join(profile["dietary_preferences"])
        summary_parts.append(f"Dietary Preference: {diet}")
    
    return "📋 **Your Profile Summary:**\n" + "\n".join([f"• {part}" for part in summary_parts])

# KNOWLEDGE RETRIEVAL AGENT - Enhanced with fallback
def knowledge_retrieval_agent(state: State) -> State:
    """Knowledge agent using strict explanation generation from retrieved context."""
    messages = state["messages"]
    query = messages[-1].content

    knowledge_retriever = GLOBAL_CONFIG.get("knowledge_retriever")
    wrapper = get_strict_wrapper()

    if not wrapper:
        state["final_response"] = "I'm experiencing technical difficulties. Please try again."
        return state

    try:
        if knowledge_retriever:
            try:
                relevant_docs = knowledge_retriever.invoke(query)

                if relevant_docs and len(relevant_docs) > 0:
                    context_chunks = [doc.page_content for doc in relevant_docs[:3]]
                    result = wrapper.generate_explanation(context_chunks, query)
                    data = result.get("data", {})
                    explanation = str(data.get("explanation", "")).strip()
                    if explanation:
                        state["final_response"] = f"Based on Ayurvedic Texts:\n\n{explanation}"
                        return state

            except Exception as e:
                print(f"Error retrieving from knowledge base: {e}")

        state["final_response"] = (
            "I could not find grounded reference chunks for that question right now. "
            "Please rephrase with a specific herb, recipe, or condition so I can answer from retrieved texts only."
        )

    except Exception as e:
        print(f"Error in knowledge retrieval: {e}")
        state["final_response"] = "I apologize, but I'm having difficulty accessing information right now. Please try again."

    return state

# DIET PLAN AGENT - Professional diet planning
def diet_plan_agent(state: State) -> State:
    """Deterministic diet plan agent using constraint-first decision engine."""
    messages = state["messages"]
    user_profile = state.get("user_profile", {})
    query = messages[-1].content
    datasets = GLOBAL_CONFIG.get("datasets", {})

    try:
        engine = get_decision_engine()
        session_id = state.get("session_id", "default")
        decision = engine.recommend_meal(
            session_id=session_id,
            user_profile=user_profile,
            query=query,
            datasets=datasets,
            template_id="lunch_roti_dal_sabzi",
        )

        meal = decision.meal
        trace = decision.trace
        selected_lines = [f"- {slot}: {food}" for slot, food in meal.items()]
        relax_line = decision.relaxation_level_used or "None"

        state["context"]["latest_decision_trace"] = decision.model_dump()
        state["final_response"] = (
            "Deterministic Meal Recommendation\n\n"
            f"Template: {decision.template_id}\n"
            f"Confidence: {decision.confidence:.2f}\n"
            f"Fallback Used: {decision.fallback_used}\n"
            f"Relaxation Level Used: {relax_line}\n\n"
            "Selected Meal:\n"
            + "\n".join(selected_lines)
            + "\n\nTop Optimization Steps:\n"
            + "\n".join([f"- {step}" for step in trace.optimization_steps[:6]])
        )

    except Exception as e:
        print(f"Error in diet plan generation: {e}")
        state["final_response"] = "I'd be happy to create a diet plan for you! Could you share more about your preferences and goals?"

    return state


def calculate_daily_calories(profile: Dict) -> int:
    """Calculate daily caloric needs using Mifflin-St Jeor equation"""
    try:
        age = profile.get("age", 25)
        weight = profile.get("weight", 65)
        height_str = profile.get("height", "165")
        gender = profile.get("gender", "female").lower()
        activity = profile.get("activity_level", "moderate")
        goals = profile.get("health_goals", ["maintenance"])
        
        # Extract height in cm
        height_cm = extract_height_in_cm(height_str)
        
        # Calculate BMR
        if gender in ["male", "man", "m"]:
            bmr = 10 * weight + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height_cm - 5 * age - 161
        
        # Activity multiplier
        activity_multipliers = {
            "sedentary": 1.2,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9
        }
        
        multiplier = activity_multipliers.get(activity, 1.55)
        tdee = bmr * multiplier
        
        # Adjust for goals
        if "weight_loss" in goals:
            return int(tdee - 500)  # 500 cal deficit
        elif "weight_gain" in goals:
            return int(tdee + 500)  # 500 cal surplus
        else:
            return int(tdee)
            
    except Exception as e:
        print(f"Error calculating calories: {e}")
        return 2000  # Default

def get_dosha_appropriate_foods(dosha: str, datasets: Dict) -> str:
    """Get foods appropriate for the primary dosha"""
    try:
        foods_list = []
        
        # Check different datasets for dosha-appropriate foods
        for dataset_name, df in datasets.items():
            if df is not None and not df.empty:
                # Look for Ayurvedic columns
                dosha_columns = [col for col in df.columns if dosha in col.lower()]
                
                if dosha_columns:
                    # Filter foods good for this dosha
                    good_foods = df[df[dosha_columns[0]].str.contains("balancing|good", case=False, na=False)]
                    
                    if not good_foods.empty and 'food_name' in df.columns:
                        foods_list.extend(good_foods['food_name'].head(10).tolist())
        
        if foods_list:
            return f"Recommended {dosha}-balancing foods: " + ", ".join(foods_list[:20])
        else:
            # Default recommendations
            dosha_foods = {
                "vata": "warm cooked foods, rice, ghee, sesame oil, sweet fruits, root vegetables",
                "pitta": "cooling foods, coconut, cilantro, cucumber, sweet fruits, leafy greens",
                "kapha": "light warm foods, spices, ginger, garlic, bitter vegetables, legumes"
            }
            return f"Recommended {dosha}-balancing foods: {dosha_foods.get(dosha, 'balanced nutritious foods')}"
            
    except Exception as e:
        print(f"Error getting dosha foods: {e}")
        return "Balanced nutritious foods from all food groups"

def extract_timeframe(query: str) -> int:
    """Extract number of days from query"""
    try:
        # Look for number + days/weeks/months
        day_patterns = [
            r'(\d+)\s*days?',
            r'(\d+)\s*day',
        ]
        
        week_patterns = [
            r'(\d+)\s*weeks?',
            r'(\d+)\s*week',
        ]
        
        month_patterns = [
            r'(\d+)\s*months?',
            r'(\d+)\s*month',
        ]
        
        query_lower = query.lower()
        
        # Check for days
        for pattern in day_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return int(match.group(1))
        
        # Check for weeks
        for pattern in week_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return int(match.group(1)) * 7
        
        # Check for months
        for pattern in month_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return int(match.group(1)) * 30
        
        # Default timeframes
        if "week" in query_lower:
            return 7
        elif "month" in query_lower:
            return 30
        else:
            return 7  # Default 1 week
            
    except Exception:
        return 7

# RECIPE AGENT - Enhanced recipe recommendations
def recipe_agent(state: State) -> State:
    """Deterministic recipe agent from curated dataset lookups only."""
    messages = state["messages"]
    user_profile = state.get("user_profile", {})
    query = messages[-1].content

    datasets = GLOBAL_CONFIG.get("datasets", {})

    try:
        recipe_text = get_recipe_details(query, datasets, user_profile)
        state["final_response"] = f"Recipe Recommendation\n\n{recipe_text}"

    except Exception as e:
        print(f"Error in recipe agent: {e}")
        state["final_response"] = "I'd love to help you with recipes! Could you tell me what dish you're interested in?"

    return state


def get_recipe_details(query: str, datasets: Dict, user_profile: Dict) -> str:
    """Get recipe details from available datasets"""
    try:
        recipe_details = []
        
        # Look for recipes in datasets
        recipes_df = datasets.get("recipes")
        recipe_names_df = datasets.get("recipes_names")
        
        if recipes_df is not None and recipe_names_df is not None:
            # Simple search in recipe names
            query_lower = query.lower()
            
            if 'recipe_name' in recipe_names_df.columns:
                matching_recipes = recipe_names_df[
                    recipe_names_df['recipe_name'].str.contains(query_lower, case=False, na=False)
                ].head(3)
                
                if not matching_recipes.empty:
                    recipe_details.append("Found matching recipes:")
                    for _, recipe in matching_recipes.iterrows():
                        recipe_details.append(f"- {recipe['recipe_name']}")
        
        # Add general recipe guidance
        dosha = user_profile.get("primary_dosha", "pitta")
        recipe_details.append(f"\nRecommended for {dosha} constitution")
        
        return "\n".join(recipe_details) if recipe_details else "General recipe guidance based on Ayurvedic principles"
        
    except Exception as e:
        print(f"Error getting recipe details: {e}")
        return "Recipe guidance based on Ayurvedic principles"

# GREETING HANDLER
def greeting_handler(state: State) -> State:
    """Handle greetings and general conversation"""
    user_profile = state.get("user_profile", {})
    
    if user_profile.get("name"):
        greeting = f"Hello {user_profile['name']}! 🙏"
    else:
        greeting = "Hello! 🙏 Welcome to your Ayurvedic Diet Assistant!"
    
    response = f"""{greeting}
    
I'm here to help you with personalized Ayurvedic nutrition and diet planning. I can assist you with:

🔹 **Personalized Diet Plans** - Based on your constitution (Dosha)
🔹 **Recipe Recommendations** - Traditional and modern Ayurvedic recipes  
🔹 **Nutritional Guidance** - Food properties and combinations
🔹 **Health Consultations** - Ayurvedic principles for wellness

To get started, you can:
• Tell me about yourself (age, health goals, preferences)
• Ask for a diet plan
• Request specific recipes
• Ask questions about Ayurveda

What would you like to explore today?"""
    
    state["final_response"] = response
    return state

# ERROR HANDLER
def error_handler(state: State) -> State:
    """Handle errors and provide helpful responses"""
    state["final_response"] = """I apologize, but I encountered an issue processing your request. 

Please try:
• Asking your question in a different way
• Being more specific about what you need
• Starting with basic information like "I want a diet plan" or "Tell me about..."

I'm here to help with your Ayurvedic nutrition journey! 🌿"""
    
    return state

# FEEDBACK AND CLARIFICATION AGENT
def feedback_agent(state: State) -> State:
    """Handle feedback by using strict feedback parsing and deterministic replies."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    wrapper = get_strict_wrapper()
    if not wrapper:
        state["final_response"] = "I'm here to help! Could you please rephrase your question?"
        return state

    try:
        result = wrapper.parse_feedback(last_message)
        feedback = result.get("data", {})
        feedback_type = str(feedback.get("feedback_type", "DISLIKE")).upper()
        target = str(feedback.get("target", "")).strip()

        if feedback_type == "LIKE":
            state["final_response"] = "Happy that helped. Tell me what you want next, and I will build on it."
        elif feedback_type == "REPLACE":
            focus = f" for '{target}'" if target else ""
            state["final_response"] = (
                f"Understood. I will replace the previous recommendation{focus}. "
                "Share any preferences (budget, taste, time, medical constraints) so I can tailor it."
            )
        else:
            focus = f" about '{target}'" if target else ""
            state["final_response"] = (
                f"Thanks for the feedback{focus}. I will correct the direction. "
                "Tell me one thing you want changed most, and I will update the plan."
            )
    except Exception as e:
        print(f"Error in feedback agent: {e}")
        state["final_response"] = "I want to make sure I'm providing exactly what you need. Tell me what to change first."

    return state


