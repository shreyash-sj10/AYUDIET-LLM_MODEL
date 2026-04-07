import json
from typing import Dict


def build_llm_prompt(request: dict) -> str:
    """
    Build a task-specific prompt while preserving behavior-first safety:
    grounded in input, uncertainty-aware, and non-speculative.
    """
    req_type = request.get("type")

    if req_type == "EXPLAIN_DECISION":
        required = ["decision_trace", "selected_meal", "constraints_applied"]
        if not all(request.get(field) for field in required):
            return json.dumps({"success": False, "data": None, "error": "INSUFFICIENT_CONTEXT"})

        chat_history = request.get("chat_history", "")
        return f"""Explain the system decision clearly to the user.

Conversation Context:
{chat_history}

User Profile:
{request.get("user_profile", "")}

Selected Meal:
{request["selected_meal"]}

Constraints Applied:
{request["constraints_applied"]}

Decision Trace:
{request["decision_trace"]}

Instructions:
- Explain WHY the decision was made
- Break explanation into:
  1. key reasoning
  2. how it relates to user condition
  3. practical meaning
- Keep explanation understandable (not technical)

Do NOT:
- suggest alternative foods
- add new reasoning
- go beyond given trace

Return natural explanation text."""

    if req_type == "KNOWLEDGE_QUERY":
        query = request.get("query", "")
        docs = request.get("retrieved_documents")
        chat_history = request.get("chat_history", "")
        return f"""You are answering a user's question about Ayurveda.

Conversation Context:
{chat_history}

Current Question:
{query}

Instructions:
- If context documents are provided:
  -> Use ONLY those documents
- If no documents are provided:
  -> Use general knowledge
- If this question is a follow-up:
  -> Use previous conversation to maintain continuity

Structure your answer:
1. Start with a clear direct answer
2. Explain key idea in simple terms
3. Add practical meaning or real-world relevance

Guidelines:
- Do NOT give very short answers
- Do NOT repeat generic definitions
- Do NOT ignore previous context
- Avoid giving:
  - specific diet plans
  - specific food prescriptions
- Keep tone: helpful, natural, slightly explanatory

Context Documents:
{docs if docs else "None"}

Return plain text (not JSON)."""

    if req_type == "INTERPRET_SYMPTOMS":
        query = request.get("query", "")
        return get_profiling_prompt(query, request.get("current_profile"))

    return json.dumps({"success": False, "data": None, "error": "INVALID_TYPE"})


def get_profiling_prompt(user_input: str, current_profile: Dict | None = None) -> str:
    profile_json = json.dumps(current_profile or {}, separators=(",", ":"))
    return f"""You are interpreting user-reported symptoms into structured health signals.

Your goal is to extract meaningful patterns WITHOUT over-interpreting.

Return valid JSON matching the schema below.
Do not include any text outside JSON.
{{
  "risk_flags": [string],
  "dosha_estimate": {{
    "vata": number,
    "pitta": number,
    "kapha": number
  }},
  "confidence": number
}}

GUIDELINES:
- Base your output ONLY on the given input
- Do NOT assume diseases unless clearly indicated
- Every output field must be directly traceable to the input symptoms
- If a value cannot be justified from the input, do not include it
- If symptoms are vague (e.g., fatigue, body pain, sleepiness):
  - return general patterns, not specific conditions
  - use "risk_flags": ["none"] if no strong signal exists
- Confidence must reflect clarity of input:
  - strong symptom signals -> higher confidence
  - vague/general symptoms -> low confidence
- Do NOT default to mid-range values without justification
- Dosha estimate:
  - should reflect symptom tendencies
  - if unclear -> keep relatively balanced

IMPORTANT BEHAVIOR:
- If the input does not clearly indicate any specific condition,
  then returning:
    "risk_flags": ["none"]
  is the correct and preferred outcome
- Do NOT try to make the output more informative by guessing
- Absence of signal is a valid result

IMPORTANT:
- You are NOT required to fill all fields with strong signals
- It is acceptable to return uncertainty
- Prefer conservative inference over over-claiming
- Use snake_case strings in risk_flags
- If you output specific risk flags, keep them clinically justified by explicit symptom evidence
- Allow minimal output when evidence is limited

Current profile:
{profile_json}

Example:

Input: "body pain"
Output:
{{
  "risk_flags": ["none"],
  "dosha_estimate": {{ "vata": 0.5, "pitta": 0.3, "kapha": 0.2 }},
  "confidence": 0.3
}}

INPUT:
"{user_input}"

Return valid JSON matching the schema below.
Do not include any text outside JSON."""


def get_rag_explanation_prompt(context: str, decision_trace: str) -> str:
    return f"""You are answering a user's question about Ayurveda.

Conversation Context:
{decision_trace}

Current Question:
Use the latest user question from the conversation context.

Instructions:
- If context documents are provided:
  -> Use ONLY those documents
- If no documents are provided:
  -> Use general knowledge
- If this question is a follow-up:
  -> Use previous conversation to maintain continuity

Structure your answer:
1. Start with a clear direct answer
2. Explain key idea in simple terms
3. Add practical meaning or real-world relevance

Guidelines:
- Do NOT give very short answers
- Target 3-6 sentences unless context is insufficient
- Do NOT repeat generic definitions
- Do NOT ignore previous context
- Avoid giving:
  - specific diet plans
  - specific food prescriptions
- Keep tone: helpful, natural, slightly explanatory

Return valid JSON matching the schema below.
Do not include any text outside JSON.
{{
  "explanation": "string",
  "sources": ["string"]
}}

Context Documents:
{context}

Return valid JSON matching the schema below.
Do not include any text outside JSON."""


def get_feedback_parsing_prompt(feedback_text: str) -> str:
    return f"""Convert feedback into JSON.

Return valid JSON matching the schema below.
Do not include any text outside JSON.
{{
  "feedback_type": "LIKE | DISLIKE | REPLACE",
  "target": "string"
}}

Guidelines:
- Base only on provided text
- If uncertain, prefer DISLIKE
- No extra fields
- No markdown

Feedback:
"{feedback_text}"

Return valid JSON matching the schema below.
Do not include any text outside JSON."""


def build_safe_context(user_input: str, max_length: int = 1000) -> str:
    if not user_input:
        return ""
    text = str(user_input).strip()[:max_length]
    for pattern in ("```", "<json>", "</json>", "ignore instructions", "override"):
        text = text.replace(pattern, "")
    return text.strip()


PROMPT_TEMPLATES = {
    "profiling": get_profiling_prompt,
    "feedback": get_feedback_parsing_prompt,
    "rag_explanation": get_rag_explanation_prompt,
}
