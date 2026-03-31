import json
from typing import Dict


def get_profiling_prompt(user_input: str, current_profile: Dict | None = None) -> str:
    profile_json = json.dumps(current_profile or {}, separators=(",", ":"))
    return f"""Convert input into structured health signals.

Return STRICT JSON:
{{
  \"risk_flags\": [valid enum only],
  \"dosha_estimate\": {{
    \"vata\": number (0-1),
    \"pitta\": number (0-1),
    \"kapha\": number (0-1)
  }},
  \"confidence\": number (0-1)
}}

Rules:
- No explanation
- No extra text
- Only known risk flags
- Dosha values must sum to ~1
- If uncertain -> lower confidence

Calibration:
- clear symptoms -> confidence > 0.7
- vague input -> confidence 0.4-0.6
- unclear -> confidence < 0.4

Valid risk_flags enum:
- high_blood_pressure
- diabetes
- kidney_disease
- liver_disease
- heart_condition
- thyroid_disorder
- allergy
- intolerance
- none

Current profile: {profile_json}
User input: \"{user_input}\"

Respond ONLY JSON."""


def get_rag_explanation_prompt(context: str, decision_trace: str) -> str:
    return f"""Generate explanation using ONLY the provided context.

Rules:
- Do NOT add new information
- Do NOT hallucinate
- Do NOT infer beyond context
- Keep explanation concise
- Include citation style in explanation text: (Source: Charaka Samhita, Chapter X)

Output JSON:
{{
  \"explanation\": \"...\",
  \"sources\": [\"...\"]
}}

Context:
{context}

Decision:
{decision_trace}

Respond ONLY JSON."""


def get_feedback_parsing_prompt(feedback_text: str) -> str:
    return f"""Convert feedback into structured JSON:
{{
  \"feedback_type\": \"LIKE | DISLIKE | REPLACE\",
  \"target\": \"string\"
}}

Rules:
- No explanation
- No extra text
- If unclear -> DISLIKE

Feedback:
\"{feedback_text}\"

Respond ONLY JSON."""


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
