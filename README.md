# AYUDIET LLM Service and Chatbot Backend

AYUDIET is an Ayurveda-focused backend with two main runtimes:
- A strict FastAPI service (`main.py`) for structured Node.js/backend integration.
- A multi-agent chatbot runtime (`main_enhanced.py`, `agents_enhanced.py`) for richer diet and nutrition conversations.

## What this repo provides

### 1) FastAPI service (`main.py`)
Production endpoints:
- `GET /health`
- `POST /profile`
- `POST /explain`

Key behavior:
- strict response envelope: `success`, `data`, `error`
- schema validation on LLM output
- timeout + safe fallback responses
- request/error logging
- optional API-key and rate-limiting for protected routes

### 2) Enhanced chatbot runtime
Primary files:
- `main_enhanced.py`
- `graph_builder_enhanced.py`
- `agents_enhanced.py`
- `app.py` (Flask API wrapper)

This runtime supports personalized diet/meal guidance based on profile + dosha context.

### 3) LLM safety/control modules
- `llm_schemas_strict.py`
- `llm_prompts_strict.py`
- `llm_hallucination_control.py`
- `llm_confidence_calibration.py`
- `llm_strict_wrapper.py`

## Can this model suggest diet plans?
Yes, via the chatbot runtime (`main_enhanced.py` or Flask `/chat` flow).

Important:
- The strict FastAPI `/explain` route is intentionally constrained and does not return direct meal/diet recommendations.
- For diet planning conversations, use the enhanced chatbot pipeline.

## Tech stack
- Python 3.13+
- FastAPI + Uvicorn
- Flask (legacy/chat API path)
- LangChain / LangGraph
- Groq LLM
- FAISS + HuggingFace embeddings

## Setup

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Configure environment
Create `.env` in project root:
```env
GROQ_API_KEY=your_groq_key
HF_TOKEN=your_hf_token
GROQ_MODEL=llama-3.1-8b-instant
LLM_TIMEOUT_SECONDS=12
AYUDIET_API_KEY=optional_shared_secret
RATE_LIMIT_PER_MINUTE=60
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

## Run

### FastAPI (recommended for backend integration)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Flask API (legacy/chat wrapper)
```bash
python app.py
```

### Interactive enhanced chatbot
```bash
python main_enhanced.py
```

## FastAPI request examples

### Health
```bash
curl http://localhost:8000/health
```

### Profile
```bash
curl -X POST http://localhost:8000/profile \
  -H "Content-Type: application/json" \
  -d "{\"symptoms\":\"I have acidity and burning sensation\"}"
```

### Explain
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d "{\"context\":\"Pitta is associated with heat.\",\"reasoning\":\"user has heat symptoms\"}"
```

If `AYUDIET_API_KEY` is set, add:
```bash
-H "X-API-Key: your_secret"
```

## Notes
- If LLM output is invalid or timeout occurs, safe fallback JSON is returned.
- Keep `.env` private and never commit secrets.
- Data files under `data/` can be large; manage repository size carefully.

## License
Add your preferred license before public release.
